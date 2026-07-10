"""Result records, run manifest, JSONL writer, workspace archiving, telemetry.

One row per task-trial (:class:`TrialRecord`) is the study's atomic datum; the
report layer aggregates ``records.jsonl`` across runs into paper-ready tables.
Records are append-only and resumable: a runner restarted with an existing
``run_id`` skips task-trials already present (see :meth:`RecordWriter.done`).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

logger = logging.getLogger("pandabench.results")

__all__ = [
    "SCHEMA_VERSION",
    "HarnessTelemetry",
    "RecordWriter",
    "RunManifest",
    "TrialRecord",
    "archive_workspace",
    "collect_harness_telemetry",
    "resume_key",
]

SCHEMA_VERSION = 1

# The tuple identifying a task-trial for resumability + label joins. Must be
# stable across process restarts.
ResumeKey = tuple[str, str, str, str, str, int, int, str]


@dataclass(frozen=True, slots=True)
class HarnessTelemetry:
    """Per-session harness state at the end of a trial (arm B only)."""

    session_id: str
    reliability: float | None
    consistency: float | None
    breached: bool
    rules_active: int
    rules_candidate: int
    rules_retired: int
    notices: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One task-trial outcome. Round-trippable via to_json / from_json."""

    run_id: str
    benchmark: str
    task_id: str
    arm: str
    model: str
    provider: str
    backend: str | None
    resolved_model: str
    seed: int
    trial: int
    phase: str  # "learning" | "eval"
    passed: bool
    native_metrics: dict[str, Any]
    turns: int
    wall_time_s: float
    usage: dict[str, Any]
    harness: dict[str, Any] | None
    error: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TrialRecord:
        fields = {
            "run_id", "benchmark", "task_id", "arm", "model", "provider", "backend",
            "resolved_model", "seed", "trial", "phase", "passed", "native_metrics",
            "turns", "wall_time_s", "usage", "harness", "error", "schema_version",
        }
        return cls(**{k: v for k, v in data.items() if k in fields})

    @property
    def resume_key(self) -> ResumeKey:
        return resume_key(
            self.benchmark, self.task_id, self.arm, self.model,
            self.backend, self.seed, self.trial, self.phase,
        )


def resume_key(
    benchmark: str, task_id: str, arm: str, model: str,
    backend: str | None, seed: int, trial: int, phase: str,
) -> ResumeKey:
    """The dedup identity of a task-trial (backend normalized to '')."""

    return (benchmark, task_id, arm, model, backend or "", seed, trial, phase)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Full resolved provenance for a run, written to ``manifest.json``."""

    run_id: str
    benchmark: str
    model: str
    arm: str
    seed: int
    backend: str | None
    started_at: str
    git_sha: str
    uv_lock_hash: str
    pandaprobe_harness_version: str
    litellm_version: str
    resolved_config: dict[str, Any]
    env_fingerprint: dict[str, Any]
    learning_outcome: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")


class RecordWriter:
    """Append-only JSONL writer with in-memory dedup for resumability."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[ResumeKey] = set()
        if self._path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._done.add(TrialRecord.from_json(json.loads(line)).resume_key)
            except Exception as exc:  # noqa: BLE001 - tolerate a partial last line
                logger.warning("skipping unreadable record line: %s", exc)

    def done(self, key: ResumeKey) -> bool:
        return key in self._done

    def append(self, record: TrialRecord) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_json()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._done.add(record.resume_key)

    @property
    def count(self) -> int:
        return len(self._done)


# -- workspace archiving ------------------------------------------------------

# The harness workspace artifacts worth keeping (telemetry gold). Transient
# atomic-write temp files (*.tmp) are skipped.
_ARCHIVE_ENTRIES = (
    "rules.jsonl",
    "journal.jsonl",
    "harness_rules.md",
    "evalset",
    "mailbox",
    "state",
    "traces",
)


def archive_workspace(harness_root: Path, dest: Path) -> None:
    """Copy the durable harness workspace into ``dest`` (for the run's ``harness/``)."""

    dest.mkdir(parents=True, exist_ok=True)
    for entry in _ARCHIVE_ENTRIES:
        src = harness_root / entry
        if not src.exists():
            continue
        target = dest / entry
        if src.is_dir():
            shutil.copytree(
                src, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp")
            )
        else:
            shutil.copy2(src, target)


# -- telemetry ----------------------------------------------------------------


def collect_harness_telemetry(
    harness: Any, session_id: str, report: Any | None
) -> HarnessTelemetry:
    """Best-effort harness state for a session (never raises — telemetry only)."""

    reliability = consistency = None
    breached = False
    if report is not None:
        try:
            breached = bool(report.any_breach)
            for score in report.scores:
                name = str(score.metric)
                if name == "agent_reliability":
                    reliability = score.value
                elif name == "agent_consistency":
                    consistency = score.value
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: report parse failed: %s", exc)

    active = candidate = retired = 0
    try:
        for rule in harness.rules.all():
            status = getattr(rule, "status", "")
            active += status == "active"
            candidate += status == "candidate"
            retired += status == "retired"
    except Exception as exc:  # noqa: BLE001
        logger.debug("telemetry: rules read failed: %s", exc)

    notices = _count_session_notices(harness, session_id)

    return HarnessTelemetry(
        session_id=session_id,
        reliability=reliability,
        consistency=consistency,
        breached=breached,
        rules_active=active,
        rules_candidate=candidate,
        rules_retired=retired,
        notices=notices,
    )


def _count_session_notices(harness: Any, session_id: str) -> int:
    """Count 'notice' journal events for this session (falls back to 0)."""

    try:
        events = harness.journal.recent(limit=10_000, types=("notice",))
    except Exception:  # noqa: BLE001
        return 0
    count = 0
    for event in events:
        sid = event.get("session_id")
        if sid is None:
            payload = event.get("payload") or event.get("notice") or {}
            sid = payload.get("session_id") if isinstance(payload, dict) else None
        if sid == session_id:
            count += 1
    return count


# -- provenance helpers -------------------------------------------------------


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def uv_lock_hash(lock_path: Path) -> str:
    import hashlib

    try:
        return hashlib.sha256(lock_path.read_bytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


def env_fingerprint(keys: Iterable[str]) -> dict[str, Any]:
    """Record WHICH credentials/config are present (never their values)."""

    return {key: (key in os.environ and bool(os.environ[key])) for key in keys}
