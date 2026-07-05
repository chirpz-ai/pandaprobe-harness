"""The replayable regression eval-set — scenarios worth re-running.

When a breach notice is posted (and ``capture_eval_cases`` is on) the hook
captures the session as a ``failure`` case: its signature, its baseline
scores, and — when the turn payload carried one — the input needed to replay
it. Known-good sessions can be captured as ``win`` cases. Replaying a case
against the *current* rule set is how the harness gets counterfactual
evidence: it is the strong path for candidate-rule validation and the guard
against a new rule quietly breaking an old win (``run_regression``).

The platform is passive and trace-based, so the harness cannot re-run the
developer's agent itself: replay happens through a developer-supplied
:data:`ReplayFn`. A case without a ``replay_input`` still persists (it is a
usable calibration label) — it is simply skipped by replay paths.

Storage is one JSON file per case under ``<harness_root>/evalset/`` (the
mailbox pattern): eviction is an unlink, attaching an input is one atomic
rewrite, and case ids are validated as safe path components because the
agent supplies them to ``attach_input``.

All methods are synchronous blocking I/O; async callers wrap them in
``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..config import HarnessConfig
from ._io import atomic_write_json, load_json
from .journal import Journal
from .sanitize import sanitize_text

__all__ = ["CaseKind", "EvalCase", "EvalSet", "ReplayFn"]

logger = logging.getLogger("pandaprobe_harness.workspace")

CaseKind = Literal["failure", "win"]

# Case ids become filenames (and the agent supplies them to attach_input), so
# they must be a single safe path component — same contract as notice ids.
_SAFE_CASE_ID = re.compile(r"\A[A-Za-z0-9._-]{1,128}\Z")


def _safe_case_id(case_id: str) -> bool:
    return bool(_SAFE_CASE_ID.match(case_id)) and case_id not in {".", ".."}


def _as_kind(value: object) -> CaseKind:
    """Forgiving kind parse; unknown values degrade to ``failure``."""

    return "win" if value == "win" else "failure"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One captured scenario: a failure to fix or a win to protect."""

    id: str
    created_at: str
    session_id: str
    kind: CaseKind = "failure"
    signature: tuple[str, ...] = ()
    baseline_scores: dict[str, float] = field(default_factory=dict)
    replay_input: Any | None = None
    notes: str = ""

    @staticmethod
    def new_id() -> str:
        return f"c-{uuid.uuid4().hex[:10]}"

    @property
    def replayable(self) -> bool:
        return self.replay_input is not None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "kind": self.kind,
            "signature": list(self.signature),
            "baseline_scores": dict(self.baseline_scores),
            "replay_input": self.replay_input,
            "notes": self.notes,
        }

    def summary(self) -> dict[str, Any]:
        """Agent/tool-facing view: everything except the replay payload bulk."""

        return {
            "id": self.id,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "kind": self.kind,
            "signature": list(self.signature),
            "baseline_scores": dict(self.baseline_scores),
            "replayable": self.replayable,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EvalCase:
        signature = data.get("signature")
        scores_raw = data.get("baseline_scores")
        scores: dict[str, float] = {}
        if isinstance(scores_raw, dict):
            for metric, value in scores_raw.items():
                if isinstance(value, (int, float)):
                    scores[str(metric)] = float(value)
        return cls(
            id=str(data.get("id", "")),
            created_at=str(data.get("created_at", "")),
            session_id=str(data.get("session_id", "")),
            kind=_as_kind(data.get("kind")),
            signature=(
                tuple(str(s) for s in signature) if isinstance(signature, list) else ()
            ),
            baseline_scores=scores,
            replay_input=data.get("replay_input"),
            notes=str(data.get("notes", "")),
        )


#: The replay seam: given a case and the system-context string to run under,
#: re-run the agent on the case's input and return the NEW session id the run
#: produced (the harness then scores that session via the evaluator). Lives
#: here so ``validation/`` can import it without a cycle.
ReplayFn = Callable[[EvalCase, str], Awaitable[str]]


class EvalSet:
    """Filesystem-backed eval-case store under ``<harness_root>/evalset/``."""

    def __init__(self, config: HarnessConfig, *, journal: Journal | None = None) -> None:
        self._config = config
        self._journal = journal
        self._lock = threading.Lock()

    # -- provisioning ---------------------------------------------------------

    def provision(self) -> None:
        """Create the eval-set directory (idempotent)."""

        self._config.evalset_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str) -> Path:
        return self._config.evalset_dir / f"{case_id}.json"

    # -- writes -----------------------------------------------------------------

    def capture(
        self,
        *,
        session_id: str,
        kind: CaseKind = "failure",
        signature: Sequence[str] = (),
        baseline_scores: Mapping[str, float] | None = None,
        replay_input: Any | None = None,
        notes: str = "",
    ) -> EvalCase | None:
        """Persist a new case; idempotent per (session, signature, kind); capped.

        Returns the stored (or pre-existing identical) case, or ``None`` when
        the corpus is full of ``win`` cases and nothing may be evicted.
        """

        clean_notes = sanitize_text(notes, max_len=self._config.sanitize_max_len)
        sig = tuple(str(s) for s in signature)
        dedup_key = (session_id, tuple(sorted(sig)), kind)

        with self._lock:
            self.provision()
            cases = self._load_all()
            for case in cases:
                if (case.session_id, tuple(sorted(case.signature)), case.kind) == dedup_key:
                    return case
            cap = self._config.eval_case_max
            if cap > 0 and len(cases) >= cap and not self._evict_one_locked(cases):
                logger.warning(
                    "eval-set at capacity (%s) with only win cases; not capturing "
                    "session=%s — remove a case or raise eval_case_max",
                    cap,
                    session_id,
                )
                if self._journal is not None:
                    self._journal.record(
                        {"type": "evalset_capture", "skipped": "cap", "session_id": session_id}
                    )
                return None
            entry = EvalCase(
                id=EvalCase.new_id(),
                created_at=_now_iso(),
                session_id=session_id,
                kind=kind,
                signature=sig,
                baseline_scores=dict(baseline_scores or {}),
                replay_input=replay_input,
                notes=clean_notes,
            )
            atomic_write_json(self._path(entry.id), entry.to_json())
        if self._journal is not None:
            self._journal.record(
                {
                    "type": "evalset_capture",
                    "case_id": entry.id,
                    "session_id": entry.session_id,
                    "kind": entry.kind,
                    "signatures": list(entry.signature),
                    "replayable": entry.replayable,
                }
            )
        return entry

    def attach_input(self, case_id: str, payload: Any) -> EvalCase:
        """Attach the replay payload to an existing case (atomic rewrite).

        Raises ``KeyError`` when the case does not exist.
        """

        if not _safe_case_id(case_id):
            raise KeyError(f"invalid eval case id {case_id!r}")
        with self._lock:
            data = load_json(self._path(case_id))
            if data is None:
                raise KeyError(f"no eval case {case_id!r}")
            case = EvalCase.from_json(data)
            updated = replace(case, replay_input=payload)
            atomic_write_json(self._path(case_id), updated.to_json())
        return updated

    def remove(self, case_id: str) -> bool:
        """Delete a case; returns whether it existed."""

        if not _safe_case_id(case_id):
            return False
        with self._lock:
            path = self._path(case_id)
            existed = path.exists()
            path.unlink(missing_ok=True)
        return bool(existed)

    def _evict_one_locked(self, cases: list[EvalCase]) -> bool:
        """Drop the oldest ``failure`` case; ``win`` cases are never evicted."""

        for case in cases:  # cases are oldest-first
            if case.kind == "failure":
                self._path(case.id).unlink(missing_ok=True)
                return True
        return False

    # -- reads ------------------------------------------------------------------

    def get(self, case_id: str) -> EvalCase | None:
        if not _safe_case_id(case_id):
            return None
        data = load_json(self._path(case_id))
        return EvalCase.from_json(data) if data is not None else None

    def cases(self, *, kind: CaseKind | None = None) -> list[EvalCase]:
        """All cases, oldest first, optionally filtered by kind."""

        cases = self._load_all()
        if kind is not None:
            cases = [case for case in cases if case.kind == kind]
        return cases

    def matching(
        self, signature: Sequence[str], *, kind: CaseKind | None = "failure"
    ) -> list[EvalCase]:
        """Cases whose signature overlaps ``signature``, newest first."""

        wanted = {str(s) for s in signature}
        matches = [
            case
            for case in self.cases(kind=kind)
            if wanted and wanted.intersection(case.signature)
        ]
        matches.reverse()  # cases() is oldest-first
        return matches

    def _load_all(self) -> list[EvalCase]:
        """Every parseable case, oldest first. Corrupt files are skipped."""

        try:
            paths = sorted(self._config.evalset_dir.glob("*.json"))
        except OSError:
            return []
        cases: list[EvalCase] = []
        for path in paths:
            data = load_json(path)
            if data is not None:
                case = EvalCase.from_json(data)
                if case.id:
                    cases.append(case)
        return sorted(cases, key=lambda c: (c.created_at, c.id))
