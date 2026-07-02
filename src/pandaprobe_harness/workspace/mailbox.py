"""The diagnostic mailbox — the pull-model replacement for alert injection.

The hook *posts* structured ``DiagnosticNotice``s to ``<harness_root>/mailbox/
pending/``; the agent *pulls* them through its harness toolset, analyzes the
flagged traces, records a mitigation rule, and *acknowledges* each notice
(moving it to ``processed/``). ``status.json`` is a cheap always-current
summary the system-context banner reads without scanning the directory.

All methods are synchronous blocking I/O; async callers wrap them in
``asyncio.to_thread``. A ``threading.Lock`` guards post/acknowledge/status
because one workspace may be shared by multiple sessions on the thread pool.
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, cast

from ..config import HarnessConfig
from ._io import atomic_write_json, load_json

__all__ = [
    "DiagnosticNotice",
    "Mailbox",
    "MailboxStatus",
    "NoticeMetric",
    "Resolution",
    "Severity",
]

Severity = Literal["breach", "relative", "trend", "needs_human"]

_SEVERITY_RANK: dict[str, int] = {
    "trend": 0,
    "relative": 1,
    "breach": 2,
    "needs_human": 3,
}

# Notice ids become filenames, so they must be a single safe path component.
# Anything with a separator, "..", or other funny business is rejected before
# it can escape the mailbox directory (the agent supplies ids to read/ack).
_SAFE_NOTICE_ID = re.compile(r"\A[A-Za-z0-9._-]{1,128}\Z")


def _safe_notice_id(notice_id: str) -> bool:
    return bool(_SAFE_NOTICE_ID.match(notice_id)) and notice_id not in {".", ".."}


def _as_severity(value: object) -> Severity:
    """Forgiving severity parse; unknown values degrade to ``breach``."""

    if isinstance(value, str) and value in _SEVERITY_RANK:
        return cast(Severity, value)
    return "breach"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class NoticeMetric:
    """One metric's contribution to a notice."""

    name: str
    value: float | None
    threshold: float
    reason: str | None = None
    conditions: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "reason": self.reason,
            "conditions": list(self.conditions),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> NoticeMetric:
        value = data.get("value")
        threshold = data.get("threshold")
        conditions = data.get("conditions")
        return cls(
            name=str(data.get("name", "")),
            value=float(value) if isinstance(value, (int, float)) else None,
            threshold=float(threshold) if isinstance(threshold, (int, float)) else 0.0,
            reason=str(data["reason"]) if data.get("reason") is not None else None,
            conditions=tuple(str(c) for c in conditions) if isinstance(conditions, list) else (),
        )


@dataclass(frozen=True, slots=True)
class Resolution:
    """How a notice was resolved when it was acknowledged."""

    acked_at: str
    rule_id: str | None = None
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"acked_at": self.acked_at, "rule_id": self.rule_id, "note": self.note}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Resolution:
        return cls(
            acked_at=str(data.get("acked_at", "")),
            rule_id=str(data["rule_id"]) if data.get("rule_id") is not None else None,
            note=str(data["note"]) if data.get("note") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticNotice:
    """A single self-diagnostic finding awaiting the agent's attention."""

    id: str
    created_at: str
    session_id: str
    turn_index: int
    severity: Severity
    metrics: tuple[NoticeMetric, ...] = ()
    flagged_traces: tuple[str, ...] = ()
    signal_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    dump_path: str = ""
    summary: str = ""
    signatures: tuple[str, ...] = ()
    status: Literal["pending", "acknowledged"] = "pending"
    resolution: Resolution | None = None

    @staticmethod
    def new_id(now: datetime | None = None) -> str:
        """Sortable, collision-proof notice id (lexicographic ≈ chronological)."""

        stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%f")
        return f"n-{stamp}-{uuid.uuid4().hex[:8]}"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "severity": self.severity,
            "metrics": [m.to_json() for m in self.metrics],
            "flagged_traces": list(self.flagged_traces),
            "signal_breakdown": self.signal_breakdown,
            "dump_path": self.dump_path,
            "summary": self.summary,
            "signatures": list(self.signatures),
            "status": self.status,
            "resolution": self.resolution.to_json() if self.resolution else None,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DiagnosticNotice:
        metrics_raw = data.get("metrics")
        metrics = (
            tuple(NoticeMetric.from_json(m) for m in metrics_raw if isinstance(m, dict))
            if isinstance(metrics_raw, list)
            else ()
        )
        flagged = data.get("flagged_traces")
        signatures = data.get("signatures")
        breakdown_raw = data.get("signal_breakdown")
        breakdown: dict[str, dict[str, Any]] = {}
        if isinstance(breakdown_raw, dict):
            for trace_id, signals in breakdown_raw.items():
                if isinstance(signals, dict):
                    breakdown[str(trace_id)] = dict(signals)
        resolution_raw = data.get("resolution")
        status = data.get("status")
        return cls(
            id=str(data.get("id", "")),
            created_at=str(data.get("created_at", "")),
            session_id=str(data.get("session_id", "")),
            turn_index=int(data.get("turn_index", 0)),
            severity=_as_severity(data.get("severity")),
            metrics=metrics,
            flagged_traces=tuple(str(t) for t in flagged) if isinstance(flagged, list) else (),
            signal_breakdown=breakdown,
            dump_path=str(data.get("dump_path", "")),
            summary=str(data.get("summary", "")),
            signatures=tuple(str(s) for s in signatures) if isinstance(signatures, list) else (),
            status="acknowledged" if status == "acknowledged" else "pending",
            resolution=(
                Resolution.from_json(resolution_raw) if isinstance(resolution_raw, dict) else None
            ),
        )


@dataclass(frozen=True, slots=True)
class MailboxStatus:
    """Compact mailbox summary for the system-context banner."""

    pending_count: int
    max_severity: Severity | None
    latest_id: str | None
    updated_at: str

    def to_json(self) -> dict[str, Any]:
        return {
            "pending_count": self.pending_count,
            "max_severity": self.max_severity,
            "latest_id": self.latest_id,
            "updated_at": self.updated_at,
        }


class Mailbox:
    """Filesystem-backed notice store under ``<harness_root>/mailbox/``."""

    def __init__(self, config: HarnessConfig) -> None:
        self._config = config
        self._lock = threading.Lock()

    # -- provisioning ---------------------------------------------------------

    def provision(self) -> None:
        """Create the mailbox tree (idempotent)."""

        self._config.mailbox_pending_dir.mkdir(parents=True, exist_ok=True)
        self._config.mailbox_processed_dir.mkdir(parents=True, exist_ok=True)

    # -- producing side --------------------------------------------------------

    def post(self, notice: DiagnosticNotice) -> None:
        """Persist a pending notice and refresh ``status.json``."""

        with self._lock:
            self.provision()
            atomic_write_json(
                self._config.mailbox_pending_dir / f"{notice.id}.json", notice.to_json()
            )
            self._refresh_status()

    # -- consuming side (the agent) --------------------------------------------

    def pending(self) -> list[DiagnosticNotice]:
        """All pending notices, oldest first."""

        notices: list[DiagnosticNotice] = []
        try:
            paths = sorted(self._config.mailbox_pending_dir.glob("*.json"))
        except OSError:
            return []
        for path in paths:
            data = load_json(path)
            if data is not None:
                notices.append(DiagnosticNotice.from_json(data))
        return notices

    def read(self, notice_id: str) -> DiagnosticNotice | None:
        """Look a notice up by id, in ``pending/`` then ``processed/``."""

        if not _safe_notice_id(notice_id):
            return None
        for directory in (self._config.mailbox_pending_dir, self._config.mailbox_processed_dir):
            data = load_json(directory / f"{notice_id}.json")
            if data is not None:
                return DiagnosticNotice.from_json(data)
        return None

    def acknowledge(
        self, notice_id: str, *, rule_id: str | None = None, note: str | None = None
    ) -> DiagnosticNotice:
        """Move a pending notice to ``processed/`` with its resolution.

        Raises ``KeyError`` when the notice is not currently pending.
        """

        if not _safe_notice_id(notice_id):
            raise KeyError(f"invalid notice id {notice_id!r}")
        with self._lock:
            pending_path = self._config.mailbox_pending_dir / f"{notice_id}.json"
            data = load_json(pending_path)
            if data is None:
                raise KeyError(f"notice {notice_id!r} is not pending")
            notice = DiagnosticNotice.from_json(data)
            acknowledged = replace(
                notice,
                status="acknowledged",
                resolution=Resolution(acked_at=_now_iso(), rule_id=rule_id, note=note),
            )
            atomic_write_json(
                self._config.mailbox_processed_dir / f"{notice_id}.json",
                acknowledged.to_json(),
            )
            pending_path.unlink(missing_ok=True)
            self._refresh_status()
            return acknowledged

    # -- status ------------------------------------------------------------------

    def status(self) -> MailboxStatus:
        """The current summary; recomputed from ``pending/`` when stale/absent."""

        data = load_json(self._config.mailbox_status_file)
        if data is not None:
            count = data.get("pending_count")
            if isinstance(count, int):
                raw_severity = data.get("max_severity")
                return MailboxStatus(
                    pending_count=count,
                    max_severity=_as_severity(raw_severity) if raw_severity else None,
                    latest_id=(
                        str(data["latest_id"]) if data.get("latest_id") is not None else None
                    ),
                    updated_at=str(data.get("updated_at", "")),
                )
        with self._lock:
            return self._refresh_status()

    def _refresh_status(self) -> MailboxStatus:
        """Recompute status from ``pending/`` and persist it. Caller holds the lock."""

        notices = self.pending()
        max_severity: Severity | None = None
        for notice in notices:
            if max_severity is None or (
                _SEVERITY_RANK[notice.severity] > _SEVERITY_RANK[max_severity]
            ):
                max_severity = notice.severity
        status = MailboxStatus(
            pending_count=len(notices),
            max_severity=max_severity,
            latest_id=notices[-1].id if notices else None,
            updated_at=_now_iso(),
        )
        atomic_write_json(self._config.mailbox_status_file, status.to_json())
        return status
