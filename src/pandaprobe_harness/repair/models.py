"""Structured assignments, usage, and outcomes for managed repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..workspace.mailbox import DiagnosticNotice

__all__ = [
    "RepairAssignment",
    "RepairResult",
    "RepairStatus",
    "RepairUsage",
]

RepairStatus = Literal[
    "candidate_added",
    "duplicate",
    "already_covered",
    "no_proposal",
    "unactionable",
    "timed_out",
    "failed",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class RepairUsage:
    """Provider-normalized usage accumulated across repair model turns."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None

    def plus(self, other: RepairUsage) -> RepairUsage:
        costs = [value for value in (self.cost, other.cost) if value is not None]
        return RepairUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=sum(costs) if costs else None,
        )

    def to_json(self) -> dict[str, int | float | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
        }


@dataclass(frozen=True, slots=True)
class RepairAssignment:
    """Bounded evidence assigned to one package-owned repair run."""

    task_session_id: str
    repair_session_id: str
    turn_index: int
    notice: DiagnosticNotice
    episode_id: str = ""
    notices: tuple[DiagnosticNotice, ...] = ()
    scope_hints: tuple[dict[str, Any], ...] = ()
    #: A precise host-recommended scope, or ``None`` for "no recommendation".
    #: Never defaulted: the scope decision belongs to the repair model, and a
    #: pre-filled value would read as an instruction it should follow.
    recommended_scope: str | None = None
    generic_scopes: tuple[str, ...] = ()
    task_descriptor: dict[str, Any] = field(default_factory=dict)
    #: What the failing turn was trying to do, in the host's own words.
    task_summary: str = ""
    domain_policy: str | None = None

    @property
    def notice_id(self) -> str:
        return self.notice.id

    @property
    def all_notices(self) -> tuple[DiagnosticNotice, ...]:
        return self.notices or (self.notice,)

    @property
    def notice_ids(self) -> tuple[str, ...]:
        return tuple(notice.id for notice in self.all_notices)

    def summary(self) -> dict[str, Any]:
        return {
            "task_session_id": self.task_session_id,
            "repair_session_id": self.repair_session_id,
            "repair_episode_id": self.episode_id,
            "turn_index": self.turn_index,
            "notice_id": self.notice.id,
            "notice_ids": list(self.notice_ids),
            "severity": self.notice.severity,
            "summary": self.notice.summary,
            "signatures": list(self.notice.signatures),
            "metrics": [metric.to_json() for metric in self.notice.metrics],
            "flagged_traces": list(self.notice.flagged_traces),
            "diagnostic_dump": self.notice.dump_path,
            "notices": [
                {
                    "notice_id": notice.id,
                    "severity": notice.severity,
                    "summary": notice.summary,
                    "signatures": list(notice.signatures),
                    "metrics": [metric.to_json() for metric in notice.metrics],
                    "flagged_traces": list(notice.flagged_traces),
                    "diagnostic_dump": notice.dump_path,
                }
                for notice in self.all_notices
            ],
            "scope_hints": [dict(hint) for hint in self.scope_hints],
            "recommended_scope": self.recommended_scope,
            "task_descriptor": self.task_descriptor,
            "task_summary": self.task_summary,
            "domain_policy": self.domain_policy,
        }


@dataclass(frozen=True, slots=True)
class RepairResult:
    """A settlement-safe managed repair outcome."""

    task_session_id: str
    repair_session_id: str
    notice_id: str
    status: RepairStatus
    started_at: str
    ended_at: str
    model: str
    provider: str
    episode_id: str = ""
    notice_ids: tuple[str, ...] = ()
    turns: int = 0
    tool_calls: int = 0
    candidate_rule_ids: tuple[str, ...] = ()
    existing_rule_id: str | None = None
    recommended_scope: str | None = None
    selected_scope: str | None = None
    scope_rationale: str | None = None
    considered_rule_ids: tuple[str, ...] = ()
    resolution_kind: str | None = None
    candidate_suppression_reason: str | None = None
    usage: RepairUsage = RepairUsage()
    error_category: str | None = None
    message: str | None = None
    tracing_enabled: bool = False

    @property
    def succeeded(self) -> bool:
        return self.status in {
            "candidate_added", "duplicate", "already_covered", "no_proposal",
            "unactionable",
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "task_session_id": self.task_session_id,
            "repair_session_id": self.repair_session_id,
            "notice_id": self.notice_id,
            "repair_episode_id": self.episode_id,
            "notice_ids": list(self.notice_ids or (self.notice_id,)),
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "model": self.model,
            "provider": self.provider,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "candidate_rule_ids": list(self.candidate_rule_ids),
            "existing_rule_id": self.existing_rule_id,
            "recommended_scope": self.recommended_scope,
            "selected_scope": self.selected_scope,
            "scope_rationale": self.scope_rationale,
            "considered_rule_ids": list(self.considered_rule_ids),
            "resolution_kind": self.resolution_kind or self.status,
            "candidate_suppression_reason": self.candidate_suppression_reason,
            "usage": self.usage.to_json(),
            "error_category": self.error_category,
            "message": self.message,
            "tracing_enabled": self.tracing_enabled,
        }
