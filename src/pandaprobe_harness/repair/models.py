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
    "completed",
    "duplicate",
    "no_proposal",
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
    task_descriptor: dict[str, Any] = field(default_factory=dict)
    domain_policy: str | None = None

    @property
    def notice_id(self) -> str:
        return self.notice.id

    def summary(self) -> dict[str, Any]:
        return {
            "task_session_id": self.task_session_id,
            "repair_session_id": self.repair_session_id,
            "turn_index": self.turn_index,
            "notice_id": self.notice.id,
            "severity": self.notice.severity,
            "summary": self.notice.summary,
            "signatures": list(self.notice.signatures),
            "metrics": [metric.to_json() for metric in self.notice.metrics],
            "flagged_traces": list(self.notice.flagged_traces),
            "diagnostic_dump": self.notice.dump_path,
            "task_descriptor": self.task_descriptor,
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
    turns: int = 0
    tool_calls: int = 0
    candidate_rule_ids: tuple[str, ...] = ()
    existing_rule_id: str | None = None
    usage: RepairUsage = RepairUsage()
    error_category: str | None = None
    message: str | None = None
    tracing_enabled: bool = False

    @property
    def succeeded(self) -> bool:
        return self.status in {"completed", "duplicate", "no_proposal"}

    def to_json(self) -> dict[str, Any]:
        return {
            "task_session_id": self.task_session_id,
            "repair_session_id": self.repair_session_id,
            "notice_id": self.notice_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "model": self.model,
            "provider": self.provider,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "candidate_rule_ids": list(self.candidate_rule_ids),
            "existing_rule_id": self.existing_rule_id,
            "usage": self.usage.to_json(),
            "error_category": self.error_category,
            "message": self.message,
            "tracing_enabled": self.tracing_enabled,
        }
