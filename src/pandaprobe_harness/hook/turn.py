"""Framework-agnostic representation of a completed agent turn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..workspace.sanitize import sanitize_text
from ..workspace.scopes import normalize_scope, normalize_scope_description

__all__ = [
    "TASK_SUMMARY_MAX_LEN",
    "RuleScopeHint",
    "TurnContext",
    "clean_task_summary",
    "parse_turn_payload",
]

RuleApplicabilityHint = Literal["global", "topical", "task"]


@dataclass(frozen=True, slots=True)
class RuleScopeHint:
    """Bounded host-owned topic metadata carried with one task turn."""

    key: str
    description: str = ""
    applicability: RuleApplicabilityHint = "topical"
    recommended: bool = True

    def __post_init__(self) -> None:
        key = normalize_scope(self.key)
        applicability: RuleApplicabilityHint = (
            self.applicability
            if self.applicability in {"global", "topical", "task"}
            else "topical"
        )
        object.__setattr__(self, "key", key)
        object.__setattr__(
            self,
            "description",
            normalize_scope_description(self.description, scope=key),
        )
        object.__setattr__(self, "applicability", applicability)

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "applicability": self.applicability,
            "recommended": self.recommended,
        }

    @classmethod
    def from_json(cls, value: object) -> RuleScopeHint | None:
        if isinstance(value, RuleScopeHint):
            return value
        if not isinstance(value, Mapping) or not value.get("key"):
            return None
        applicability = value.get("applicability")
        return cls(
            key=str(value["key"]),
            description=str(value.get("description") or ""),
            applicability=(
                applicability
                if applicability in {"global", "topical", "task"}
                else "topical"
            ),
            recommended=bool(value.get("recommended", True)),
        )


#: Upper bound on a host-supplied task summary. Long enough for a real task
#: statement, short enough that it cannot crowd out the repair evidence.
TASK_SUMMARY_MAX_LEN = 600


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Normalized turn end-state, produced by an adapter or the facade.

    ``session_id`` groups the conversation; ``turn_index`` orders turns within
    it; ``end_state`` carries any framework-specific payload (messages, tool
    calls) that the evaluator may inspect.

    ``task_summary`` is an optional short statement of what the turn was trying
    to do. Any host may supply it; managed repair reads it as untrusted evidence
    when diagnosing a failure and choosing a rule scope. It is the generic
    alternative to the harness guessing a topic from an opaque task id.
    """

    session_id: str
    turn_index: int
    end_state: Mapping[str, Any] = field(default_factory=dict)
    rule_scope_hints: tuple[RuleScopeHint, ...] = ()
    task_summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_summary", clean_task_summary(self.task_summary))


def clean_task_summary(value: object) -> str:
    """Bound and sanitize a host-supplied task summary.

    Passes through the same prompt-injection boundary as every other
    externally-authored string, so a task statement cannot forge harness framing
    in the repair prompt.
    """

    if not isinstance(value, str) or not value.strip():
        return ""
    collapsed = " ".join(value.split())
    return sanitize_text(collapsed, max_len=TASK_SUMMARY_MAX_LEN).strip()


def parse_turn_payload(raw_turn: object) -> TurnContext:
    """Normalize a plain turn payload (mapping or ``TurnContext``) — the hook's
    default parser when no adapter-specific parser is wired."""

    if isinstance(raw_turn, TurnContext):
        return raw_turn
    if not isinstance(raw_turn, Mapping):
        raise TypeError(
            f"expected a mapping turn payload or TurnContext, got {type(raw_turn).__name__}"
        )
    session_id = raw_turn.get("session_id")
    if not session_id:
        raise ValueError("turn payload is missing a session_id")
    raw_end = raw_turn.get("end_state", {})
    end_state = dict(raw_end) if isinstance(raw_end, Mapping) else {}
    raw_hints = raw_turn.get("rule_scope_hints", ())
    hints: list[RuleScopeHint] = []
    if isinstance(raw_hints, (list, tuple)):
        for value in raw_hints[:16]:
            hint = RuleScopeHint.from_json(value)
            if hint is not None and hint.key not in {item.key for item in hints}:
                hints.append(hint)
    # Accepted at the top level or inside end_state: adapters that already build
    # an end_state payload can carry it there without a signature change.
    summary = raw_turn.get("task_summary")
    if not isinstance(summary, str) or not summary.strip():
        summary = end_state.get("task_summary")
    return TurnContext(
        session_id=str(session_id),
        turn_index=int(raw_turn.get("turn_index", 0)),
        end_state=end_state,
        rule_scope_hints=tuple(hints),
        task_summary=clean_task_summary(summary),
    )
