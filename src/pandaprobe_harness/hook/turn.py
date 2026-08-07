"""Framework-agnostic representation of a completed agent turn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..workspace.rules import normalize_scope, normalize_scope_description

__all__ = ["RuleScopeHint", "TurnContext", "parse_turn_payload"]

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


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Normalized turn end-state, produced by an adapter or the facade.

    ``session_id`` groups the conversation; ``turn_index`` orders turns within
    it; ``end_state`` carries any framework-specific payload (messages, tool
    calls) that the evaluator may inspect.
    """

    session_id: str
    turn_index: int
    end_state: Mapping[str, Any] = field(default_factory=dict)
    rule_scope_hints: tuple[RuleScopeHint, ...] = ()


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
    return TurnContext(
        session_id=str(session_id),
        turn_index=int(raw_turn.get("turn_index", 0)),
        end_state=end_state,
        rule_scope_hints=tuple(hints),
    )
