"""Framework-agnostic representation of a completed agent turn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TurnContext", "parse_turn_payload"]


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
    return TurnContext(
        session_id=str(session_id),
        turn_index=int(raw_turn.get("turn_index", 0)),
        end_state=end_state,
    )
