"""Framework-agnostic representation of a completed agent turn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TurnContext"]


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Normalized turn end-state, produced by a ``FrameworkAdapter``.

    ``session_id`` groups the conversation; ``turn_index`` orders turns within
    it; ``end_state`` carries any framework-specific payload (messages, tool
    calls) that the evaluator or alert builder may inspect.
    """

    session_id: str
    turn_index: int
    end_state: Mapping[str, Any] = field(default_factory=dict)
