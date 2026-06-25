"""Thin, optional CrewAI adapter.

Install with the ``crewai`` extra. The adapter appends the alert to the next
task's context list — it never mutates persisted crew memory.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from typing import TYPE_CHECKING, Any

from ..hook.turn import TurnContext

if TYPE_CHECKING:
    from ..hook.core import PandaHarnessHook

__all__ = ["CrewAIAdapter"]


class CrewAIAdapter:
    """Bridge ``PandaHarnessHook`` to a CrewAI run.

    ``context_sink`` is the mutable context sequence handed to the next task.
    """

    def __init__(self, context_sink: MutableSequence[Any]) -> None:
        self._sink = context_sink
        self._hook: PandaHarnessHook | None = None

    def parse_turn(self, raw_turn: object) -> TurnContext:
        if not isinstance(raw_turn, Mapping):
            raise TypeError("CrewAI raw_turn must be a mapping (task output)")
        session_id = raw_turn.get("session_id") or raw_turn.get("crew_id")
        if not session_id:
            raise ValueError("could not resolve session_id (crew_id) from task output")
        return TurnContext(
            session_id=str(session_id),
            turn_index=int(raw_turn.get("turn_index", 0)),
            end_state={"output": raw_turn.get("output")},
        )

    def inject_alert(self, alert: str) -> None:
        self._sink.append(alert)

    def register(self, hook: PandaHarnessHook) -> None:
        self._hook = hook
