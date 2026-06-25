"""Thin, optional AutoGen adapter.

Install with the ``autogen`` extra. The adapter appends the alert as a system
message to the next inbound message list — it never mutates conversation state
stored elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from typing import TYPE_CHECKING, Any

from ..hook.turn import TurnContext

if TYPE_CHECKING:
    from ..hook.core import PandaHarnessHook

__all__ = ["AutoGenAdapter"]


class AutoGenAdapter:
    """Bridge ``PandaHarnessHook`` to an AutoGen agent.

    ``message_sink`` is the mutable inbound message list for the next reply.
    """

    def __init__(self, message_sink: MutableSequence[Any]) -> None:
        self._sink = message_sink
        self._hook: PandaHarnessHook | None = None

    def parse_turn(self, raw_turn: object) -> TurnContext:
        if not isinstance(raw_turn, Mapping):
            raise TypeError("AutoGen raw_turn must be a mapping (chat result)")
        session_id = raw_turn.get("session_id") or raw_turn.get("chat_id")
        if not session_id:
            raise ValueError("could not resolve session_id (chat_id) from chat result")
        return TurnContext(
            session_id=str(session_id),
            turn_index=int(raw_turn.get("turn_index", 0)),
            end_state={"messages": list(raw_turn.get("messages", []))},
        )

    def inject_alert(self, alert: str) -> None:
        self._sink.append({"role": "system", "content": alert})

    def register(self, hook: PandaHarnessHook) -> None:
        self._hook = hook
