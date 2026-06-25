"""Thin, optional LangGraph adapter.

Install with the ``langgraph`` extra: ``pip install pandaprobe-harness[langgraph]``.
The adapter injects a ``SystemMessage`` into the next graph state's ``messages``
list — it never writes the checkpoint store directly.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from typing import TYPE_CHECKING, Any

from ..hook.turn import TurnContext

if TYPE_CHECKING:
    from ..hook.core import PandaHarnessHook

__all__ = ["LangGraphAdapter"]


class LangGraphAdapter:
    """Bridge ``PandaHarnessHook`` to a LangGraph execution.

    ``message_sink`` is the mutable ``messages`` sequence of the state that the
    next graph step will consume.
    """

    def __init__(self, message_sink: MutableSequence[Any]) -> None:
        self._sink = message_sink
        self._hook: PandaHarnessHook | None = None

    def parse_turn(self, raw_turn: object) -> TurnContext:
        if not isinstance(raw_turn, Mapping):
            raise TypeError("LangGraph raw_turn must be a state mapping")
        config = raw_turn.get("config", {})
        configurable = config.get("configurable", {}) if isinstance(config, Mapping) else {}
        session_id = configurable.get("thread_id") or raw_turn.get("session_id")
        if not session_id:
            raise ValueError("could not resolve session_id (thread_id) from state")
        return TurnContext(
            session_id=str(session_id),
            turn_index=int(raw_turn.get("turn_index", 0)),
            end_state={"messages": list(raw_turn.get("messages", []))},
        )

    def inject_alert(self, alert: str) -> None:
        try:
            from langchain_core.messages import SystemMessage
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "LangGraphAdapter requires langchain-core; install the "
                "'langgraph' extra."
            ) from exc
        self._sink.append(SystemMessage(content=alert))

    def register(self, hook: PandaHarnessHook) -> None:
        self._hook = hook
