"""Reference adapter for a raw, framework-less agent loop.

This is the canonical, dependency-free implementation of ``FrameworkAdapter``
and documents the intended contract. It owns an explicit inbound message queue;
``inject_alert`` appends to it and the driving loop consumes it at the start of
the next turn via :meth:`consume_alerts`.

A turn payload is a plain mapping::

    {"session_id": "s-1", "turn_index": 3, "end_state": {...}}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..hook.turn import TurnContext

if TYPE_CHECKING:
    from ..hook.core import PandaHarnessHook

__all__ = ["RawLoopAdapter"]


class RawLoopAdapter:
    """A minimal adapter backed by an in-memory message queue."""

    def __init__(self) -> None:
        self._inbox: list[str] = []
        self._hook: PandaHarnessHook | None = None

    def parse_turn(self, raw_turn: object) -> TurnContext:
        if not isinstance(raw_turn, Mapping):
            raise TypeError(f"raw_turn must be a mapping, got {type(raw_turn).__name__}")
        session_id = raw_turn.get("session_id")
        if not session_id:
            raise ValueError("raw_turn is missing 'session_id'")
        end_state = raw_turn.get("end_state", {})
        return TurnContext(
            session_id=str(session_id),
            turn_index=int(raw_turn.get("turn_index", 0)),
            end_state=dict(end_state) if isinstance(end_state, Mapping) else {},
        )

    def inject_alert(self, alert: str) -> None:
        self._inbox.append(alert)

    def register(self, hook: PandaHarnessHook) -> None:
        self._hook = hook

    # -- raw-loop specific helpers -------------------------------------------

    @property
    def pending_alerts(self) -> tuple[str, ...]:
        """Alerts currently queued for the next turn (non-destructive view)."""

        return tuple(self._inbox)

    def consume_alerts(self) -> list[str]:
        """Pop and return all queued alerts (called at the start of a turn)."""

        alerts, self._inbox = self._inbox, []
        return alerts

    @staticmethod
    def make_turn(
        session_id: str, turn_index: int, **end_state: Any
    ) -> dict[str, Any]:
        """Convenience constructor for a raw turn payload."""

        return {
            "session_id": session_id,
            "turn_index": turn_index,
            "end_state": dict(end_state),
        }
