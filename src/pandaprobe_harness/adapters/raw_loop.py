"""Reference adapter for a raw, framework-less agent loop.

This is the canonical, dependency-free implementation of ``FrameworkAdapter``
and documents the intended contract: parse a plain turn payload, register the
hook, and let the driving loop call ``hook.on_turn_end`` at each turn end.
The agent receives its diagnostics by pulling the workspace mailbox through
the harness toolset — there is no inbound message queue.

A turn payload is a plain mapping::

    {"session_id": "s-1", "turn_index": 3, "end_state": {...}}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..hook.turn import TurnContext, parse_turn_payload

if TYPE_CHECKING:
    from ..hook.core import PandaHarnessHook

__all__ = ["RawLoopAdapter"]


class RawLoopAdapter:
    """A minimal, dependency-free turn-detector."""

    def __init__(self) -> None:
        self._hook: PandaHarnessHook | None = None

    def parse_turn(self, raw_turn: object) -> TurnContext:
        return parse_turn_payload(raw_turn)

    def register(self, hook: PandaHarnessHook) -> None:
        self._hook = hook

    # -- raw-loop specific helpers -------------------------------------------

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
