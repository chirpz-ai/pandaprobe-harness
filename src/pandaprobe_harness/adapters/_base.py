"""Shared base for framework adapters: turn detection + session bridging.

Holds the common ``FrameworkAdapter`` contract (``parse_turn`` with a session
bridge, ``register``) plus a ``notify_turn_end`` convenience that
framework-specific instrumentation (callbacks, monkeypatch wrappers) calls to
fire ``hook.on_turn_end`` for one completed turn. That is the adapter's whole
job in the pull model — diagnostic delivery happens through the workspace
mailbox and the agent's harness toolset, never through the adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..hook.turn import TurnContext
from ._session import current_session_id

if TYPE_CHECKING:
    from ..hook.core import PandaHarnessHook

__all__ = ["BaseSinkAdapter"]


class BaseSinkAdapter:
    """Common adapter contract: session resolution + turn-end firing."""

    #: Extra mapping keys (besides ``session_id``) a turn payload may carry the
    #: session under (e.g. ``crew_id``, ``chat_id``). Tried in order.
    _id_keys: tuple[str, ...] = ()

    def __init__(self, *, session_id: str | None = None) -> None:
        self._session_id = session_id
        self._hook: PandaHarnessHook | None = None
        self._turn_index = 0

    # -- FrameworkAdapter contract -------------------------------------------

    def register(self, hook: PandaHarnessHook) -> None:
        self._hook = hook

    def parse_turn(self, raw_turn: object) -> TurnContext:
        session_id: str | None = None
        turn_index = 0
        end_state: Mapping[str, Any] = {}
        if isinstance(raw_turn, Mapping):
            session_id = raw_turn.get("session_id")
            for key in self._id_keys:
                session_id = session_id or raw_turn.get(key)
            turn_index = int(raw_turn.get("turn_index", 0))
            raw_end = raw_turn.get("end_state", {})
            end_state = raw_end if isinstance(raw_end, Mapping) else {}
        session_id = session_id or self._session_id or current_session_id()
        if not session_id:
            raise ValueError(
                f"{type(self).__name__} could not resolve a session_id; pass "
                "session_id=... or run inside `pandaprobe.session(...)`."
            )
        return TurnContext(
            session_id=str(session_id), turn_index=turn_index, end_state=dict(end_state)
        )

    # -- helpers -----------------------------------------------------------------

    def startup_context_text(self) -> str:
        """The harness system context (rules + protocol + mailbox banner)."""

        if self._hook is None:
            return ""
        return self._hook.startup_context()

    def notify_turn_end(
        self,
        *,
        session_id: str | None = None,
        turn_index: int | None = None,
        end_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Fire ``hook.on_turn_end`` for one completed turn (used by instrumentation)."""

        if self._hook is None:
            return
        self._turn_index += 1
        self._hook.on_turn_end(
            {
                "session_id": session_id or current_session_id() or self._session_id,
                "turn_index": turn_index if turn_index is not None else self._turn_index,
                "end_state": dict(end_state) if end_state else {},
            }
        )
