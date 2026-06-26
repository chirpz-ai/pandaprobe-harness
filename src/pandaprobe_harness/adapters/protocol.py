"""The contract every framework adapter implements.

An adapter has exactly two jobs:

1. ``parse_turn`` — translate a framework's turn-end payload into a normalized
   ``TurnContext``.
2. ``inject_alert`` — append an alert string to the agent's *next-turn* inbound
   message queue (never mutating any checkpoint/state store).

``register`` wires the hook into the framework's event system; sync frameworks
should schedule ``hook.on_turn_end`` and call ``hook.drain_pending`` at the
start of each turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..hook.turn import TurnContext

if TYPE_CHECKING:
    from ..hook.core import PandaHarnessHook

__all__ = ["FrameworkAdapter"]


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Glue between ``PandaHarnessHook`` and a concrete agent framework."""

    def parse_turn(self, raw_turn: object) -> TurnContext:
        """Normalize a framework turn-end payload into a ``TurnContext``."""
        ...

    def inject_alert(self, alert: str) -> None:
        """Queue ``alert`` for the agent's next turn."""
        ...

    def register(self, hook: PandaHarnessHook) -> None:
        """Wire the hook into the framework's lifecycle callbacks."""
        ...
