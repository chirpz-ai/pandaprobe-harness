"""The contract every framework adapter implements.

An adapter is a pure turn detector with exactly two
jobs:

1. ``parse_turn`` — translate a framework's turn-end payload into a normalized
   ``TurnContext`` (resolving the session id).
2. ``register`` — wire the hook into the framework's event system so a
   completed turn fires ``hook.on_turn_end``.

Managed repair and read-only guidance are framework-agnostic and live in the
facade. Adapters expose no injection or administrative surface.
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

    def register(self, hook: PandaHarnessHook) -> None:
        """Wire the hook into the framework's lifecycle callbacks."""
        ...
