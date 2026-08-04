"""The seam for pluggable score-history backends.

``ScoreHistoryStore`` (local JSON) is the default implementation. A remote
store can satisfy this Protocol to share trajectory state across replicas.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .history import GateFold, GateState

__all__ = ["HistorySource"]


@runtime_checkable
class HistorySource(Protocol):
    """Per-``(session, metric)`` score series with trajectory-gate state."""

    def record_gated(
        self,
        session_id: str,
        samples: Sequence[tuple[str, float, GateFold]],
    ) -> dict[str, GateState]:
        """Append samples and atomically fold their trajectory state."""
        ...

    def values(self, session_id: str, metric: str) -> list[float]:
        """The recorded score values, oldest first."""
        ...
