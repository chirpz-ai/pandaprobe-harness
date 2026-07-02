"""The seam for pluggable score-history backends.

``ScoreHistoryStore`` (local JSON) is the default implementation. A future
remote store only has to satisfy this Protocol; horizontally-scaled agents
already converge on shared backend state via the hook's one-time-per-session
hydration (``evals scores list --target session`` → :meth:`HistorySource.seed`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .history import EwmaState

__all__ = ["HistorySource"]


@runtime_checkable
class HistorySource(Protocol):
    """Per-``(session, metric)`` score series with incremental EWMA state."""

    def record(
        self,
        session_id: str,
        metric: str,
        value: float,
        *,
        run_id: str | None = None,
        ts: str | None = None,
    ) -> EwmaState:
        """Append one score and return the updated EWMA state."""
        ...

    def values(self, session_id: str, metric: str) -> list[float]:
        """The recorded score values, oldest first."""
        ...

    def seed(
        self,
        session_id: str,
        metric: str,
        samples: Sequence[tuple[float, str, str | None]],
    ) -> None:
        """Bulk-insert backend samples ``(value, ts, run_id)``, idempotently."""
        ...
