"""Low-latency trend detection over per-(session, metric) score history.

Primary detector: a **dual-EWMA crossover** (fast vs slow exponential moving
averages). It consumes the score the harness already obtained for the turn and
runs in O(1) against the persisted EWMA state — **no extra CLI/network call on
the turn path**, so it adds effectively no latency while being recency-weighted
(matches "gradually declining over turns") and noise-robust (smoothing rejects
single-turn jitter).

Two further, optional conditions reuse the same local store (still no network):
* an **adaptive (relative) threshold** — a single turn dropping far below its own
  session baseline (the slow EWMA), even while above the absolute floor;
* a **percentile-over-window** corroborator — the latest score sitting in the
  bottom quantile of the recent local window (off by default).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import HarnessConfig
from .history import ScoreHistoryStore

__all__ = ["TrendVerdict", "TrendDetector"]


@dataclass(frozen=True, slots=True)
class TrendVerdict:
    metric: str
    value: float
    fast: float
    slow: float
    count: int
    declining: bool
    relative_breach: bool
    percentile_breach: bool

    @property
    def alerting(self) -> bool:
        return self.declining or self.relative_breach or self.percentile_breach


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated ``q``-quantile of ``values`` (q in [0, 1])."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


class TrendDetector:
    """Detects declining/relative/percentile conditions for a metric."""

    def __init__(self, config: HarnessConfig, store: ScoreHistoryStore) -> None:
        self._config = config
        self._store = store

    def update(
        self, session_id: str, metric: str, value: float, *, run_id: str | None = None
    ) -> TrendVerdict:
        """Record ``value`` and evaluate trend conditions for it."""

        cfg = self._config
        state = self._store.record(session_id, metric, value, run_id=run_id)
        enough = state.count >= cfg.trend_min_samples

        declining = enough and state.fast < (state.slow - cfg.trend_margin_cross)

        relative_breach = (
            cfg.adaptive_threshold
            and enough
            and value < (state.slow - cfg.adaptive_margin_drop)
        )

        percentile_breach = False
        if cfg.percentile_window > 0:
            window = self._store.values(session_id, metric)[-cfg.percentile_window :]
            # Clamp the minimum-sample gate to the window size so a small window
            # (< trend_min_samples) does not make the corroborator a silent no-op.
            needed = min(max(cfg.trend_min_samples, 2), cfg.percentile_window)
            if len(window) >= needed:
                percentile_breach = value < _percentile(window, cfg.percentile_floor)

        return TrendVerdict(
            metric=metric,
            value=value,
            fast=state.fast,
            slow=state.slow,
            count=state.count,
            declining=declining,
            relative_breach=relative_breach,
            percentile_breach=percentile_breach,
        )
