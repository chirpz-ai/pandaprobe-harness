"""The trajectory gate — a Tier-1 breach is a *trend*, never a point threshold.

v1 breached whenever a score sat below an absolute floor. On a session-aggregate
metric that fires on essentially everything; on a per-trace metric it would fire
on every early trace of every task, because an agent three steps into a job has
legitimately not completed it yet. Neither carries signal.

So Tier-1 metrics breach on the *shape* of the series instead:

* **STALL** — no gain toward ``gate_target`` for ``gate_window`` consecutive
  traces, while the best score so far is still below the target. The agent is
  working but not getting closer.
* **REGRESSION** — the current value drops ``gate_drop`` below the running peak.
  The agent had it and lost it.
* **RESET-ON-GAIN** — any improvement of at least ``gate_gain`` resets the peak
  and the counter, so *a healthy climbing session never breaches* however long it
  runs. This is the property a smoothed moving average cannot give you.

After firing, the counter resets so one stall does not re-fire on every
subsequent trace — the agent gets a notice, a chance to write a rule, and a fresh
window to show improvement.

State is per ``(session, metric)`` and lives in the shared score-history file, so
it is O(1) per update, survives restarts, and needs no network call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ..config import HarnessConfig
from .history import GateFold, GateState
from .history_source import HistorySource
from .metrics import MetricScore

__all__ = ["GateVerdict", "TrajectoryGate"]


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """The gate's decision for one observed value."""

    metric: str
    value: float
    peak: float
    turns_since_gain: int
    stalled: bool
    regressed: bool

    @property
    def breached(self) -> bool:
        return self.stalled or self.regressed

class TrajectoryGate:
    """Peak/stall/regression detection over a Tier-1 metric's per-trace series."""

    def __init__(self, config: HarnessConfig, store: HistorySource) -> None:
        self._config = config
        self._store = store

    def _decide(self, state: GateState, value: float) -> tuple[GateState, bool, bool]:
        """The gate, as a pure function: ``(next_state, stalled, regressed)``."""

        cfg = self._config
        if state.peak is None or value >= state.peak + cfg.gate_gain:
            # Progressing: adopt the new high-water mark and reset the window.
            peak = value if state.peak is None else max(state.peak, value)
            return GateState(peak=peak, turns_since_gain=0), False, False

        turns = state.turns_since_gain + 1
        stalled = turns >= max(1, cfg.gate_window) and state.peak < cfg.gate_target
        regressed = value < state.peak - cfg.gate_drop
        if stalled or regressed:
            turns = 0  # fire once, then start a fresh window
        return replace(state, turns_since_gain=turns), stalled, regressed

    def update(self, session_id: str, metric: str, value: float) -> GateVerdict:
        """Fold ``value`` into the gate state and decide whether it breaches."""

        return self.apply_many(session_id, {metric: value})[metric]

    def apply_many(
        self, session_id: str, values: Mapping[str, float]
    ) -> dict[str, GateVerdict]:
        """Fold a whole trace's Tier-1 values in one store write.

        Samples are also appended to the series inspected by operators and read
        by ``calibration.collect_scores``.
        """

        verdicts: dict[str, GateVerdict] = {}

        def _fold_for(metric: str, value: float) -> GateFold:
            def _fold(state: GateState) -> GateState:
                nxt, stalled, regressed = self._decide(state, value)
                verdicts[metric] = GateVerdict(
                    metric=metric,
                    value=value,
                    peak=nxt.peak if nxt.peak is not None else value,
                    turns_since_gain=nxt.turns_since_gain,
                    stalled=stalled,
                    regressed=regressed,
                )
                return nxt

            return _fold

        self._store.record_gated(
            session_id,
            [(metric, value, _fold_for(metric, value)) for metric, value in values.items()],
        )
        return verdicts

    def apply(self, session_id: str, score: MetricScore) -> MetricScore:
        """Record ``score``'s value and return it annotated with the gate flags."""

        return self.apply_all(session_id, [score])[0]

    def apply_all(
        self, session_id: str, scores: Sequence[MetricScore]
    ) -> list[MetricScore]:
        """Annotate a trace's scores with their gate flags, in one store write.

        Pending (``None``) scores pass through untouched: the harness never alerts
        on the absence of a score, and folding a missing value into the peak would
        corrupt the trajectory.
        """

        resolved = {
            str(score.metric): score.value
            for score in scores
            if score.value is not None
        }
        if not resolved:
            return list(scores)
        verdicts = self.apply_many(session_id, resolved)
        out: list[MetricScore] = []
        for score in scores:
            verdict = verdicts.get(str(score.metric)) if score.value is not None else None
            out.append(
                score
                if verdict is None
                else replace(score, stalled=verdict.stalled, regressed=verdict.regressed)
            )
        return out
