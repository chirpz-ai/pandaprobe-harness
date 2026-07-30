"""The trajectory gate: stall, regression, reset-on-gain, fire-then-reset."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.evaluation.history import ScoreHistoryStore
from pandaprobe_harness.evaluation.metrics import Metric, MetricScore
from pandaprobe_harness.evaluation.trajectory import GateVerdict, TrajectoryGate

METRIC = "task_completion"


def _gate(tmp_path: Path, **kw: object) -> TrajectoryGate:
    cfg = HarnessConfig(harness_root=tmp_path / "h", **kw)  # type: ignore[arg-type]
    return TrajectoryGate(cfg, ScoreHistoryStore(cfg))


def _feed(gate: TrajectoryGate, values: Sequence[float]) -> list[GateVerdict]:
    return [gate.update("s", METRIC, v) for v in values]


def test_healthy_climb_never_breaches(tmp_path: Path) -> None:
    """The defining property: an agent making progress is never flagged, however
    long the session runs and however low it started."""

    verdicts = _feed(_gate(tmp_path, gate_window=3), [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])

    assert not any(v.breached for v in verdicts)
    assert verdicts[-1].peak == 1.0


def test_stall_below_target_breaches_after_the_window(tmp_path: Path) -> None:
    verdicts = _feed(_gate(tmp_path, gate_window=3, gate_target=0.5), [0.2, 0.2, 0.2, 0.2])

    # The first value sets the peak; three more with no gain close the window.
    assert [v.stalled for v in verdicts] == [False, False, False, True]
    assert verdicts[-1].breached


def test_a_plateau_at_or_above_target_does_not_stall(tmp_path: Path) -> None:
    """Stall means "plateaued *below* the target" — an agent holding a good score
    is not failing."""

    verdicts = _feed(_gate(tmp_path, gate_window=3, gate_target=0.5), [0.8, 0.8, 0.8, 0.8, 0.8])
    assert not any(v.breached for v in verdicts)


def test_regression_from_the_peak_breaches_immediately(tmp_path: Path) -> None:
    verdicts = _feed(_gate(tmp_path, gate_window=10, gate_drop=0.15), [0.9, 0.6])

    assert verdicts[1].regressed
    assert not verdicts[1].stalled  # the window is nowhere near closed
    assert verdicts[1].peak == 0.9


def test_a_drop_within_the_margin_is_noise_not_regression(tmp_path: Path) -> None:
    verdicts = _feed(_gate(tmp_path, gate_window=10, gate_drop=0.15), [0.9, 0.8])
    assert not verdicts[1].breached


def test_gain_resets_the_stall_window(tmp_path: Path) -> None:
    gate = _gate(tmp_path, gate_window=3, gate_target=0.9, gate_gain=0.02)

    # Two no-gain traces, then real progress, then two more no-gain traces: the
    # window restarted, so nothing fires.
    verdicts = _feed(gate, [0.2, 0.2, 0.2, 0.5, 0.5, 0.5])

    assert [v.turns_since_gain for v in verdicts] == [0, 1, 2, 0, 1, 2]
    assert not any(v.breached for v in verdicts)


def test_stall_fires_once_then_starts_a_fresh_window(tmp_path: Path) -> None:
    """After firing, the agent gets a new window to show improvement rather than
    a notice on every subsequent trace."""

    verdicts = _feed(
        _gate(tmp_path, gate_window=2, gate_target=0.5), [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    )

    assert [v.stalled for v in verdicts] == [False, False, True, False, True, False]


def test_gate_state_survives_a_new_store_instance(tmp_path: Path) -> None:
    cfg = HarnessConfig(harness_root=tmp_path / "h", gate_window=3, gate_target=0.5)

    _feed(TrajectoryGate(cfg, ScoreHistoryStore(cfg)), [0.2, 0.2, 0.2])
    # A fresh store re-reads the persisted peak and counter, so the fourth trace
    # closes the window that the first three opened.
    reborn = TrajectoryGate(cfg, ScoreHistoryStore(cfg))

    assert reborn.update("s", METRIC, 0.2).stalled


def test_sessions_do_not_share_a_trajectory(tmp_path: Path) -> None:
    gate = _gate(tmp_path, gate_window=2, gate_target=0.5)

    for value in (0.2, 0.2, 0.2):
        gate.update("s-a", METRIC, value)
    # A different session starts from scratch — one value cannot close a window.
    assert not gate.update("s-b", METRIC, 0.2).breached


def test_update_appends_to_the_series(tmp_path: Path) -> None:
    """The series is what the operator inspects and what calibration reads, so
    the trace path must not leave it empty."""

    cfg = HarnessConfig(harness_root=tmp_path / "h")
    store = ScoreHistoryStore(cfg)
    _feed(TrajectoryGate(cfg, store), [0.1, 0.4, 0.7])

    assert store.values("s", METRIC) == [0.1, 0.4, 0.7]


def test_a_traces_scores_cost_one_store_write(tmp_path: Path) -> None:
    """The store persists by re-serializing the whole file, so folding a trace's
    Tier-1 metrics must take one write — not one per metric, and not two per metric
    (append then fold), which is what the naive shape costs."""

    cfg = HarnessConfig(harness_root=tmp_path / "h")
    store = ScoreHistoryStore(cfg)
    gate = TrajectoryGate(cfg, store)
    writes = 0
    original = store._persist  # noqa: SLF001 - counting I/O is the point of the test

    def counting() -> None:
        nonlocal writes
        writes += 1
        original()

    store._persist = counting  # type: ignore[method-assign]

    gated = gate.apply_all(
        "s",
        [
            MetricScore(metric=Metric.TASK_COMPLETION, value=0.2, threshold=0.5, tier=1),
            MetricScore(metric=Metric.COHERENCE, value=0.4, threshold=0.5, tier=1),
        ],
    )

    assert writes == 1
    assert len(gated) == 2
    # Both series were still recorded.
    assert store.values("s", METRIC) == [0.2]
    assert store.values("s", "coherence") == [0.4]


def test_apply_annotates_a_score_and_passes_pending_through(tmp_path: Path) -> None:
    gate = _gate(tmp_path, gate_window=1, gate_target=0.5, gate_drop=0.15)

    first = gate.apply("s", MetricScore(metric=Metric.TASK_COMPLETION, value=0.2, threshold=0.5))
    second = gate.apply("s", MetricScore(metric=Metric.TASK_COMPLETION, value=0.2, threshold=0.5))
    pending = gate.apply("s", MetricScore(metric=Metric.TASK_COMPLETION, value=None, threshold=0.5))

    assert not first.stalled
    assert second.stalled and second.gate_breached
    # A missing score is never an alert, and must not pollute the peak.
    assert not pending.gate_breached
    assert gate.update("s", METRIC, 0.2).peak == 0.2
