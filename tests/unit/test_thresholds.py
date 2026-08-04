from __future__ import annotations

from pandaprobe_harness.evaluation import is_breach
from pandaprobe_harness.evaluation.metrics import Metric, MetricScore


def test_below_threshold_is_breach() -> None:
    assert is_breach(0.49, 0.5) is True


def test_equal_threshold_is_not_breach() -> None:
    assert is_breach(0.5, 0.5) is False


def test_above_threshold_is_not_breach() -> None:
    assert is_breach(0.9, 0.5) is False


def test_none_is_not_breach() -> None:
    assert is_breach(None, 0.5) is False


def test_metric_score_breached_property() -> None:
    assert MetricScore(Metric.OUTCOME, 0.3, 0.5).breached is True
    assert MetricScore(Metric.OUTCOME, 0.6, 0.5).breached is False
    assert MetricScore(Metric.OUTCOME, None, 0.5).breached is False
    assert MetricScore(Metric.OUTCOME, None, 0.5).pending is True
    assert MetricScore(Metric.TASK_COMPLETION, 0.3, 0.5, tier=1).breached is False


def test_metric_targets() -> None:
    assert Metric.TASK_COMPLETION.target == "trace"
    assert Metric.COHERENCE.target == "trace"
    assert Metric.OUTCOME.target == "local"
