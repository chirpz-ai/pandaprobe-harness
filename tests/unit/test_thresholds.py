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
    assert MetricScore(Metric.RELIABILITY, 0.3, 0.5).breached is True
    assert MetricScore(Metric.RELIABILITY, 0.6, 0.5).breached is False
    assert MetricScore(Metric.RELIABILITY, None, 0.5).breached is False
    assert MetricScore(Metric.RELIABILITY, None, 0.5).pending is True


def test_metric_targets() -> None:
    # Both agent_reliability and agent_consistency are session-level metrics.
    assert Metric.RELIABILITY.target == "session"
    assert Metric.CONSISTENCY.target == "session"
