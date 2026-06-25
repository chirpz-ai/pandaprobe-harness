from __future__ import annotations

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.evaluation.metrics import EvalReport, Metric, MetricScore
from pandaprobe_harness.hook.alert import build_system_alert


def _report() -> EvalReport:
    return EvalReport.from_scores(
        "s-1",
        2,
        [
            MetricScore(Metric.RELIABILITY, 0.30, 0.5, reason="tail risk"),
            MetricScore(Metric.CONSISTENCY, 0.90, 0.5),  # not breached
        ],
    )


def test_alert_names_files_and_commands() -> None:
    cfg = HarnessConfig(harness_root="/harness")
    alert = build_system_alert(_report(), cfg)
    assert "SYSTEM ALERT" in alert
    assert str(cfg.latest_eval_file) in alert
    assert str(cfg.rules_file) in alert
    assert "pandaprobe" in alert


def test_alert_lists_only_breached_metric_with_score() -> None:
    alert = build_system_alert(_report(), HarnessConfig(harness_root="/harness"))
    assert "agent_reliability = 0.30" in alert
    # consistency passed, so it should not appear in the breach list line
    assert "agent_consistency = 0.90" not in alert
