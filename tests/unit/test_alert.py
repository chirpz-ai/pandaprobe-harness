from __future__ import annotations

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.evaluation.metrics import EvalReport, Metric, MetricScore
from pandaprobe_harness.hook.alert import build_system_alert, build_trend_alert


def _breach_report() -> EvalReport:
    return EvalReport.from_scores(
        "s-1",
        2,
        [
            MetricScore(
                Metric.RELIABILITY,
                0.30,
                0.5,
                reason="tail risk",
                metadata={"flagged_traces": ["trace-7"]},
            ),
            MetricScore(Metric.CONSISTENCY, 0.90, 0.5),  # not breached
        ],
    )


def _trend_report() -> EvalReport:
    return EvalReport.from_scores(
        "s-1",
        5,
        [MetricScore(Metric.RELIABILITY, 0.62, 0.5, trend_declining=True)],
    )


def test_system_alert_names_files_commands_and_flagged_trace() -> None:
    cfg = HarnessConfig(harness_root="/harness")
    alert = build_system_alert(_breach_report(), cfg)
    assert "SYSTEM ALERT" in alert
    assert str(cfg.latest_eval_file) in alert
    assert str(cfg.rules_file) in alert
    assert "pandaprobe" in alert
    assert "trace-7" in alert  # flagged trace surfaced + used in inspect command


def test_system_alert_lists_only_alerting_metric() -> None:
    alert = build_system_alert(_breach_report(), HarnessConfig(harness_root="/harness"))
    assert "agent_reliability" in alert
    assert "absolute breach" in alert
    # consistency passed, so it is not listed as a condition
    assert "agent_consistency" not in alert


def test_trend_alert_is_advisory() -> None:
    alert = build_trend_alert(_trend_report(), HarnessConfig(harness_root="/harness"))
    assert "TREND ALERT" in alert
    assert "declining trend" in alert
    assert "agent_reliability" in alert
