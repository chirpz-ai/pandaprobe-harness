"""The trace-target evaluation path (v2)."""

from __future__ import annotations

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.evaluation.evaluator import MetricEvaluator
from pandaprobe_harness.evaluation.metrics import Metric
from tests.fakes.fake_cli_client import FakeCliClient


def _cfg(**kw: object) -> HarnessConfig:
    base: dict[str, object] = {
        "poll_interval_s": 0.0,
        "poll_max_attempts": 5,
        "eval_retry_attempts": 3,
        "eval_retry_backoff_s": 0.0,
    }
    base.update(kw)
    return HarnessConfig(**base)  # type: ignore[arg-type]


async def test_trace_run_targets_trace_and_passes_trace_ids() -> None:
    cli = FakeCliClient()
    scores = await MetricEvaluator(cli, _cfg()).evaluate_trace(
        "tr-1", ["task_completion", "coherence"], tier=1
    )

    assert len(cli.batch_calls) == 1
    joined = " ".join(cli.batch_calls[0])
    assert "--target trace" in joined
    assert "--trace-ids tr-1" in joined
    assert "--session-ids" not in joined
    # Scores are polled with --target trace.
    score_calls = [c for c in cli.calls if c[:3] == ("evals", "runs", "scores")]
    assert score_calls and "trace" in score_calls[0]
    # Every score is attributed to its trace and tier.
    assert {s.metric for s in scores} == {Metric.TASK_COMPLETION, Metric.COHERENCE}
    assert all(s.trace_id == "tr-1" and s.tier == 1 for s in scores)


async def test_signal_weights_are_session_only() -> None:
    """`--signal-weights` is a session-only flag; sending it on the trace path
    would be a validation error."""

    weights = {"tool_correctness": 2.0}
    cli = FakeCliClient()
    await MetricEvaluator(cli, _cfg(signal_weights=weights)).evaluate_trace(
        "tr-1", ["task_completion"]
    )
    assert "--signal-weights" not in " ".join(cli.batch_calls[0])


async def test_non_trace_runnable_metrics_are_dropped() -> None:
    """The session composites are not trace-runnable (the platform 422s), so the
    evaluator must never request them against a trace."""

    cli = FakeCliClient()
    scores = await MetricEvaluator(cli, _cfg()).evaluate_trace(
        "tr-1", ["agent_reliability", "task_completion"]
    )
    assert {s.metric for s in scores} == {Metric.TASK_COMPLETION}
    assert "agent_reliability" not in " ".join(cli.batch_calls[0])


async def test_unknown_metric_is_skipped_without_raising() -> None:
    cli = FakeCliClient()
    scores = await MetricEvaluator(cli, _cfg()).evaluate_trace("tr-1", ["not_a_metric"])
    assert scores == []
    assert cli.batch_calls == []  # nothing to ask for → no CLI call at all


async def test_threshold_falls_back_to_the_platform_reported_one() -> None:
    cli = FakeCliClient(metric_metadata={"task_completion": {"threshold": 0.75}})
    scores = await MetricEvaluator(cli, _cfg()).evaluate_trace("tr-1", ["task_completion"])
    assert scores[0].threshold == 0.75


async def test_local_threshold_overrides_the_reported_one() -> None:
    cli = FakeCliClient(metric_metadata={"task_completion": {"threshold": 0.75}})
    config = _cfg(thresholds={"task_completion": 0.4})
    scores = await MetricEvaluator(cli, config).evaluate_trace("tr-1", ["task_completion"])
    assert scores[0].threshold == 0.4


async def test_score_last_trace_picks_the_newest_completed_trace() -> None:
    cli = FakeCliClient()
    cli.script_trajectory("s-1", "task_completion", [0.1, 0.2, 0.9])

    scores = await MetricEvaluator(cli, _cfg()).score_last_trace("s-1", ["task_completion"])

    assert [s.value for s in scores] == [0.9]
    assert scores[0].trace_id == "s-1-tr3"


async def test_score_last_trace_degrades_when_no_trace_exists() -> None:
    cli = FakeCliClient(auto_traces=False)
    cli.session_traces["s-1"] = []
    assert await MetricEvaluator(cli, _cfg()).score_last_trace("s-1", ["task_completion"]) == []


async def test_tier1_absolute_floor_is_not_a_breach() -> None:
    """Tier 1 breaches on trajectory only: an early trace legitimately scores
    low, so its absolute value must not raise a breach."""

    cli = FakeCliClient(metric_values={"task_completion": 0.1})
    scores = await MetricEvaluator(cli, _cfg()).evaluate_trace(
        "tr-1", ["task_completion"], tier=1
    )
    assert scores[0].below_threshold is True
    assert scores[0].breached is False
    assert scores[0].alerting is False


async def test_tier2_absolute_floor_is_a_breach() -> None:
    cli = FakeCliClient(metric_values={"tool_correctness": 0.1})
    scores = await MetricEvaluator(cli, _cfg()).evaluate_trace(
        "tr-1", ["tool_correctness"], tier=2
    )
    assert scores[0].breached is True
