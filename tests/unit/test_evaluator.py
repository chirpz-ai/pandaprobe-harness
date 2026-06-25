from __future__ import annotations

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.cli.errors import CliAuthError
from pandaprobe_harness.evaluation.evaluator import MetricEvaluator
from pandaprobe_harness.evaluation.metrics import Metric
from pandaprobe_harness.hook.turn import TurnContext
from tests.fakes.fake_cli_client import FakeCliClient

CTX = TurnContext(session_id="s-1", turn_index=1, end_state={})


def _cfg(**kw) -> HarnessConfig:
    base = dict(poll_interval_s=0.0, poll_max_attempts=5)
    base.update(kw)
    return HarnessConfig(**base)


async def test_both_metrics_breach() -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.3, "agent_consistency": 0.4})
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    assert report.any_breach
    assert {s.metric for s in report.breached_scores} == {
        Metric.RELIABILITY,
        Metric.CONSISTENCY,
    }


async def test_passing_scores_no_breach() -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9, "agent_consistency": 0.8})
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    assert not report.any_breach


async def test_reliability_uses_trace_target_consistency_uses_session() -> None:
    cli = FakeCliClient()
    await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    batch_calls = [c for c in cli.calls if c[:3] == ("evals", "runs", "batch")]
    joined = [" ".join(c) for c in batch_calls]
    assert any(
        "--target trace" in j and "agent_reliability" in j and "--trace-ids" in j
        for j in joined
    )
    assert any(
        "--target session" in j and "agent_consistency" in j and "--session-ids" in j
        for j in joined
    )


async def test_selective_flag_skips_metric() -> None:
    cli = FakeCliClient()
    cfg = _cfg(eval_consistency=False)
    report = await MetricEvaluator(cli, cfg).evaluate_turn(CTX)
    assert {s.metric for s in report.scores} == {Metric.RELIABILITY}
    assert not any(
        c[:3] == ("evals", "runs", "batch") and "agent_consistency" in c
        for c in cli.calls
    )


async def test_polling_loops_until_terminal() -> None:
    cli = FakeCliClient(running_polls=2)  # 2 'running' then completed
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    score_calls = [c for c in cli.calls if c[:3] == ("evals", "runs", "scores")]
    # at least 3 score polls per metric (2 running + 1 terminal) * 2 metrics
    assert len(score_calls) >= 6
    assert all(s.value is not None for s in report.scores)


async def test_poll_budget_exhausted_yields_pending() -> None:
    cli = FakeCliClient(running_polls=99)  # never terminal
    cfg = _cfg(poll_max_attempts=3)
    report = await MetricEvaluator(cli, cfg).evaluate_turn(CTX)
    assert all(s.pending for s in report.scores)
    assert not report.any_breach  # pending is never a breach


async def test_no_traces_skips_reliability() -> None:
    cli = FakeCliClient(trace_ids=[])
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    reliability = next(s for s in report.scores if s.metric is Metric.RELIABILITY)
    assert reliability.pending


async def test_cli_error_degrades_to_pending() -> None:
    cli = FakeCliClient(
        error_on_prefix={("evals", "runs", "batch"): CliAuthError("no auth")}
    )
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    assert all(s.pending for s in report.scores)
    assert not report.any_breach


async def test_sequential_mode(monkeypatch) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.3, "agent_consistency": 0.4})
    cfg = _cfg(concurrent_eval=False)
    report = await MetricEvaluator(cli, cfg).evaluate_turn(CTX)
    assert report.any_breach
