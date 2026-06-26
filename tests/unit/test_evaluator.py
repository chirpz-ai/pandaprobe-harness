from __future__ import annotations

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.cli.errors import CliAuthError, CliNotFoundError, CliValidationError
from pandaprobe_harness.evaluation.evaluator import MetricEvaluator
from pandaprobe_harness.evaluation.metrics import Metric
from pandaprobe_harness.hook.turn import TurnContext
from tests.fakes.fake_cli_client import FakeCliClient

CTX = TurnContext(session_id="s-1", turn_index=1, end_state={})


def _cfg(**kw: object) -> HarnessConfig:
    base: dict[str, object] = {
        "poll_interval_s": 0.0,
        "poll_max_attempts": 5,
        "eval_retry_attempts": 3,
        "eval_retry_backoff_s": 0.0,
    }
    base.update(kw)
    return HarnessConfig(**base)  # type: ignore[arg-type]


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


async def test_single_session_run_with_both_metrics() -> None:
    cli = FakeCliClient()
    await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    # Exactly one batch run covering both metrics, session-scoped.
    assert len(cli.batch_calls) == 1
    joined = " ".join(cli.batch_calls[0])
    assert "--target session" in joined
    assert "--session-ids s-1" in joined
    assert "agent_reliability" in joined and "agent_consistency" in joined
    # Scores are polled with --target session.
    score_calls = [c for c in cli.calls if c[:3] == ("evals", "runs", "scores")]
    assert score_calls and "--target" in score_calls[0] and "session" in score_calls[0]


async def test_selective_flag_skips_metric() -> None:
    cli = FakeCliClient()
    report = await MetricEvaluator(cli, _cfg(eval_consistency=False)).evaluate_turn(CTX)
    assert {s.metric for s in report.scores} == {Metric.RELIABILITY}
    joined = " ".join(cli.batch_calls[0])
    assert "agent_reliability" in joined
    assert "agent_consistency" not in joined


async def test_polling_loops_until_terminal() -> None:
    cli = FakeCliClient(running_polls=2)  # 2 PENDING rounds then SUCCESS
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    score_calls = [c for c in cli.calls if c[:3] == ("evals", "runs", "scores")]
    assert len(score_calls) >= 3  # one run, polled to terminal
    assert all(s.value is not None for s in report.scores)


async def test_signal_weights_passthrough() -> None:
    cli = FakeCliClient()
    cfg = _cfg(signal_weights={"confidence": 1.0, "coherence": 0.5})
    await MetricEvaluator(cli, cfg).evaluate_turn(CTX)
    joined = " ".join(cli.batch_calls[0])
    assert "--signal-weights" in joined
    assert "confidence" in joined


async def test_transient_empty_then_populated_retries() -> None:
    # First run yields no scores (trace-ingestion lag); retry produces a breach.
    cli = FakeCliClient(
        empty_runs=1, metric_values={"agent_reliability": 0.3, "agent_consistency": 0.4}
    )
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    assert len(cli.batch_calls) >= 2  # retried after the empty run
    assert report.any_breach


async def test_poll_budget_exhausted_yields_pending() -> None:
    cli = FakeCliClient(running_polls=99)  # never terminal
    cfg = _cfg(poll_max_attempts=2, eval_retry_attempts=1)
    report = await MetricEvaluator(cli, cfg).evaluate_turn(CTX)
    assert all(s.pending for s in report.scores)
    assert not report.any_breach


async def test_auth_error_degrades_without_retry() -> None:
    cli = FakeCliClient(
        error_on_prefix={("evals", "runs", "batch"): CliAuthError("401 unauthorized")}
    )
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    assert all(s.pending for s in report.scores)
    assert not report.any_breach
    assert len(cli.batch_calls) == 1  # auth is not transient → no retry


async def test_not_found_is_transient_and_retried() -> None:
    cli = FakeCliClient(
        error_on_prefix={("evals", "runs", "batch"): CliNotFoundError("404 not found")}
    )
    report = await MetricEvaluator(cli, _cfg(eval_retry_attempts=3)).evaluate_turn(CTX)
    assert all(s.pending for s in report.scores)
    assert len(cli.batch_calls) == 3  # retried up to the attempt budget


async def test_concurrent_flag_is_noop() -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.3, "agent_consistency": 0.4})
    report = await MetricEvaluator(cli, _cfg(concurrent_eval=False)).evaluate_turn(CTX)
    assert report.any_breach
    assert len(cli.batch_calls) == 1


async def test_validation_error_degrades_without_retry() -> None:
    cli = FakeCliClient(
        error_on_prefix={("evals", "runs", "batch"): CliValidationError("invalid metric")}
    )
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    assert all(s.pending for s in report.scores)
    assert not report.any_breach
    assert len(cli.batch_calls) == 1  # validation is not transient → no retry


async def test_terminal_failed_score_is_pending_not_breach() -> None:
    # A terminal FAILED score has a null value; it must be pending, not a breach.
    cli = FakeCliClient(
        metric_status={"agent_reliability": "FAILED", "agent_consistency": "FAILED"}
    )
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    assert all(s.pending for s in report.scores)
    assert not report.any_breach
    # One poll suffices: FAILED is terminal.
    score_calls = [c for c in cli.calls if c[:3] == ("evals", "runs", "scores")]
    assert len(score_calls) == 1


async def test_non_numeric_value_degrades_to_pending() -> None:
    cli = FakeCliClient(raw_metric_values={"agent_reliability": "N/A", "agent_consistency": ""})
    report = await MetricEvaluator(cli, _cfg()).evaluate_turn(CTX)
    assert all(s.pending for s in report.scores)
    assert not report.any_breach
