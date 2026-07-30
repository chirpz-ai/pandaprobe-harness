"""The three-tier escalation ladder."""

from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.evaluation.evaluator import MetricEvaluator
from pandaprobe_harness.evaluation.history import ScoreHistoryStore
from pandaprobe_harness.evaluation.metrics import Metric
from pandaprobe_harness.evaluation.trajectory import TrajectoryGate
from pandaprobe_harness.hook.tiers import TierRunner
from pandaprobe_harness.hook.turn import TurnContext
from tests.fakes.fake_cli_client import FakeCliClient

CTX = TurnContext(session_id="s-1", turn_index=1, end_state={})


def _runner(
    tmp_path: Path, cli: FakeCliClient, **kw: object
) -> tuple[TierRunner, HarnessConfig]:
    verifier = kw.pop("verifier", None)
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_attempts=1,
        eval_retry_backoff_s=0.0,
        gate_window=2,
        **kw,  # type: ignore[arg-type]
    )
    evaluator = MetricEvaluator(cli, cfg)
    return (
        TierRunner(
            cfg,
            evaluator,
            evaluator.locator,
            TrajectoryGate(cfg, ScoreHistoryStore(cfg)),
            verifier=verifier,  # type: ignore[arg-type]
        ),
        cfg,
    )


def _requested(cli: FakeCliClient) -> list[set[str]]:
    """The metric set of each batch run, in order."""

    out: list[set[str]] = []
    for call in cli.batch_calls:
        joined = list(call)
        idx = joined.index("--metrics")
        out.append(set(joined[idx + 1].split(",")))
    return out


async def test_tier1_runs_on_every_new_trace(tmp_path: Path) -> None:
    cli = FakeCliClient()
    cli.script_trajectory("s-1", "task_completion", [0.3, 0.5, 0.9])
    runner, _ = _runner(tmp_path, cli)

    report = await runner.run(CTX)

    tier1 = report.scores_for_tier(1)
    assert {s.trace_id for s in tier1} == {"s-1-tr1", "s-1-tr2", "s-1-tr3"}
    assert all(s.tier == 1 for s in tier1)


async def test_a_healthy_climb_never_escalates(tmp_path: Path) -> None:
    cli = FakeCliClient()
    cli.script_trajectory("s-1", "task_completion", [0.2, 0.5, 0.8, 1.0])
    runner, _ = _runner(tmp_path, cli)

    report = await runner.run(CTX)

    assert not report.gate_breached
    assert report.scores_for_tier(2) == ()
    # Only Tier-1 metric sets were ever requested — no Tier-2 cost incurred.
    assert all(names == {"task_completion", "coherence"} for names in _requested(cli))


async def test_tier2_runs_only_on_the_last_trace_once_the_gate_opens(
    tmp_path: Path,
) -> None:
    cli = FakeCliClient(metric_values={"tool_correctness": 0.2, "argument_correctness": 0.9})
    # A flat, below-target series: the gate stalls on the third trace.
    cli.script_trajectory("s-1", "task_completion", [0.2, 0.2, 0.2])
    runner, _ = _runner(tmp_path, cli)

    report = await runner.run(CTX)

    assert report.gate_breached
    tier2 = report.scores_for_tier(2)
    assert {s.metric for s in tier2} == {Metric.TOOL_CORRECTNESS, Metric.ARGUMENT_CORRECTNESS}
    # Exactly one Tier-2 run, against the last trace only.
    assert {s.trace_id for s in tier2} == {"s-1-tr3"}
    assert _requested(cli).count({"tool_correctness", "argument_correctness"}) == 1


async def test_tier1_scores_the_whole_turn_in_one_platform_run(tmp_path: Path) -> None:
    """The batch endpoint takes every trace id at once, so a turn costs one eval
    run rather than one per trace — the difference is N round-trips of judge
    latency on the barrier path."""

    cli = FakeCliClient()
    cli.script_trajectory("s-1", "task_completion", [0.2, 0.5, 0.8, 1.0])
    runner, _ = _runner(tmp_path, cli)

    report = await runner.run(CTX)

    assert len(cli.batch_calls) == 1
    joined = " ".join(cli.batch_calls[0])
    assert "--trace-ids s-1-tr1,s-1-tr2,s-1-tr3,s-1-tr4" in joined
    # Every trace still gets its own attributed scores.
    assert {s.trace_id for s in report.scores_for_tier(1)} == {
        "s-1-tr1", "s-1-tr2", "s-1-tr3", "s-1-tr4",
    }


async def test_tier3_never_breaches_on_its_own(tmp_path: Path) -> None:
    """Tier 3 exists only to explain a confirmed Tier-2 breach, so its own scores
    must not be a breach source — otherwise a low plan_quality would drive
    severity and signatures by itself."""

    cli = FakeCliClient(metric_values={"tool_correctness": 0.2, "plan_quality": 0.1})
    cli.script_trajectory("s-1", "task_completion", [0.2, 0.2, 0.2])
    runner, _ = _runner(tmp_path, cli, enable_tier3=True)

    report = await runner.run(CTX)

    tier3 = report.scores_for_tier(3)
    plan_quality = next(s for s in tier3 if str(s.metric) == "plan_quality")
    assert plan_quality.below_threshold is True  # still reported for diagnosis
    assert plan_quality.breached is False
    assert plan_quality.conditions == ()


async def test_tier3_is_off_by_default_even_on_a_tier2_breach(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"tool_correctness": 0.2})
    cli.script_trajectory("s-1", "task_completion", [0.2, 0.2, 0.2])
    runner, _ = _runner(tmp_path, cli)

    report = await runner.run(CTX)

    assert any(s.breached for s in report.scores_for_tier(2))
    assert report.scores_for_tier(3) == ()


async def test_tier3_enriches_a_confirmed_tier2_breach(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"tool_correctness": 0.2})
    cli.script_trajectory("s-1", "task_completion", [0.2, 0.2, 0.2])
    runner, _ = _runner(tmp_path, cli, enable_tier3=True)

    report = await runner.run(CTX)

    tier3 = report.scores_for_tier(3)
    assert {str(s.metric) for s in tier3} == {
        "plan_adherence",
        "plan_quality",
        "step_efficiency",
    }
    assert {s.trace_id for s in tier3} == {"s-1-tr3"}


async def test_tier3_is_skipped_when_tier2_is_clean(tmp_path: Path) -> None:
    """The gate opening is not enough — Tier 3 only ever enriches a *confirmed*
    Tier-2 breach."""

    cli = FakeCliClient()  # every step metric healthy
    cli.script_trajectory("s-1", "task_completion", [0.2, 0.2, 0.2])
    runner, _ = _runner(tmp_path, cli, enable_tier3=True)

    report = await runner.run(CTX)

    assert report.gate_breached
    assert not any(s.breached for s in report.scores_for_tier(2))
    assert report.scores_for_tier(3) == ()


async def test_a_trace_is_scored_once_across_turns(tmp_path: Path) -> None:
    cli = FakeCliClient()
    cli.script_trajectory("s-1", "task_completion", [0.4, 0.6])
    runner, _ = _runner(tmp_path, cli)

    first = await runner.run(CTX)
    second = await runner.run(TurnContext(session_id="s-1", turn_index=2, end_state={}))

    assert len(first.scores_for_tier(1)) == 4  # 2 traces x 2 Tier-1 metrics
    assert second.scores_for_tier(1) == ()  # nothing new to score


async def test_no_traces_yet_yields_an_empty_report(tmp_path: Path) -> None:
    cli = FakeCliClient(auto_traces=False)
    cli.session_traces["s-1"] = []
    runner, _ = _runner(tmp_path, cli)

    report = await runner.run(CTX)

    assert report.scores == ()
    assert not report.any_alert


async def test_verifier_contributes_a_synthetic_outcome_score(tmp_path: Path) -> None:
    cli = FakeCliClient()
    runner, _ = _runner(tmp_path, cli, verifier=lambda _sid, _state: 0.4)

    report = await runner.run(CTX)

    (outcome,) = [s for s in report.scores if s.metric is Metric.OUTCOME]
    assert outcome.value == 0.4
    assert outcome.breached  # below the 0.9 default outcome threshold


async def test_verifier_may_be_async_and_may_return_a_bool(tmp_path: Path) -> None:
    async def verify(_sid: str, _state: object) -> bool:
        return True

    runner, _ = _runner(tmp_path, FakeCliClient(), verifier=verify)
    report = await runner.run(CTX)

    (outcome,) = [s for s in report.scores if s.metric is Metric.OUTCOME]
    assert outcome.value == 1.0
    assert not outcome.breached


async def test_a_broken_verifier_never_breaks_the_turn(tmp_path: Path) -> None:
    def explode(_sid: str, _state: object) -> float:
        raise RuntimeError("verifier is broken")

    cli = FakeCliClient()
    cli.script_trajectory("s-1", "task_completion", [0.9])
    runner, _ = _runner(tmp_path, cli, verifier=explode)

    report = await runner.run(CTX)

    assert not [s for s in report.scores if s.metric is Metric.OUTCOME]
    assert report.scores_for_tier(1)  # the rest of the turn still evaluated
