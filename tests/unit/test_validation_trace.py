"""Validation graded on the trace trigger's own signal (and on a verifier)."""

from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import (
    HarnessConfig,
    Journal,
    RulesStore,
    ValidationEngine,
)
from pandaprobe_harness.evaluation.evaluator import MetricEvaluator
from pandaprobe_harness.validation.validator import ReplayValidator
from pandaprobe_harness.workspace.evalset import EvalCase, EvalSet
from tests.fakes.fake_cli_client import FakeCliClient


def _config(tmp_path: Path, **kw: object) -> HarnessConfig:
    return HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_attempts=1,
        eval_retry_backoff_s=0.0,
        **kw,  # type: ignore[arg-type]
    )


def _case(
    evalset: EvalSet, *, signature: str = "breach:tool_correctness", **baseline: float
) -> EvalCase:
    case = evalset.capture(
        session_id="s-failed",
        kind="failure",
        signature=(signature,),
        baseline_scores=dict(baseline),
        replay_input={"task_id": "t-1"},
    )
    assert case is not None
    return case


def _validator(
    cfg: HarnessConfig, cli: FakeCliClient, replay_session: str, **kw: object
) -> tuple[ReplayValidator, RulesStore, EvalSet]:
    journal = Journal(cfg)
    rules = RulesStore(cfg, journal=journal)
    evalset = EvalSet(cfg, journal=journal)
    evalset.provision()

    async def replay(case: EvalCase, context: str) -> str:
        del case, context
        return replay_session

    return (
        ReplayValidator(
            config=cfg,
            rules=rules,
            evalset=evalset,
            evaluator=MetricEvaluator(cli, cfg),
            replay=replay,
            verifier=kw.get("verifier"),  # type: ignore[arg-type]
        ),
        rules,
        evalset,
    )


async def test_replay_is_graded_on_the_trace_metric_that_triggered(
    tmp_path: Path,
) -> None:
    """v1 re-scored a replay with the session composites — the same degenerate
    signal the trigger used — so promotion was effectively random. The replay must
    be graded on the trace metric the breach was about."""

    cli = FakeCliClient()
    cli.script_trajectory("s-replay", "tool_correctness", [0.9])
    cfg = _config(tmp_path)
    validator, rules, evalset = _validator(cfg, cli, "s-replay")
    _case(evalset, tool_correctness=0.2)
    candidate = rules.add("check the tool first", "why", metric="tool_correctness")

    verdict = await validator.validate(candidate)

    assert verdict.outcome == "promote"
    assert "tool_correctness" in verdict.reason
    # It scored the replay's last trace, not the session.
    joined = " ".join(" ".join(call) for call in cli.batch_calls)
    assert "--target trace" in joined
    assert "--target session" not in joined


async def test_no_improvement_on_the_trace_metric_retires(tmp_path: Path) -> None:
    cli = FakeCliClient()
    cli.script_trajectory("s-replay", "tool_correctness", [0.21])  # within the margin
    cfg = _config(tmp_path)
    validator, rules, evalset = _validator(cfg, cli, "s-replay")
    _case(evalset, tool_correctness=0.2)
    candidate = rules.add("a rule that does not help", "why", metric="tool_correctness")

    verdict = await validator.validate(candidate)

    assert verdict.outcome == "retire"


async def test_session_mode_still_grades_on_the_composites(tmp_path: Path) -> None:
    """The ablation arm must keep working unchanged."""

    cli = FakeCliClient(metric_values={"agent_reliability": 0.9, "agent_consistency": 0.9})
    cfg = _config(tmp_path, trigger_mode="session")
    validator, rules, evalset = _validator(cfg, cli, "s-replay")
    _case(evalset, signature="breach:agent_reliability", agent_reliability=0.2)
    candidate = rules.add("a rule", "why", metric="agent_reliability")

    verdict = await validator.validate(candidate)

    assert verdict.outcome == "promote"
    joined = " ".join(" ".join(call) for call in cli.batch_calls)
    assert "--target session" in joined


async def test_a_wired_verifier_decides_promotion(tmp_path: Path) -> None:
    """A developer verifier knows what success actually means, so it outranks a
    judged proxy: the trace metric may improve, but the outcome is what counts."""

    cli = FakeCliClient()
    # The judged step metric improves a lot …
    cli.script_trajectory("s-replay", "tool_correctness", [0.95])
    cfg = _config(tmp_path)
    # … while the verifier says the task still is not actually done.
    validator, rules, evalset = _validator(
        cfg, cli, "s-replay", verifier=lambda _sid, _state: 0.1
    )
    _case(evalset, tool_correctness=0.2, outcome_correct=0.1)
    candidate = rules.add("looks helpful but is not", "why", metric="tool_correctness")

    verdict = await validator.validate(candidate)

    assert verdict.outcome == "retire"


async def test_the_verifier_sees_the_cases_replay_input(tmp_path: Path) -> None:
    """The replay only hands back a session id, so the case's replay_input stands
    in for the end state — that is what lets a task-keyed verifier grade a replay."""

    seen: list[tuple[str, object]] = []

    def verify(session_id: str, state: object) -> float:
        seen.append((session_id, state))
        return 1.0

    cli = FakeCliClient()
    cli.script_trajectory("s-replay", "tool_correctness", [0.9])
    cfg = _config(tmp_path)
    validator, rules, evalset = _validator(cfg, cli, "s-replay", verifier=verify)
    _case(evalset, tool_correctness=0.2, outcome_correct=0.0)
    candidate = rules.add("a promising rule", "why", metric="tool_correctness")

    verdict = await validator.validate(candidate)

    assert verdict.outcome == "promote"
    assert seen and seen[0][0] == "s-replay"
    assert seen[0][1] == {"task_id": "t-1"}


async def test_a_verifier_with_no_verdict_does_not_veto_promotion(
    tmp_path: Path,
) -> None:
    """The target metric is chosen from the deltas that actually arrived, not from
    whether a verifier happens to be configured. A grader that cannot score this
    task contributes nothing, so the judged metric still decides — otherwise wiring
    a verifier at all would block promotion for every task it cannot grade."""

    cli = FakeCliClient()
    cli.script_trajectory("s-replay", "tool_correctness", [0.9])
    cfg = _config(tmp_path)
    validator, rules, evalset = _validator(
        cfg, cli, "s-replay", verifier=lambda _sid, _state: None
    )
    _case(evalset, tool_correctness=0.2)
    candidate = rules.add("a genuinely helpful rule", "why", metric="tool_correctness")

    verdict = await validator.validate(candidate)

    assert verdict.outcome == "promote"
    assert "tool_correctness" in verdict.reason


async def test_engine_passes_the_verifier_through(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    journal = Journal(cfg)
    evalset = EvalSet(cfg, journal=journal)
    evalset.provision()

    async def replay(case: EvalCase, context: str) -> str:
        del case, context
        return "s-replay"

    engine = ValidationEngine(
        config=cfg,
        rules=RulesStore(cfg, journal=journal),
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), cfg),
        journal=journal,
        replay=replay,
        verifier=lambda _sid, _state: True,
    )

    assert engine.has_replay
