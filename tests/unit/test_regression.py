"""Unit tests for regression runs over the replayable eval-set."""

from __future__ import annotations

import logging

import pytest

from pandaprobe_harness import HarnessConfig, Journal, RulesStore
from pandaprobe_harness.evaluation.evaluator import MetricEvaluator
from pandaprobe_harness.validation.regression import run_regression
from pandaprobe_harness.workspace.evalset import EvalCase, EvalSet
from tests.fakes.fake_cli_client import FakeCliClient


@pytest.fixture
def evaluator(config: HarnessConfig, fake_cli: FakeCliClient) -> MetricEvaluator:
    return MetricEvaluator(fake_cli, config)


def _capture(
    evalset: EvalSet,
    session: str,
    *,
    kind: str = "failure",
    baseline: dict[str, float] | None = None,
    replayable: bool = True,
) -> EvalCase:
    case = evalset.capture(
        session_id=session,
        kind="win" if kind == "win" else "failure",
        signature=("breach:agent_reliability",),
        baseline_scores=baseline or {"agent_reliability": 0.3, "agent_consistency": 0.4},
        replay_input={"task": "charge"} if replayable else None,
    )
    assert case is not None
    return case


async def test_improved_case(
    config: HarnessConfig,
    rules: RulesStore,
    evalset: EvalSet,
    evaluator: MetricEvaluator,
    fake_cli: FakeCliClient,
    journal: Journal,
) -> None:
    case = _capture(evalset, "s-fail")
    fake_cli.set_session_scores(
        "s-replay-1", agent_reliability=0.92, agent_consistency=0.88
    )
    contexts: list[str] = []

    async def replay(case_arg: EvalCase, context: str) -> str:
        contexts.append(context)
        assert case_arg.id == case.id
        return "s-replay-1"

    report = await run_regression(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=evaluator,
        journal=journal,
        replay=replay,
    )

    assert report.total_cases == 1 and report.replayed == 1
    assert report.improved == 1 and report.clean
    (result,) = report.results
    assert result.status == "improved"
    assert result.replay_session_id == "s-replay-1"
    assert result.deltas is not None
    assert result.deltas["agent_reliability"] == pytest.approx(0.62)
    assert contexts and "Harness Rules" in contexts[0]

    (event,) = journal.recent(types=("regression",))
    assert event["improved"] == 1 and event["clean"] is True


async def test_regressed_win_case_flags_report(
    config: HarnessConfig,
    rules: RulesStore,
    evalset: EvalSet,
    evaluator: MetricEvaluator,
    fake_cli: FakeCliClient,
) -> None:
    _capture(evalset, "s-win", kind="win", baseline={"agent_reliability": 0.9})
    fake_cli.set_session_scores("s-replay-w", agent_reliability=0.3)

    async def replay(case: EvalCase, context: str) -> str:
        return "s-replay-w"

    report = await run_regression(
        config=config, rules=rules, evalset=evalset, evaluator=evaluator, replay=replay
    )
    assert report.regressed == 1 and not report.clean
    assert report.results[0].status == "regressed"


async def test_unchanged_within_margins(
    config: HarnessConfig,
    rules: RulesStore,
    evalset: EvalSet,
    evaluator: MetricEvaluator,
) -> None:
    # Fake defaults score 0.9/0.9 → deltas of 0.0 against an equal baseline.
    _capture(
        evalset,
        "s-same",
        baseline={"agent_reliability": 0.9, "agent_consistency": 0.9},
    )

    async def replay(case: EvalCase, context: str) -> str:
        return "s-replay-same"

    report = await run_regression(
        config=config, rules=rules, evalset=evalset, evaluator=evaluator, replay=replay
    )
    assert report.unchanged == 1 and report.clean


async def test_wins_replay_before_failures_and_sample_caps(
    config: HarnessConfig,
    rules: RulesStore,
    evalset: EvalSet,
    evaluator: MetricEvaluator,
) -> None:
    _capture(evalset, "s-fail")
    win = _capture(evalset, "s-win", kind="win", baseline={"agent_reliability": 0.9})

    order: list[str] = []

    async def replay(case: EvalCase, context: str) -> str:
        order.append(case.id)
        return f"s-replay-{case.id}"

    report = await run_regression(
        config=config, rules=rules, evalset=evalset, evaluator=evaluator, replay=replay
    )
    assert order[0] == win.id and report.total_cases == 2

    order.clear()
    sampled = await run_regression(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=evaluator,
        replay=replay,
        sample=1,
    )
    assert sampled.total_cases == 1 and order == [win.id]


async def test_without_replay_degrades_to_skips(
    config: HarnessConfig,
    rules: RulesStore,
    evalset: EvalSet,
    evaluator: MetricEvaluator,
    journal: Journal,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _capture(evalset, "s-fail")
    with caplog.at_level(logging.WARNING, logger="pandaprobe_harness.validation"):
        report = await run_regression(
            config=config, rules=rules, evalset=evalset, evaluator=evaluator, journal=journal
        )

    assert report.skipped == 1 and report.replayed == 0
    assert report.clean  # skips are honest, not failures
    warnings = [r for r in caplog.records if "no replay function wired" in r.message]
    assert len(warnings) == 1
    (event,) = journal.recent(types=("regression",))
    assert event["skipped"] == 1


async def test_unreplayable_and_broken_cases_skip_with_reasons(
    config: HarnessConfig,
    rules: RulesStore,
    evalset: EvalSet,
    evaluator: MetricEvaluator,
) -> None:
    no_input = evalset.capture(session_id="s-1", signature=("x",))
    assert no_input is not None
    broken = _capture(evalset, "s-2")

    async def replay(case: EvalCase, context: str) -> str:
        raise RuntimeError("agent exploded")

    report = await run_regression(
        config=config, rules=rules, evalset=evalset, evaluator=evaluator, replay=replay
    )
    assert report.skipped == 2
    reasons = {r.case_id: r.reason for r in report.results}
    assert "no replay_input" in reasons[no_input.id]
    assert "agent exploded" in reasons[broken.id]


def test_render_text_summarizes(config: HarnessConfig) -> None:
    from pandaprobe_harness.validation.regression import CaseResult, RegressionReport

    report = RegressionReport(
        started_at="t0",
        finished_at="t1",
        total_cases=2,
        replayed=1,
        improved=1,
        unchanged=0,
        regressed=0,
        skipped=1,
        results=(
            CaseResult(
                case_id="c-1",
                kind="failure",
                status="improved",
                replay_session_id="s-r",
                deltas={"agent_reliability": 0.62},
            ),
            CaseResult(case_id="c-2", kind="win", status="skipped", reason="no replay_input"),
        ),
    )
    text = report.render_text()
    assert "improved 1" in text
    assert "agent_reliability +0.62" in text
    assert "no replay_input" in text
    assert text.endswith("CLEAN")
