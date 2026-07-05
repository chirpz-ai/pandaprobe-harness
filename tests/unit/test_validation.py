"""Unit tests for candidate-rule validation (replay + forward trial + engine)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pandaprobe_harness import HarnessConfig, Journal, RulesStore
from pandaprobe_harness.evaluation.evaluator import MetricEvaluator
from pandaprobe_harness.validation.validator import (
    ForwardTrialValidator,
    ReplayValidator,
    ValidationEngine,
)
from pandaprobe_harness.workspace.evalset import EvalCase, EvalSet
from pandaprobe_harness.workspace.rules import Rule, TrialState
from tests.fakes.fake_cli_client import FakeCliClient


def _rule(trial: TrialState | None, *, metric: str | None = "agent_reliability") -> Rule:
    return Rule(
        id="r-under-test",
        created_at="2026-07-01T00:00:00+00:00",
        rule="verify before retrying",
        rationale="repeated failures",
        metric=metric,
        status="candidate",
        tags=("breach:agent_reliability",),
        trial=trial,
    )


# -- forward trial -----------------------------------------------------------------


async def test_forward_trial_pending_until_min_sessions(config: HarnessConfig) -> None:
    validator = ForwardTrialValidator(config=config)

    verdict = await validator.validate(_rule(None))
    assert verdict.outcome == "pending"

    partial = TrialState(observed_sessions=("s-1", "s-2"))
    verdict = await validator.validate(_rule(partial))
    assert verdict.outcome == "pending"
    assert "2/5" in verdict.reason


async def test_forward_trial_promotes_on_zero_breaches(config: HarnessConfig) -> None:
    trial = TrialState(
        baseline_breached_sessions=4,
        baseline_sessions=4,
        observed_sessions=("s-1", "s-2", "s-3", "s-4", "s-5"),
        breached_sessions=(),
    )
    verdict = await ForwardTrialValidator(config=config).validate(_rule(trial))
    assert verdict.outcome == "promote"
    assert verdict.validator == "forward_trial"
    assert verdict.details["trial_rate"] == 0.0


async def test_forward_trial_promotes_on_rate_drop_past_margin(
    config: HarnessConfig,
) -> None:
    trial = TrialState(
        baseline_breached_sessions=8,
        baseline_sessions=10,  # baseline 0.8
        observed_sessions=("s-1", "s-2", "s-3", "s-4", "s-5"),
        breached_sessions=("s-2",),  # trial rate 0.2
    )
    verdict = await ForwardTrialValidator(config=config).validate(_rule(trial))
    assert verdict.outcome == "promote"


async def test_forward_trial_retires_without_improvement(config: HarnessConfig) -> None:
    trial = TrialState(
        baseline_breached_sessions=5,
        baseline_sessions=10,  # baseline 0.5
        observed_sessions=("s-1", "s-2", "s-3", "s-4", "s-5"),
        breached_sessions=("s-1", "s-2", "s-3"),  # trial rate 0.6
    )
    verdict = await ForwardTrialValidator(config=config).validate(_rule(trial))
    assert verdict.outcome == "retire"
    assert "did not improve" in verdict.reason


# -- replay ------------------------------------------------------------------------


def _stores(
    tmp_path: Path, **overrides: object
) -> tuple[HarnessConfig, Journal, RulesStore, EvalSet]:
    config = HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        **overrides,  # type: ignore[arg-type]
    )
    journal = Journal(config)
    rules = RulesStore(config, journal=journal)
    evalset = EvalSet(config, journal=journal)
    return config, journal, rules, evalset


def _seed_failure_case(evalset: EvalSet, *, replayable: bool = True) -> EvalCase:
    case = evalset.capture(
        session_id="s-original",
        signature=("breach:agent_reliability",),
        baseline_scores={"agent_reliability": 0.3, "agent_consistency": 0.4},
        replay_input={"task": "charge"} if replayable else None,
    )
    assert case is not None
    return case


async def test_replay_promotes_when_metric_improves(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add(
        "verify before retrying", "x", metric="agent_reliability",
        tags=["breach:agent_reliability"],
    )
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.set_session_scores("s-replayed", agent_reliability=0.92, agent_consistency=0.88)
    contexts: list[str] = []

    async def replay(case: EvalCase, context: str) -> str:
        contexts.append(context)
        return "s-replayed"

    validator = ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        replay=replay,
    )
    verdict = await validator.validate(candidate)

    assert verdict.outcome == "promote"
    assert verdict.validator == "replay"
    # The candidate was in force during the replay: the provisional SECTION
    # rendered (the template merely mentions the phrase in prose, so the
    # "###" heading is the meaningful check).
    from pandaprobe_harness.workspace.rules import PROVISIONAL_HEADING

    assert PROVISIONAL_HEADING in contexts[0]
    assert "verify before retrying" in contexts[0]


async def test_replay_retires_without_improvement(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add("verify before retrying", "x", metric="agent_reliability")
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    # Replayed session scores exactly the baseline: no improvement, no regression.
    fake.set_session_scores("s-replayed", agent_reliability=0.3, agent_consistency=0.4)

    async def replay(case: EvalCase, context: str) -> str:
        return "s-replayed"

    validator = ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        replay=replay,
    )
    verdict = await validator.validate(candidate)
    assert verdict.outcome == "retire"
    assert "no improvement" in verdict.reason


async def test_replay_retires_on_win_regression(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add("verify before retrying", "x", metric="agent_reliability")
    failure = _seed_failure_case(evalset)
    win = evalset.capture(
        session_id="s-win",
        kind="win",
        signature=("healthy",),
        baseline_scores={"agent_reliability": 0.9},
        replay_input={"task": "browse"},
    )
    assert win is not None

    fake = FakeCliClient()
    fake.set_session_scores(
        f"s-replay-{failure.id}", agent_reliability=0.92, agent_consistency=0.88
    )
    fake.set_session_scores(f"s-replay-{win.id}", agent_reliability=0.2)

    async def replay(case: EvalCase, context: str) -> str:
        return f"s-replay-{case.id}"

    validator = ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        replay=replay,
    )
    verdict = await validator.validate(candidate)
    assert verdict.outcome == "retire"
    assert "regression" in verdict.reason
    assert win.id in verdict.reason


async def test_replay_pending_without_matching_replayable_case(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add("verify before retrying", "x", metric="agent_reliability")
    _seed_failure_case(evalset, replayable=False)

    async def replay(case: EvalCase, context: str) -> str:  # pragma: no cover
        raise AssertionError("must not be called")

    validator = ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        replay=replay,
    )
    verdict = await validator.validate(candidate)
    assert verdict.outcome == "pending"
    assert "no replayable eval case" in verdict.reason


async def test_replay_inconclusive_when_replay_raises(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add("verify before retrying", "x", metric="agent_reliability")
    _seed_failure_case(evalset)

    async def replay(case: EvalCase, context: str) -> str:
        raise RuntimeError("agent exploded")

    validator = ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        replay=replay,
    )
    verdict = await validator.validate(candidate)
    assert verdict.outcome == "pending"
    assert "inconclusive" in verdict.reason


async def test_replay_pending_when_only_win_cases_conclude(tmp_path: Path) -> None:
    """A broken failure-case replay must not read as 'no improvement': with
    zero conclusive failure cases there is no evidence either way."""

    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add("verify before retrying", "x", metric="agent_reliability")
    failure = _seed_failure_case(evalset)
    win = evalset.capture(
        session_id="s-win",
        kind="win",
        signature=("healthy",),
        baseline_scores={"agent_reliability": 0.9},
        replay_input={"task": "browse"},
    )
    assert win is not None
    fake = FakeCliClient()
    fake.set_session_scores(f"s-replay-{win.id}", agent_reliability=0.9)  # unchanged

    async def replay(case: EvalCase, context: str) -> str:
        if case.id == failure.id:
            raise RuntimeError("failure replay broke")
        return f"s-replay-{case.id}"

    validator = ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        replay=replay,
    )
    verdict = await validator.validate(candidate)
    assert verdict.outcome == "pending"
    assert "inconclusive" in verdict.reason


async def test_replay_case_without_shared_metrics_is_inconclusive(tmp_path: Path) -> None:
    """An empty-baseline case (capturable via the public API) is evidence of
    nothing — it must not retire the candidate as 'no improvement'."""

    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add("verify before retrying", "x", metric="agent_reliability")
    case = evalset.capture(
        session_id="s-original",
        signature=("breach:agent_reliability",),
        baseline_scores={},  # nothing to compare against
        replay_input={"task": "charge"},
    )
    assert case is not None

    async def replay(case_arg: EvalCase, context: str) -> str:
        return "s-replayed"

    validator = ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        replay=replay,
    )
    verdict = await validator.validate(candidate)
    assert verdict.outcome == "pending"


async def test_hung_replay_times_out_to_inconclusive(tmp_path: Path) -> None:
    """A never-resolving developer replay must degrade (bounded by
    replay_timeout_s), not wedge validation forever."""

    import asyncio

    config, journal, rules, evalset = _stores(tmp_path, replay_timeout_s=0.05)
    candidate = rules.add("verify before retrying", "x", metric="agent_reliability")
    _seed_failure_case(evalset)

    async def replay(case: EvalCase, context: str) -> str:
        await asyncio.sleep(60)
        return "never"  # pragma: no cover

    validator = ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        replay=replay,
    )
    verdict = await validator.validate(candidate)
    assert verdict.outcome == "pending"
    assert "inconclusive" in verdict.reason


# -- engine ------------------------------------------------------------------------


async def test_engine_observe_report_tracks_trials(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=3)
    candidate = rules.add("verify before retrying", "x", metric="agent_reliability")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )

    engine.observe_report("s-1", set())
    engine.observe_report("s-2", {"breach:agent_reliability"})
    engine.observe_report("s-2", {"breach:agent_reliability"})  # same session, once
    engine.observe_report("s-3", {"trend:agent_consistency"})  # other metric family

    (reloaded,) = rules.candidates()
    trial = reloaded.trial
    assert trial is not None
    assert trial.observed_sessions == ("s-1", "s-2", "s-3")
    assert trial.breached_sessions == ("s-2",)

    # The window is full (3 sessions); an unseen session no longer enrolls,
    # but a known session that breaches later still flips to breached.
    engine.observe_report("s-4", set())
    engine.observe_report("s-1", {"relative:agent_reliability"})
    (reloaded,) = rules.candidates()
    assert reloaded.trial is not None
    assert reloaded.trial.observed_sessions == ("s-1", "s-2", "s-3")
    assert set(reloaded.trial.breached_sessions) == {"s-1", "s-2"}
    assert candidate.id == reloaded.id


async def test_engine_forward_trial_promotes_and_logs_fallback_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=2)
    rule = rules.add("verify before retrying", "x", metric="agent_reliability")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )
    assert not engine.has_replay

    with caplog.at_level(logging.WARNING, logger="pandaprobe_harness.validation"):
        verdicts = await engine.evaluate_candidates()  # trial not yet satisfied
        assert [v.outcome for v in verdicts] == ["pending"]

        engine.observe_report("s-1", set())
        engine.observe_report("s-2", set())
        verdicts = await engine.evaluate_candidates()
        assert [v.outcome for v in verdicts] == ["promote"]

        await engine.evaluate_candidates()  # nothing left to validate

    fallback_logs = [r for r in caplog.records if "no replay function wired" in r.message]
    assert len(fallback_logs) == 1
    (fallback_event,) = journal.recent(types=("validation",))
    assert fallback_event["mode"] == "forward_trial"

    (promoted,) = rules.active()
    assert promoted.id == rule.id
    assert promoted.trial is not None
    assert promoted.trial.verdict.startswith("promoted:")
    (promote_event,) = journal.recent(types=("rule_promote",))
    assert promote_event["validator"] == "forward_trial"


async def test_engine_retires_failed_trial(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=2)
    rule = rules.add("verify before retrying", "x", metric="agent_reliability")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )
    # Both trial sessions still breach; the baseline was empty (rate 1.0), so
    # a full-rate trial (1.0) shows no improvement.
    engine.observe_report("s-1", {"breach:agent_reliability"})
    engine.observe_report("s-2", {"breach:agent_reliability"})

    (verdict,) = await engine.evaluate_candidates()
    assert verdict.outcome == "retire"
    assert rules.live() == []
    (retire_event,) = journal.recent(types=("rule_retire",))
    assert retire_event["id"] == rule.id
    assert "did not improve" in retire_event["reason"]
    # The verdict is stamped onto the retired record's trial bookkeeping,
    # so harness_rule_status can explain why afterwards.
    (retired,) = rules.all()
    assert retired.trial is not None
    assert retired.trial.verdict.startswith("retired:")


async def test_engine_prefers_replay_over_forward_trial(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path)
    rules.add(
        "verify before retrying", "x", metric="agent_reliability",
        tags=["breach:agent_reliability"],
    )
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.set_session_scores("s-replayed", agent_reliability=0.92, agent_consistency=0.88)

    async def replay(case: EvalCase, context: str) -> str:
        return "s-replayed"

    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        journal=journal,
    replay=replay,
    )
    assert engine.has_replay

    (verdict,) = await engine.evaluate_candidates()
    assert verdict.outcome == "promote" and verdict.validator == "replay"
    (promoted,) = rules.active()
    assert promoted.status == "active"


async def test_engine_never_raises(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path)
    rules.add("verify before retrying", "x", metric="agent_reliability")

    class _Boom:
        async def validate(self, rule: object) -> object:
            raise RuntimeError("boom")

    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )
    engine._forward = _Boom()  # type: ignore[assignment]

    verdicts = await engine.evaluate_candidates()
    assert verdicts == []  # the failure was contained, not raised
