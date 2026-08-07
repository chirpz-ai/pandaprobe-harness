"""Unit tests for candidate-rule validation (replay + forward trial + engine)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

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


def _rule(trial: TrialState | None, *, metric: str | None = "task_completion") -> Rule:
    return Rule(
        id="r-under-test",
        created_at="2026-07-01T00:00:00+00:00",
        rule="verify before retrying",
        rationale="repeated failures",
        metric=metric,
        status="candidate",
        tags=("stall:task_completion",),
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


async def _read_candidate(context: Any, rule: Rule) -> None:
    """Do what a real replayed agent does: pull the scope it was told about.

    A replay that never reads the candidate cannot produce a conclusive verdict
    (that is the point of the read ledger), so any test asserting promote/retire
    has to actually exercise it.
    """

    await context.task_tools.call("harness_rules_read", {"scope": rule.scope})


def _seed_failure_case(evalset: EvalSet, *, replayable: bool = True) -> EvalCase:
    case = evalset.capture(
        session_id="s-original",
        signature=("stall:task_completion",),
        baseline_scores={"task_completion": 0.3, "coherence": 0.4},
        replay_input={"task": "charge"} if replayable else None,
    )
    assert case is not None
    return case


async def test_replay_promotes_when_metric_improves(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.script_trace("s-replayed", task_completion=0.92, coherence=0.88)
    contexts: list[str] = []
    replay_rule_text: list[str] = []

    async def replay(case: EvalCase, context: Any) -> str:
        contexts.append(context)
        index = await context.task_tools.call("harness_rules_list", {})
        assert index["scopes"][0]["scope"] == candidate.scope
        read = await context.task_tools.call(
            "harness_rules_read", {"scope": candidate.scope}
        )
        replay_rule_text.append(read["content"])
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
    # The candidate was discoverable on demand during replay, not injected.
    from pandaprobe_harness.workspace.rules import PROVISIONAL_HEADING

    assert PROVISIONAL_HEADING not in contexts[0]
    assert PROVISIONAL_HEADING in replay_rule_text[0]
    assert "verify before retrying" not in contexts[0]
    assert "verify before retrying" in replay_rule_text[0]


async def test_replay_retires_without_improvement(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    # Replayed session scores exactly the baseline: no improvement, no regression.
    fake.script_trace("s-replayed", task_completion=0.3, coherence=0.4)

    async def replay(case: EvalCase, context: Any) -> str:
        await _read_candidate(context, candidate)
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
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
    failure = _seed_failure_case(evalset)
    win = evalset.capture(
        session_id="s-win",
        kind="win",
        signature=("healthy",),
        baseline_scores={"task_completion": 0.9},
        replay_input={"task": "browse"},
    )
    assert win is not None

    fake = FakeCliClient()
    fake.script_trace(
        f"s-replay-{failure.id}", task_completion=0.92, coherence=0.88
    )
    fake.script_trace(f"s-replay-{win.id}", task_completion=0.2)

    async def replay(case: EvalCase, context: Any) -> str:
        await _read_candidate(context, candidate)
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
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
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
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
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
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
    failure = _seed_failure_case(evalset)
    win = evalset.capture(
        session_id="s-win",
        kind="win",
        signature=("healthy",),
        baseline_scores={"task_completion": 0.9},
        replay_input={"task": "browse"},
    )
    assert win is not None
    fake = FakeCliClient()
    fake.script_trace(f"s-replay-{win.id}", task_completion=0.9)  # unchanged

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
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
    case = evalset.capture(
        session_id="s-original",
        signature=("stall:task_completion",),
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
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
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
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )

    engine.observe_report("s-1", set())
    engine.observe_report("s-2", {"stall:task_completion"})
    engine.observe_report("s-2", {"stall:task_completion"})  # same session, once
    engine.observe_report("s-3", {"stall:coherence"})  # other metric family

    (reloaded,) = rules.candidates()
    trial = reloaded.trial
    assert trial is not None
    assert trial.observed_sessions == ("s-1", "s-2", "s-3")
    assert trial.breached_sessions == ("s-2",)

    # The window is full (3 sessions); an unseen session no longer enrolls,
    # but a known session that breaches later still flips to breached.
    engine.observe_report("s-4", set())
    engine.observe_report("s-1", {"regression:task_completion"})
    (reloaded,) = rules.candidates()
    assert reloaded.trial is not None
    assert reloaded.trial.observed_sessions == ("s-1", "s-2", "s-3")
    assert set(reloaded.trial.breached_sessions) == {"s-1", "s-2"}
    assert candidate.id == reloaded.id


async def test_engine_forward_trial_promotes_and_logs_fallback_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=2)
    rule = rules.add("verify before retrying", "x", metric="task_completion")
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
    rule = rules.add("verify before retrying", "x", metric="task_completion")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )
    # Both trial sessions still breach; the baseline was empty (rate 1.0), so
    # a full-rate trial (1.0) shows no improvement.
    engine.observe_report("s-1", {"stall:task_completion"})
    engine.observe_report("s-2", {"stall:task_completion"})

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
    candidate = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.script_trace("s-replayed", task_completion=0.92, coherence=0.88)

    async def replay(case: EvalCase, context: Any) -> str:
        await _read_candidate(context, candidate)
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
    rules.add("verify before retrying", "x", metric="task_completion")

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


async def test_evidence_observed_during_a_replay_round_survives_the_verdict(
    tmp_path: Path,
) -> None:
    """_apply must not clobber trial evidence appended while the (slow) replay
    was running: the verdict is stamped onto the FRESH trial, not the round's
    snapshot."""

    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.script_trace("s-replayed", task_completion=0.92, coherence=0.88)
    engine_ref: list[ValidationEngine] = []

    async def replay(case: EvalCase, context: Any) -> str:
        await _read_candidate(context, candidate)
        # A concurrent session's report lands mid-round.
        engine_ref[0].observe_report("s-during-replay", set())
        return "s-replayed"

    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        journal=journal,
        replay=replay,
    )
    engine_ref.append(engine)

    (verdict,) = await engine.evaluate_candidates()
    assert verdict.outcome == "promote"
    (promoted,) = rules.active()
    assert promoted.trial is not None
    assert "s-during-replay" in promoted.trial.observed_sessions  # not clobbered
    assert promoted.trial.verdict.startswith("promoted:")


async def test_candidate_retired_mid_round_is_skipped(tmp_path: Path) -> None:
    """Every candidate is judged on its FRESH state: one promoted/retired while
    an earlier candidate's validation ran is skipped, not validated against a
    stale snapshot."""

    config, journal, rules, evalset = _stores(tmp_path)
    first = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    second = rules.add("an unrelated candidate", "y", metric="coherence")
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.script_trace("s-replayed", task_completion=0.92, coherence=0.88)

    async def replay(case: EvalCase, context: Any) -> str:
        await _read_candidate(context, first)
        # While the first candidate replays, the agent retires the second.
        rules.retire(second.id, reason="agent decided against it")
        return "s-replayed"

    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        journal=journal,
        replay=replay,
    )

    verdicts = await engine.evaluate_candidates()
    assert [v.rule_id for v in verdicts] == [first.id]  # the second was skipped
    assert [r.id for r in rules.active()] == [first.id]


# -- every candidate reaches a verdict ----------------------------------------------


async def test_a_beneficial_candidate_is_promoted_by_the_forward_trial(
    tmp_path: Path,
) -> None:
    """The cheap path must be able to promote on its own.

    In the inspected run three candidates had already earned this verdict and
    never received it, because the code only reached the forward validator after
    replay returned non-pending or exhausted its attempts.
    """

    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=2)
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )
    # Two clean sessions: the candidate's metric family never fired.
    engine.observe_report("s1", set())
    engine.observe_report("s2", set())

    (verdict,) = await engine.evaluate_candidates()

    assert verdict.outcome == "promote" and verdict.validator == "forward_trial"
    assert [rule.id for rule in rules.active()] == [candidate.id]


async def test_a_regressing_candidate_is_retired_by_the_forward_trial(
    tmp_path: Path,
) -> None:
    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=2)
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )
    # Both sessions still breach the metric the rule claims to fix.
    engine.observe_report("s1", {"breach:task_completion"})
    engine.observe_report("s2", {"breach:task_completion"})

    (verdict,) = await engine.evaluate_candidates()

    assert verdict.outcome == "retire"
    assert rules.active() == []
    assert [rule.status for rule in rules.all() if rule.id == candidate.id] == ["retired"]


async def test_a_replay_that_never_read_the_candidate_is_not_conclusive(
    tmp_path: Path,
) -> None:
    """Scores moved, but this rule was never seen — so it did not move them."""

    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    # Scores that would otherwise promote outright.
    fake.script_trace("s-replayed", task_completion=0.92, coherence=0.88)

    async def replay(case: EvalCase, context: Any) -> str:
        del case, context  # deliberately never reads the rule
        return "s-replayed"

    verdict = await ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        replay=replay,
    ).validate(candidate)

    assert verdict.outcome == "pending"
    assert verdict.pending_reason == "candidate_not_exercised"
    assert verdict.details["unexercised"] == 1
    assert verdict.details["cases"][0]["candidate_surfaced"] is False


async def test_a_replay_only_sees_the_candidate_under_test(tmp_path: Path) -> None:
    """Unrelated provisionals cannot steer a replay and be charged to this rule."""

    config, journal, rules, evalset = _stores(tmp_path)
    under_test = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    unrelated = rules.add("some other provisional idea", "y", metric="coherence")
    active = rules.add("an already validated rule", "z")
    rules.promote(active.id, reason="test setup")
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.script_trace("s-replayed", task_completion=0.92, coherence=0.88)
    seen: list[str] = []

    async def replay(case: EvalCase, context: Any) -> str:
        del case
        for scope in ("global", under_test.scope, unrelated.scope):
            result = await context.task_tools.call("harness_rules_read", {"scope": scope})
            seen.append(result["content"])
        found = await context.task_tools.call("harness_rules_search", {"query": "rule"})
        seen.extend(rule["id"] for rule in found["rules"])
        return "s-replayed"

    verdict = await ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        replay=replay,
    ).validate(under_test)

    blob = "\n".join(seen)
    assert "verify before retrying" in blob  # the candidate under test
    assert "an already validated rule" in blob  # actives stay visible
    assert "some other provisional idea" not in blob  # the other candidate does not
    assert unrelated.id not in seen
    assert verdict.outcome == "promote"


async def test_an_unrelated_metric_drop_does_not_retire_the_candidate(
    tmp_path: Path,
) -> None:
    """A candidate answers for its own metric, not for everything scored beside it.

    All four retirements in the inspected run fired on a metric the candidate never
    claimed — one on a coherence drop of 0.09, against a natural spread of ~0.15.
    """

    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    # Target metric improves past the margin; an unrelated judged metric dips.
    fake.script_trace("s-replayed", task_completion=0.92, coherence=0.25)

    async def replay(case: EvalCase, context: Any) -> str:
        await _read_candidate(context, candidate)
        return "s-replayed"

    verdict = await ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        replay=replay,
    ).validate(candidate)

    assert verdict.outcome == "promote"
    assert "coherence" not in verdict.details["retire_on"]


async def test_a_target_metric_drop_still_retires_the_candidate(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.script_trace("s-replayed", task_completion=0.05, coherence=0.4)

    async def replay(case: EvalCase, context: Any) -> str:
        await _read_candidate(context, candidate)
        return "s-replayed"

    verdict = await ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        replay=replay,
    ).validate(candidate)

    assert verdict.outcome == "retire"
    assert "task_completion" in verdict.reason


async def test_every_pending_replay_round_counts_toward_the_fallback(
    tmp_path: Path,
) -> None:
    """`replay_attempts` must advance on ANY pending replay, not just some.

    Counting only rounds whose prose said "inconclusive" left a candidate with no
    matching replayable case retrying replay forever, so the forward trial was
    unreachable and the candidate never got a verdict at all.
    """

    # A trial window of 3 with one observed session keeps the forward trial pending
    # too, so the only thing that can advance across rounds is the replay counter.
    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=3)
    rules.add("verify before retrying", "x", metric="task_completion")
    # No eval case exists, so replay is pending with "no matching replayable case".
    replays = 0

    async def replay(case: EvalCase, context: Any) -> str:
        nonlocal replays
        replays += 1
        return "s-never"

    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
        replay=replay,
    )
    engine.observe_report("s1", set())

    attempts = []
    for _ in range(4):
        await engine.evaluate_candidates()
        candidates = rules.candidates()
        if not candidates:
            break
        attempts.append(candidates[0].trial.replay_attempts)  # type: ignore[union-attr]

    assert replays == 0  # nothing was replayable to begin with
    # The counter advances every round, so _MAX_REPLAY_ATTEMPTS is reachable and
    # the candidate stops burning rounds on a replay that can never happen.
    assert attempts == [1, 2, 3, 3]


async def test_the_round_budget_still_yields_a_verdict_per_candidate(
    tmp_path: Path,
) -> None:
    """Past the replay budget, candidates get the cheap verdict — not none.

    Sequential replays are slower than candidate creation, so a round that only
    ever replays leaves the newest candidates permanently undecided.
    """

    config, journal, rules, evalset = _stores(
        tmp_path, rule_trial_min_sessions=1, validation_round_budget_s=0.001
    )
    for index in range(3):
        rules.add(f"rule number {index}", "x", metric="task_completion")
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.script_trace("s-replayed", task_completion=0.92, coherence=0.88)

    async def replay(case: EvalCase, context: Any) -> str:
        del case, context
        await asyncio.sleep(0.01)  # push the round past its budget
        return "s-replayed"

    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        journal=journal,
        replay=replay,
    )
    engine.observe_report("s1", set())

    verdicts = await engine.evaluate_candidates()

    assert len(verdicts) == 3  # every candidate was judged
    assert all(verdict.outcome != "pending" for verdict in verdicts)


async def test_the_round_rotates_which_candidate_replays_first(
    tmp_path: Path,
) -> None:
    """Otherwise a round always re-spends its budget on the same head of the list."""

    config, journal, rules, evalset = _stores(
        tmp_path, rule_trial_min_sessions=99  # never conclude; observe ordering only
    )
    first = rules.add("rule alpha", "x", metric="task_completion")
    second = rules.add("rule beta", "y", metric="task_completion")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )

    first_round = [v.rule_id for v in await engine.evaluate_candidates()]
    second_round = [v.rule_id for v in await engine.evaluate_candidates()]

    assert first_round == [first.id, second.id]
    assert second_round == [first.id, second.id][::-1]


async def test_env_wait_is_not_charged_to_the_replay_budget(tmp_path: Path) -> None:
    """A replay queued for a shared environment must not "time out" before running.

    AppWorld serializes every lifecycle behind one world lock, so this wait is
    routine; charging it to the run budget reports an inconclusive replay for a
    candidate that never executed.
    """

    config, journal, rules, evalset = _stores(
        tmp_path, replay_timeout_s=5.0, replay_env_wait_timeout_s=5.0
    )
    candidate = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.script_trace("s-replayed", task_completion=0.92, coherence=0.88)

    async def replay(case: EvalCase, context: Any) -> str:
        del case
        await asyncio.sleep(0.05)  # queueing for the environment
        context.mark_execution_started()
        await _read_candidate(context, candidate)
        return "s-replayed"

    verdict = await ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        replay=replay,
    ).validate(candidate)

    assert verdict.outcome == "promote"


async def test_a_replay_that_never_reaches_its_environment_stays_recoverable(
    tmp_path: Path,
) -> None:
    config, journal, rules, evalset = _stores(
        tmp_path, replay_timeout_s=5.0, replay_env_wait_timeout_s=0.05
    )
    candidate = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    _seed_failure_case(evalset)

    async def replay(case: EvalCase, context: Any) -> str:
        del case, context
        await asyncio.sleep(30)  # never acquires the environment
        return "s-never"

    verdict = await ReplayValidator(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        replay=replay,
    ).validate(candidate)

    assert verdict.outcome == "pending"
    assert verdict.pending_reason == "env_wait_timeout"
    # Recoverable: the candidate is untouched and can be validated again.
    assert [rule.status for rule in rules.candidates()] == ["candidate"]


async def test_validation_telemetry_explains_pending_and_terminal_states(
    tmp_path: Path,
) -> None:
    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=2)
    rules.add("verify before retrying", "x", metric="task_completion")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )

    await engine.evaluate_candidates()  # pending: the trial window is not full
    engine.observe_report("s1", set())
    engine.observe_report("s2", set())
    await engine.evaluate_candidates()  # now promotable

    events = journal.recent(limit=0, types=("validation_verdict",))
    reasons = [event.get("pending_reason") for event in events]
    outcomes = [event.get("outcome") for event in events]
    assert "trial_in_progress" in reasons
    assert outcomes[-1] == "promote"
    rounds = journal.recent(limit=0, types=("validation_round_finished",))
    assert rounds[-1]["promoted"] == 1
    assert rounds[-1]["decided"] == 1
    starts = journal.recent(limit=0, types=("validation_round_started",))
    assert starts and starts[0]["queued_rule_ids"]


async def test_repeated_validation_is_idempotent(tmp_path: Path) -> None:
    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=1)
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )
    engine.observe_report("s1", set())

    first = await engine.evaluate_candidates()
    second = await engine.evaluate_candidates()
    third = await engine.evaluate_candidates()

    assert [v.outcome for v in first] == ["promote"]
    assert second == [] and third == []  # nothing left to decide
    assert [rule.id for rule in rules.active()] == [candidate.id]
    promotions = journal.recent(limit=0, types=("rule_promote",))
    assert len(promotions) == 1  # promoted once, not once per round


async def test_a_concurrent_verdict_cannot_double_apply(tmp_path: Path) -> None:
    """Two rounds racing on one candidate produce one transition, not a conflict."""

    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=1)
    candidate = rules.add("verify before retrying", "x", metric="task_completion")
    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )
    engine.observe_report("s1", set())

    results = await asyncio.gather(
        engine.evaluate_candidates(), engine.evaluate_candidates()
    )

    # Both rounds may reach the same conclusion — the store is what serializes it.
    # `promote` raises KeyError once the rule is no longer a candidate, so the
    # second application is a caught no-op rather than a conflicting transition.
    assert [v.outcome for batch in results for v in batch] == ["promote", "promote"]
    assert [rule.id for rule in rules.active()] == [candidate.id]
    assert len(journal.recent(limit=0, types=("rule_promote",))) == 1
    assert [rule.status for rule in rules.all()] == ["active"]


async def test_a_retirement_journals_its_structured_evidence(tmp_path: Path) -> None:
    """The deltas that decided a retirement must survive, not just the prose."""

    config, journal, rules, evalset = _stores(tmp_path)
    candidate = rules.add(
        "verify before retrying", "x", metric="task_completion",
        tags=["stall:task_completion"],
    )
    _seed_failure_case(evalset)
    fake = FakeCliClient()
    fake.script_trace("s-replayed", task_completion=0.05, coherence=0.4)

    async def replay(case: EvalCase, context: Any) -> str:
        await _read_candidate(context, candidate)
        return "s-replayed"

    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(fake, config),
        journal=journal,
        replay=replay,
    )
    await engine.evaluate_candidates()

    (event,) = journal.recent(limit=0, types=("rule_retire",))
    assert event["validator"] == "replay"
    assert event["evidence"]["cases"][0]["deltas"]["task_completion"] < 0
    assert event["evidence"]["cases"][0]["candidate_surfaced"] is True
    cases = journal.recent(limit=0, types=("validation_replay_case",))
    assert cases and cases[0]["outcome"] == "scored"


async def test_a_trial_record_without_the_newer_fields_still_loads(
    tmp_path: Path,
) -> None:
    config, journal, rules, evalset = _stores(tmp_path, rule_trial_min_sessions=1)
    path = config.rules_store_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "id": "r-minimal",
                "created_at": "2026-01-01T00:00:00+00:00",
                "rule": "a persisted candidate with a minimal trial",
                "rationale": "x",
                "status": "candidate",
                "scope": "scoped",
                "trial": {"observed_sessions": ["s-old"], "breached_sessions": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    (candidate,) = rules.candidates()
    assert candidate.trial is not None
    assert candidate.trial.replay_attempts == 0
    assert candidate.scope == "scoped"  # an explicit scope is respected as-is

    engine = ValidationEngine(
        config=config,
        rules=rules,
        evalset=evalset,
        evaluator=MetricEvaluator(FakeCliClient(), config),
        journal=journal,
    )
    (verdict,) = await engine.evaluate_candidates()
    assert verdict.outcome == "promote"
