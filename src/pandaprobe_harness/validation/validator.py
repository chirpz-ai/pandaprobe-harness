"""Candidate-rule validation: evidence before trust.

A rule written by the agent enters as a ``candidate``. Two validators can
produce the evidence that promotes (or retires) it:

* :class:`ReplayValidator` — the strong path. Replays the failing scenario(s)
  whose signature matches the candidate (plus a small sample of protected
  ``win`` cases) through the developer-supplied ``ReplayFn`` with the
  candidate in force, scores the new sessions via the ``MetricEvaluator``,
  and promotes iff the targeted metric improves past ``rule_promote_margin``
  with no case regressing past ``rule_regress_margin``.
* :class:`ForwardTrialValidator` — the automatic fallback when no replay
  function is wired. The hook feeds every handled report into the engine's
  trial bookkeeping; once ``rule_trial_min_sessions`` distinct sessions have
  been observed, the candidate's breach rate is compared against the
  baseline captured at add time.

:class:`ValidationEngine` owns strategy selection, the trial observations,
and verdict application. It never raises into the hook: every failure is
caught, logged, and degrades to "no verdict this round". Replay scoring
builds a fresh ``TurnContext`` and calls the evaluator directly — the live
hook's ``_pending``/refresh bookkeeping is never touched.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from ..config import HarnessConfig
from ..evaluation.evaluator import MetricEvaluator
from ..hook.turn import TurnContext
from ..workspace.evalset import EvalCase, EvalSet, ReplayFn
from ..workspace.journal import Journal
from ..workspace.rules import Rule, RulesStore, TrialState

__all__ = [
    "ForwardTrialValidator",
    "ReplayValidator",
    "RuleValidator",
    "ValidationEngine",
    "ValidationVerdict",
    "VerdictOutcome",
]

logger = logging.getLogger("pandaprobe_harness.validation")

VerdictOutcome = Literal["promote", "retire", "pending"]

#: Failing cases replayed per candidate (newest matching first).
_MAX_FAILURE_CASES = 3
#: Win cases replayed alongside, to catch collateral regressions.
_MAX_WIN_CASES = 2
#: Inconclusive replay rounds tolerated before relying on the forward trial.
_MAX_REPLAY_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class ValidationVerdict:
    """One validator's decision about one candidate rule."""

    rule_id: str
    outcome: VerdictOutcome
    validator: Literal["replay", "forward_trial", "none"]
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


class RuleValidator(Protocol):
    """Pluggable candidate-validation strategy."""

    async def validate(self, rule: Rule) -> ValidationVerdict: ...


def _report_matches(rule: Rule, signatures: set[str]) -> bool:
    """Does a report's signature set hit the rule's metric family?

    Signatures look like ``breach:agent_reliability``; with no metric on the
    rule, any alerting signature counts against the trial.
    """

    if rule.metric:
        suffix = f":{rule.metric}"
        return any(signature.endswith(suffix) for signature in signatures)
    return bool(signatures)


def _target_signatures(rule: Rule) -> tuple[str, ...]:
    """The signatures used to find eval cases matching the candidate."""

    signatures = [tag for tag in rule.tags if ":" in tag]
    if rule.metric:
        breach = f"breach:{rule.metric}"
        if breach not in signatures:
            signatures.append(breach)
    return tuple(signatures)


def _metric_of(signature: str) -> str | None:
    _, sep, metric = signature.partition(":")
    return metric if sep else None


class ForwardTrialValidator:
    """Statistical fallback: watch the candidate over live sessions."""

    def __init__(self, *, config: HarnessConfig) -> None:
        self._config = config

    async def validate(self, rule: Rule) -> ValidationVerdict:
        trial = rule.trial
        needed = self._config.rule_trial_min_sessions
        observed = 0 if trial is None else len(trial.observed_sessions)
        if trial is None or observed < needed:
            return ValidationVerdict(
                rule_id=rule.id,
                outcome="pending",
                validator="forward_trial",
                reason=f"trial in progress: {observed}/{needed} sessions observed",
            )
        trial_rate = trial.trial_rate
        baseline = trial.baseline_rate
        details = {
            "trial_rate": trial_rate,
            "baseline_rate": baseline,
            "observed_sessions": observed,
        }
        if trial_rate == 0.0 or trial_rate <= baseline - self._config.rule_promote_margin:
            return ValidationVerdict(
                rule_id=rule.id,
                outcome="promote",
                validator="forward_trial",
                reason=(
                    f"forward-trial: breach rate {trial_rate:.2f} vs baseline "
                    f"{baseline:.2f} over {observed} sessions"
                ),
                details=details,
            )
        return ValidationVerdict(
            rule_id=rule.id,
            outcome="retire",
            validator="forward_trial",
            reason=(
                f"forward-trial: breach rate {trial_rate:.2f} did not improve on "
                f"baseline {baseline:.2f} after {observed} sessions"
            ),
            details=details,
        )


class ReplayValidator:
    """The strong path: replay matching eval cases with the candidate in force."""

    def __init__(
        self,
        *,
        config: HarnessConfig,
        rules: RulesStore,
        evalset: EvalSet,
        evaluator: MetricEvaluator,
        replay: ReplayFn,
    ) -> None:
        self._config = config
        self._rules = rules
        self._evalset = evalset
        self._evaluator = evaluator
        self._replay = replay

    async def validate(self, rule: Rule) -> ValidationVerdict:
        targets = _target_signatures(rule)
        matching = await asyncio.to_thread(self._evalset.matching, targets)
        failures = [case for case in matching if case.replayable][:_MAX_FAILURE_CASES]
        if not failures:
            return ValidationVerdict(
                rule_id=rule.id,
                outcome="pending",
                validator="replay",
                reason="no replayable eval case matches the candidate",
            )
        wins_all = await asyncio.to_thread(lambda: self._evalset.cases(kind="win"))
        wins = [case for case in reversed(wins_all) if case.replayable][:_MAX_WIN_CASES]

        # The full render includes the candidate (provisional section), so the
        # replayed run executes with the rule in force.
        context = await asyncio.to_thread(self._rules.render_markdown)
        target_metric = rule.metric or _metric_of(failures[0].signature[0])

        improved = False
        regression: str | None = None
        inconclusive = 0
        conclusive_failures = 0
        case_details: list[dict[str, Any]] = []
        for case in failures + wins:
            outcome = await self._replay_scores(case, context)
            deltas: dict[str, float] = {}
            if outcome is not None:
                new_session, scores = outcome
                deltas = {
                    metric: value - case.baseline_scores[metric]
                    for metric, value in scores.items()
                    if metric in case.baseline_scores
                }
            if outcome is None or not deltas:
                # No replay/scores, or no metric shared with the baseline:
                # this case is evidence of nothing, for or against.
                inconclusive += 1
                continue
            if case.kind == "failure":
                conclusive_failures += 1
            case_details.append(
                {
                    "case_id": case.id,
                    "kind": case.kind,
                    "session_id": new_session,
                    "deltas": deltas,
                }
            )
            for metric, delta in deltas.items():
                if delta <= -self._config.rule_regress_margin:
                    regression = f"case {case.id} metric {metric} (Δ={delta:+.2f})"
            if case.kind == "failure" and self._improved(target_metric, deltas):
                improved = True

        details = {"cases": case_details, "inconclusive": inconclusive}
        if regression is not None:
            return ValidationVerdict(
                rule_id=rule.id,
                outcome="retire",
                validator="replay",
                reason=f"replay: regression on {regression}",
                details=details,
            )
        if improved:
            metric_label = target_metric or "the targeted metric"
            return ValidationVerdict(
                rule_id=rule.id,
                outcome="promote",
                validator="replay",
                reason=f"replay: {metric_label} improved past margin on the failing scenario",
                details=details,
            )
        if conclusive_failures == 0:
            # Retiring requires evidence too: without one successfully
            # replayed+scored failure case there is none either way.
            return ValidationVerdict(
                rule_id=rule.id,
                outcome="pending",
                validator="replay",
                reason="replay inconclusive: no failing case produced comparable scores",
                details=details,
            )
        return ValidationVerdict(
            rule_id=rule.id,
            outcome="retire",
            validator="replay",
            reason=(
                "replay: no improvement on "
                + ", ".join(case.id for case in failures)
            ),
            details=details,
        )

    def _improved(self, target_metric: str | None, deltas: dict[str, float]) -> bool:
        margin = self._config.rule_promote_margin
        if target_metric is not None:
            return deltas.get(target_metric, 0.0) >= margin
        return any(delta >= margin for delta in deltas.values())

    async def _replay_scores(
        self, case: EvalCase, context: str
    ) -> tuple[str, dict[str, float]] | None:
        """Replay + score one case; ``None`` means inconclusive (never raises).

        The developer replay is time-bounded: a hung replay would otherwise
        wedge the single-flight validation task for the process lifetime.
        """

        try:
            new_session = str(
                await asyncio.wait_for(
                    self._replay(case, context), self._config.replay_timeout_s
                )
            )
        except Exception as exc:  # noqa: BLE001 - a broken/hung replay is inconclusive
            logger.warning("replay failed for case %s: %s", case.id, exc or type(exc).__name__)
            return None
        try:
            report = await self._evaluator.evaluate_turn(
                TurnContext(session_id=new_session, turn_index=0)
            )
        except Exception as exc:  # noqa: BLE001 - scoring failure is inconclusive
            logger.warning("scoring replayed session %s failed: %s", new_session, exc)
            return None
        scores = {
            str(score.metric): score.value
            for score in report.scores
            if score.value is not None
        }
        if not scores:
            return None
        return new_session, scores


class ValidationEngine:
    """Strategy selection + trial observation + verdict application."""

    def __init__(
        self,
        *,
        config: HarnessConfig,
        rules: RulesStore,
        evalset: EvalSet,
        evaluator: MetricEvaluator,
        journal: Journal,
        replay: ReplayFn | None = None,
    ) -> None:
        self._config = config
        self._rules = rules
        self._journal = journal
        self._forward = ForwardTrialValidator(config=config)
        self._replay_validator: ReplayValidator | None = None
        if replay is not None:
            self._replay_validator = ReplayValidator(
                config=config,
                rules=rules,
                evalset=evalset,
                evaluator=evaluator,
                replay=replay,
            )
        self._no_replay_logged = False
        # (rules.jsonl mtime_ns + size, candidates) — the per-report observation
        # path must be a cheap stat() when nothing changed. Size participates
        # because coarse-timestamp filesystems can leave mtime_ns unchanged
        # across appends; the store is append-only, so size always moves.
        self._candidate_cache: tuple[tuple[int, int], list[Rule]] | None = None

    @property
    def has_replay(self) -> bool:
        return self._replay_validator is not None

    # -- trial observation (sync; callers wrap in asyncio.to_thread) ------------

    def _candidates(self) -> list[Rule]:
        try:
            stat = self._config.rules_store_file.stat()
        except OSError:
            return []
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._candidate_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        candidates = self._rules.candidates()
        self._candidate_cache = (key, candidates)
        return candidates

    def observe_report(self, session_id: str, signatures: set[str]) -> None:
        """Record one handled report against every open forward trial.

        Every report counts (healthy or alerting) — the trial needs the
        denominator, not just the failures. A session counts as breached
        when ANY of its observed reports matched the rule's metric family.
        The mutation runs against the FRESH trial under the store lock
        (via ``update_trial``'s closure), so concurrent observers from other
        sessions' report handlers cannot erase each other's evidence.
        """

        min_sessions = self._config.rule_trial_min_sessions
        for rule in self._candidates():
            matched = _report_matches(rule, signatures)

            def _observe(trial: TrialState) -> TrialState:
                already_observed = session_id in trial.observed_sessions
                if not already_observed and len(trial.observed_sessions) >= min_sessions:
                    return trial  # window full; only known sessions may update
                observed = trial.observed_sessions
                breached = trial.breached_sessions
                if not already_observed:
                    observed = (*observed, session_id)
                if matched and session_id not in breached:  # noqa: B023 - applied immediately
                    breached = (*breached, session_id)
                if observed == trial.observed_sessions and breached == trial.breached_sessions:
                    return trial
                return replace(trial, observed_sessions=observed, breached_sessions=breached)

            try:
                self._rules.update_trial(rule.id, _observe)
            except KeyError:
                continue  # promoted/retired concurrently; nothing to record

    # -- candidate evaluation -----------------------------------------------------

    async def evaluate_candidates(self) -> list[ValidationVerdict]:
        """Validate every candidate and apply the verdicts. Never raises."""

        verdicts: list[ValidationVerdict] = []
        try:
            candidates = await asyncio.to_thread(self._rules.candidates)
        except Exception:  # noqa: BLE001 - degrade, never break the caller
            logger.exception("failed to load candidate rules for validation")
            return verdicts
        if not candidates:
            return verdicts

        if self._replay_validator is None and not self._no_replay_logged:
            self._no_replay_logged = True
            logger.warning(
                "no replay function wired — candidate rules fall back to forward-trial "
                "validation over the next %s live sessions (pass replay=... to "
                "Harness.create for replay-based validation)",
                self._config.rule_trial_min_sessions,
            )
            try:
                await asyncio.to_thread(
                    self._journal.record,
                    {
                        "type": "validation",
                        "mode": "forward_trial",
                        "reason": "no replay function wired",
                    },
                )
            except Exception:  # noqa: BLE001 - journaling is best-effort here
                logger.debug("failed to journal validation fallback", exc_info=True)

        for stale in candidates:
            # Earlier candidates' (slow, replay-bound) validations may have run
            # for a while, and observe_report keeps appending trial evidence —
            # judge every candidate on its FRESH state, not the round-start
            # snapshot, and skip it if it was promoted/retired meanwhile.
            rule = await asyncio.to_thread(self._fresh_candidate, stale.id)
            if rule is None:
                continue
            try:
                verdict = await self._validate_one(rule)
            except Exception:  # noqa: BLE001 - one bad candidate must not stop the rest
                logger.exception("validation failed for rule %s", rule.id)
                continue
            verdicts.append(verdict)
            if verdict.outcome == "pending":
                continue
            try:
                await asyncio.to_thread(self._apply, rule, verdict)
            except KeyError:
                logger.debug("rule %s changed state before its verdict applied", rule.id)
            except Exception:  # noqa: BLE001 - degrade, never break the caller
                logger.exception("failed to apply verdict for rule %s", rule.id)
        return verdicts

    def _fresh_candidate(self, rule_id: str) -> Rule | None:
        for rule in self._rules.candidates():
            if rule.id == rule_id:
                return rule
        return None

    async def _validate_one(self, rule: Rule) -> ValidationVerdict:
        if self._replay_validator is not None:
            attempts = rule.trial.replay_attempts if rule.trial is not None else 0
            if attempts < _MAX_REPLAY_ATTEMPTS:
                verdict = await self._replay_validator.validate(rule)
                if verdict.outcome != "pending":
                    return verdict
                if "inconclusive" in verdict.reason and rule.trial is not None:
                    # Mutate the FRESH trial under the lock — bumping a
                    # round-start snapshot would roll back sessions observed
                    # while the (slow) replay round was running.
                    def _bump(trial: TrialState) -> TrialState:
                        return replace(trial, replay_attempts=trial.replay_attempts + 1)

                    try:
                        await asyncio.to_thread(self._rules.update_trial, rule.id, _bump)
                    except KeyError:
                        pass
        return await self._forward.validate(rule)

    def _apply(self, rule: Rule, verdict: ValidationVerdict) -> None:
        # Stamp the verdict through the atomic mutate-under-the-lock, so trial
        # evidence appended by concurrent observers between our snapshot and
        # now is preserved — writing a snapshot-derived trial here would erase
        # it from the terminal record.
        label = "promoted" if verdict.outcome == "promote" else "retired"

        def _stamp(trial: TrialState) -> TrialState:
            return replace(trial, verdict=f"{label}:{verdict.reason}")

        self._rules.update_trial(rule.id, _stamp)
        if verdict.outcome == "promote":
            self._rules.promote(rule.id, reason=verdict.reason, validator=verdict.validator)
        elif verdict.outcome == "retire":
            self._rules.retire(rule.id, reason=verdict.reason)
