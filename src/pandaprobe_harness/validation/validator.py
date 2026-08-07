"""Candidate-rule validation: evidence before trust.

A rule written by the agent enters as a ``candidate``. Two validators can
produce the evidence that promotes (or retires) it:

* :class:`ReplayValidator` — the strong path. Replays the failing scenario(s)
  whose signature matches the candidate (plus a small sample of protected
  ``win`` cases) through the developer-supplied ``ReplayFn`` with the
  candidate discoverable through read-only tools, scores the new sessions via
  the ``MetricEvaluator``,
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
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from ..agent_tools.toolset import TaskToolset
from ..config import HarnessConfig
from ..evaluation.evaluator import MetricEvaluator
from ..evaluation.metrics import Metric
from ..hook.context import compose_system_preamble
from ..hook.tiers import VerifierFn, run_verifier
from ..workspace.evalset import EvalCase, EvalSet, ReplayContext, ReplayFn
from ..workspace.journal import Journal
from ..workspace.rules import Rule, RulesStore, TrialState

__all__ = [
    "ForwardTrialValidator",
    "PendingReason",
    "ReplayValidator",
    "RuleValidator",
    "ValidationEngine",
    "ValidationVerdict",
    "VerdictOutcome",
]

logger = logging.getLogger("pandaprobe_harness.validation")

VerdictOutcome = Literal["promote", "retire", "pending"]

#: Why a candidate has no terminal verdict yet. A closed set, so persisted
#: telemetry can be read without parsing prose, and so the retry accounting below
#: keys off the reason rather than substring-matching a human-readable string.
PendingReason = Literal[
    "trial_in_progress",
    "no_matching_replayable_case",
    "replay_inconclusive",
    "candidate_not_exercised",
    "env_wait_timeout",
    "round_budget_exhausted",
]

#: Failing cases replayed per candidate (newest matching first).
_MAX_FAILURE_CASES = 3
#: Win cases replayed alongside, to catch collateral regressions.
_MAX_WIN_CASES = 2
#: Replay rounds tolerated for one candidate before relying on the forward trial.
_MAX_REPLAY_ATTEMPTS = 3
#: How often to check whether a queued replay has reached its environment.
_ENV_WAIT_POLL_S = 0.25


@dataclass(frozen=True, slots=True)
class ValidationVerdict:
    """One validator's decision about one candidate rule."""

    rule_id: str
    outcome: VerdictOutcome
    validator: Literal["replay", "forward_trial", "none"]
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    #: Set only when ``outcome`` is ``pending`` — why there is no verdict yet.
    pending_reason: PendingReason | None = None


class RuleValidator(Protocol):
    """Pluggable candidate-validation strategy."""

    async def validate(self, rule: Rule) -> ValidationVerdict: ...


def _report_matches(rule: Rule, signatures: set[str]) -> bool:
    """Does a report's signature set hit the rule's metric family?

    Signatures look like ``breach:tool_correctness``; with no metric on the
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
        for condition in ("breach", "stall", "regression"):
            signature = f"{condition}:{rule.metric}"
            if signature not in signatures:
                signatures.append(signature)
    return tuple(signatures)


def _target_for(
    preference: Sequence[str | None], deltas: Mapping[str, float]
) -> str | None:
    """The most authoritative metric in ``preference`` that this case measured.

    Falling back down the list matters: a verifier that cannot grade a particular
    task contributes no delta, and treating its absence as a failed target would
    veto promotion for every such case.
    """

    return next(
        (name for name in preference if name and name in deltas),
        None,
    )


def _metric_of(signature: str) -> str | None:
    _, sep, metric = signature.partition(":")
    return metric if sep else None


def _retirement_metrics(rule: Rule) -> frozenset[str]:
    """Metrics on a *failure* case whose decline may retire this candidate.

    A candidate is answerable for the metric it targets and for the authoritative
    outcome score. It is not answerable for every judged metric that happens to be
    scored alongside: those move on their own between runs, so treating any drop as
    causation retires rules for reasons unrelated to what they say. (In one observed
    run all four retirements fired this way — three on ``argument_correctness`` for
    candidates targeting completion or tool use, one on a ``coherence`` drop of 0.09
    against a natural spread of ~0.15.)

    Win cases are exempt from this narrowing: a win is the collateral-damage guard,
    so *any* metric regressing there is the signal it exists to catch.
    """

    metrics = {str(Metric.OUTCOME)}
    if rule.metric:
        metrics.add(rule.metric)
    metrics.update(
        metric for metric in (_metric_of(tag) for tag in rule.failure_signatures) if metric
    )
    metrics.update(metric for metric in (_metric_of(tag) for tag in rule.tags) if metric)
    return frozenset(metrics)


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
                pending_reason="trial_in_progress",
                details={"observed_sessions": observed, "sessions_needed": needed},
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
    """Replay cases with the candidate available through read-only task tools."""

    def __init__(
        self,
        *,
        config: HarnessConfig,
        rules: RulesStore,
        evalset: EvalSet,
        evaluator: MetricEvaluator,
        replay: ReplayFn,
        verifier: VerifierFn | None = None,
    ) -> None:
        self._config = config
        self._rules = rules
        self._evalset = evalset
        self._evaluator = evaluator
        self._replay = replay
        self._verifier = verifier

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
                pending_reason="no_matching_replayable_case",
                details={"target_signatures": list(targets)},
            )
        wins_all = await asyncio.to_thread(lambda: self._evalset.cases(kind="win"))
        wins = [case for case in reversed(wins_all) if case.replayable][:_MAX_WIN_CASES]

        # Replay exposes the same on-demand rule tools as a live task turn: the
        # candidate is discoverable immediately but never forced into context.
        # Crucially the store is narrowed to THIS candidate, so a delta cannot be
        # produced by five unrelated provisionals and then charged to this one.
        # Active rules stay visible — they are the baseline being measured against.
        task_tools = TaskToolset(
            config=self._config,
            rules=self._rules.restricted_to_candidates(frozenset({rule.id})),
        )
        context = ReplayContext(
            compose_system_preamble(),
            task_tools=task_tools,
            candidate_rule_id=rule.id,
        )
        # Trust order, most authoritative first. Resolved per case against the
        # deltas that actually arrived (see `_target_for`) rather than up front:
        # a verifier may legitimately have no verdict for a given task, and
        # picking `outcome_correct` anyway would then veto every promotion.
        preference = (str(Metric.OUTCOME), rule.metric, _metric_of(failures[0].signature[0]))
        # Metrics whose decline is attributable to this candidate. A drop on some
        # other judged metric of a failure case is run-to-run noise as often as
        # causation, and retiring on it discards good rules for reasons unrelated
        # to what they say (see `_retiring_regression`).
        retire_on = _retirement_metrics(rule)

        improved = False
        improved_on: str | None = None
        regression: str | None = None
        inconclusive = 0
        env_timeouts = 0
        unexercised = 0
        conclusive_failures = 0
        case_details: list[dict[str, Any]] = []
        for case in failures + wins:
            before = task_tools.surfaced_rule_ids
            outcome = await self._replay_scores(case, context)
            surfaced = rule.id in (task_tools.surfaced_rule_ids - before) or (
                rule.id in before
            )
            deltas: dict[str, float] = {}
            new_session = ""
            if outcome is not None:
                new_session, scores = outcome
                deltas = {
                    metric: value - case.baseline_scores[metric]
                    for metric, value in scores.items()
                    if metric in case.baseline_scores
                }
            detail: dict[str, Any] = {
                "case_id": case.id,
                "kind": case.kind,
                "session_id": new_session,
                "candidate_surfaced": surfaced,
                "deltas": deltas,
            }
            if outcome is None or not deltas:
                # No replay/scores, or no metric shared with the baseline:
                # this case is evidence of nothing, for or against.
                inconclusive += 1
                if outcome is None and not context.execution_started:
                    env_timeouts += 1
                    detail["outcome"] = "env_wait_timeout"
                else:
                    detail["outcome"] = "inconclusive"
                case_details.append(detail)
                continue
            if not surfaced:
                # The replay ran and scored, but never read the candidate. Whatever
                # moved, this rule did not move it — counting that as a verdict is
                # how a good rule gets retired for someone else's variance.
                unexercised += 1
                inconclusive += 1
                detail["outcome"] = "candidate_not_exercised"
                case_details.append(detail)
                continue
            if case.kind == "failure":
                conclusive_failures += 1
            detail["outcome"] = "scored"
            case_details.append(detail)
            for metric, delta in deltas.items():
                if delta <= -self._config.rule_regress_margin and (
                    case.kind == "win" or metric in retire_on
                ):
                    regression = f"case {case.id} metric {metric} (Δ={delta:+.2f})"
            if case.kind == "failure":
                target_metric = _target_for(preference, deltas)
                if self._improved(target_metric, deltas):
                    improved = True
                    improved_on = target_metric

        details: dict[str, Any] = {
            "cases": case_details,
            "inconclusive": inconclusive,
            "unexercised": unexercised,
            "env_wait_timeouts": env_timeouts,
            "retire_on": sorted(retire_on),
        }
        if regression is not None:
            return ValidationVerdict(
                rule_id=rule.id,
                outcome="retire",
                validator="replay",
                reason=f"replay: regression on {regression}",
                details=details,
            )
        if improved:
            metric_label = improved_on or "the targeted metric"
            return ValidationVerdict(
                rule_id=rule.id,
                outcome="promote",
                validator="replay",
                reason=f"replay: {metric_label} improved past margin on the failing scenario",
                details=details,
            )
        if conclusive_failures == 0:
            # Retiring requires evidence too: without one failure case that both
            # exercised the candidate and produced comparable scores there is
            # none either way.
            if env_timeouts and env_timeouts >= unexercised:
                reason = "replay inconclusive: environment never became available"
                pending: PendingReason = "env_wait_timeout"
            elif unexercised:
                reason = "replay inconclusive: the candidate was never read"
                pending = "candidate_not_exercised"
            else:
                reason = "replay inconclusive: no failing case produced comparable scores"
                pending = "replay_inconclusive"
            return ValidationVerdict(
                rule_id=rule.id,
                outcome="pending",
                validator="replay",
                reason=reason,
                pending_reason=pending,
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

    async def _bounded_replay(self, case: EvalCase, context: ReplayContext) -> object:
        """Await one replay, charging queueing and execution to separate budgets.

        Where a replay must first acquire a shared environment — one world, one
        container — waiting can consume the entire execution budget, and the
        resulting timeout looks identical to a hung agent: the case is recorded as
        inconclusive evidence about a rule that never ran. Hosts that signal via
        ``ReplayContext.mark_execution_started`` get ``replay_env_wait_timeout_s``
        to reach the starting line, then the full ``replay_timeout_s`` to run.

        With no wait budget configured (the default) this is exactly the previous
        single ``wait_for``.
        """

        task = asyncio.ensure_future(self._replay(case, context))
        wait_budget = max(0.0, self._config.replay_env_wait_timeout_s)
        if wait_budget <= 0:
            return await asyncio.wait_for(task, self._config.replay_timeout_s)

        loop = asyncio.get_running_loop()
        wait_deadline = loop.time() + wait_budget
        # Poll rather than await an Event: the signal may come from a worker
        # thread, where an asyncio primitive bound to this loop is unsafe to set.
        while not context.execution_started:
            if task.done():
                return await task
            if loop.time() >= wait_deadline:
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
                raise TimeoutError(
                    f"replay for case {case.id} never acquired its environment "
                    f"within {wait_budget:.0f}s"
                )
            await asyncio.sleep(min(_ENV_WAIT_POLL_S, max(0.0, wait_deadline - loop.time())))
        return await asyncio.wait_for(task, self._config.replay_timeout_s)

    async def _replay_scores(
        self, case: EvalCase, context: ReplayContext
    ) -> tuple[str, dict[str, float]] | None:
        """Replay + score one case; ``None`` means inconclusive (never raises).

        The developer replay is time-bounded: a hung replay would otherwise
        wedge the single-flight validation task for the process lifetime.
        """

        try:
            new_session = str(await self._bounded_replay(case, context))
        except Exception as exc:  # noqa: BLE001 - a broken/hung replay is inconclusive
            logger.warning("replay failed for case %s: %s", case.id, exc or type(exc).__name__)
            return None
        try:
            resolved = await self._evaluator.score_for_trigger(new_session)
        except Exception as exc:  # noqa: BLE001 - scoring failure is inconclusive
            logger.warning("scoring replayed session %s failed: %s", new_session, exc)
            return None
        # The case's `replay_input` is the same payload the live turn carried, so
        # it stands in for the end state — which is what lets a verifier keyed on a
        # task id grade a replay at all.
        payload = case.replay_input if isinstance(case.replay_input, Mapping) else {}
        outcome = await run_verifier(self._verifier, new_session, payload)
        if outcome is not None:
            resolved[str(Metric.OUTCOME)] = outcome
        if not resolved:
            return None
        return new_session, resolved


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
        verifier: VerifierFn | None = None,
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
                verifier=verifier,
            )
        self._no_replay_logged = False
        # (rules.jsonl mtime_ns + size, candidates) — the per-report observation
        # path must be a cheap stat() when nothing changed. Size participates
        # because coarse-timestamp filesystems can leave mtime_ns unchanged
        # across appends; the store is append-only, so size always moves.
        self._candidate_cache: tuple[tuple[int, int], list[Rule]] | None = None
        #: Last candidate examined, so the next round resumes after it instead of
        #: re-spending its budget on the same head of the list.
        self._cursor: str | None = None

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

        ordered = self._round_order(candidates)
        # Next round starts one past this round's head, so a budget that only
        # covers the first few candidates still works its way through all of them.
        self._cursor = ordered[0].id if ordered else None
        loop = asyncio.get_running_loop()
        budget = max(0.0, self._config.validation_round_budget_s)
        deadline = loop.time() + budget if budget else None
        await self._journal_event(
            "validation_round_started",
            {
                "queued_rule_ids": [rule.id for rule in ordered],
                "budget_s": budget or None,
                "has_replay": self._replay_validator is not None,
            },
        )
        budget_exhausted = False

        for stale in ordered:
            # Earlier candidates' (slow, replay-bound) validations may have run
            # for a while, and observe_report keeps appending trial evidence —
            # judge every candidate on its FRESH state, not the round-start
            # snapshot, and skip it if it was promoted/retired meanwhile.
            rule = await asyncio.to_thread(self._fresh_candidate, stale.id)
            if rule is None:
                continue
            # Past the budget, candidates still get a decision — just the cheap
            # forward-trial one instead of a replay slot that may never come.
            # Starving them of any verdict is how a promotable candidate sat
            # untouched with `replay_attempts=0` and an empty verdict.
            allow_replay = deadline is None or loop.time() < deadline
            budget_exhausted = budget_exhausted or not allow_replay
            try:
                verdict = await self._validate_one(rule, allow_replay=allow_replay)
            except Exception:  # noqa: BLE001 - one bad candidate must not stop the rest
                logger.exception("validation failed for rule %s", rule.id)
                continue
            verdicts.append(verdict)
            await self._journal_event(
                "validation_verdict",
                {
                    "rule_id": rule.id,
                    "outcome": verdict.outcome,
                    "validator": verdict.validator,
                    "reason": verdict.reason,
                    "pending_reason": (
                        verdict.pending_reason
                        if verdict.outcome == "pending"
                        else ("round_budget_exhausted" if not allow_replay else None)
                    ),
                    "details": verdict.details,
                },
            )
            if verdict.outcome == "pending":
                continue
            try:
                await asyncio.to_thread(self._apply, rule, verdict)
            except KeyError:
                logger.debug("rule %s changed state before its verdict applied", rule.id)
            except Exception:  # noqa: BLE001 - degrade, never break the caller
                logger.exception("failed to apply verdict for rule %s", rule.id)

        decided = sum(1 for verdict in verdicts if verdict.outcome != "pending")
        await self._journal_event(
            "validation_round_finished",
            {
                "decided": decided,
                "pending": len(verdicts) - decided,
                "promoted": sum(1 for v in verdicts if v.outcome == "promote"),
                "retired": sum(1 for v in verdicts if v.outcome == "retire"),
                "budget_exhausted": budget_exhausted,
                "elapsed_s": round(loop.time() - (deadline - budget), 3)
                if deadline is not None
                else None,
            },
        )
        return verdicts

    def _round_order(self, candidates: list[Rule]) -> list[Rule]:
        """Rotate the starting point so replay attention is not monopolized.

        Rounds are single-flight and replays are sequential, so a round that
        always restarts at the oldest candidate spends its whole budget on the
        same few rules while newer ones never get looked at. Resuming after the
        last candidate examined gives every candidate a turn across rounds.
        """

        if self._cursor is None:
            return candidates
        ids = [rule.id for rule in candidates]
        if self._cursor not in ids:
            return candidates
        start = ids.index(self._cursor) + 1
        return candidates[start:] + candidates[:start]

    async def _journal_event(self, event_type: str, fields: dict[str, Any]) -> None:
        """Record bounded validation telemetry; never break the round."""

        try:
            await asyncio.to_thread(
                self._journal.record, {"type": event_type, **fields}
            )
        except Exception:  # noqa: BLE001 - telemetry is best-effort
            logger.debug("failed to journal %s", event_type, exc_info=True)

    def _fresh_candidate(self, rule_id: str) -> Rule | None:
        for rule in self._rules.candidates():
            if rule.id == rule_id:
                return rule
        return None

    async def _validate_one(
        self, rule: Rule, *, allow_replay: bool = True
    ) -> ValidationVerdict:
        attempts = rule.trial.replay_attempts if rule.trial is not None else 0
        if self._replay_validator is not None and allow_replay:
            if attempts < _MAX_REPLAY_ATTEMPTS:
                await self._journal_event(
                    "validation_candidate_started",
                    {
                        "rule_id": rule.id,
                        "validator": "replay",
                        "replay_attempts": attempts,
                        "observed_sessions": (
                            len(rule.trial.observed_sessions) if rule.trial else 0
                        ),
                        "sessions_needed": self._config.rule_trial_min_sessions,
                    },
                )
                verdict = await self._replay_validator.validate(rule)
                for detail in verdict.details.get("cases", ()):
                    await self._journal_event(
                        "validation_replay_case", {"rule_id": rule.id, **detail}
                    )
                if verdict.outcome != "pending":
                    return verdict
                # Count EVERY inconclusive replay round, not just the ones whose
                # prose happened to say "inconclusive". Keying off the reason text
                # meant a candidate with no matching replayable case never
                # incremented, so it retried replay forever and the forward-trial
                # fallback below was unreachable.
                if rule.trial is not None:
                    # Mutate the FRESH trial under the lock — bumping a
                    # round-start snapshot would roll back sessions observed
                    # while the (slow) replay round was running.
                    def _bump(trial: TrialState) -> TrialState:
                        return replace(trial, replay_attempts=trial.replay_attempts + 1)

                    try:
                        await asyncio.to_thread(self._rules.update_trial, rule.id, _bump)
                    except KeyError:
                        pass
        await self._journal_event(
            "validation_candidate_started",
            {
                "rule_id": rule.id,
                "validator": "forward_trial",
                "replay_attempts": attempts,
                "observed_sessions": (
                    len(rule.trial.observed_sessions) if rule.trial else 0
                ),
                "sessions_needed": self._config.rule_trial_min_sessions,
                "reason": None if allow_replay else "round_budget_exhausted",
            },
        )
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
            self._rules.retire(
                rule.id,
                reason=verdict.reason,
                validator=verdict.validator,
                evidence=verdict.details,
            )
