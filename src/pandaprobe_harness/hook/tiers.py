"""The three-tier trace evaluation ladder — cost-aware escalation.

Every LLM-judge metric reads the whole target trace, so cost scales with the
*number of metrics run*, not which ones. The tiers exist to spend that budget
only where it buys something:

* **Tier 1** — ``task_completion`` + ``coherence``, on **every** new trace. The
  always-on progress signal. It breaches on a trajectory (see
  ``evaluation/trajectory.py``), never a single low value, so a healthy climbing
  session costs one judge call per trace and never raises a notice.
* **Tier 2** — ``tool_correctness`` + ``argument_correctness``, on the **last
  trace only**, and only once Tier 1 has breached. The last trace carries the
  whole trajectory in its context and is the state that actually needs fixing, so
  this is the surgical diagnosis that drives a scoped rule.
* **Tier 3** — planning/efficiency metrics, last trace only, only on a confirmed
  Tier-2 breach, and opt-in. It never breaches on its own; it exists to give the
  agent better material for the rule.

An optional developer verifier contributes one synthetic ``outcome_correct``
score. Everything downstream keys off ``EvalReport.scores``, so a verifier breach
flows through notice → capture → validation with no extra wiring.

This lives beside the hook rather than inside ``MetricEvaluator`` so the evaluator
stays a pure platform client, and beside ``core.py`` rather than inside it so the
escalation can be unit-tested on its own.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..config import HarnessConfig
from ..evaluation.evaluator import MetricEvaluator
from ..evaluation.metrics import EvalReport, Metric, MetricScore
from ..evaluation.traces import TraceLocator, TraceRef
from ..evaluation.trajectory import TrajectoryGate
from .turn import TurnContext

__all__ = ["TierRunner", "VerifierFn", "run_verifier"]

logger = logging.getLogger("pandaprobe_harness.hook")

#: A developer-supplied outcome oracle: ``(session_id, end_state) -> score``.
#: Returns a float in ``[0, 1]`` (higher is better), a bool, or ``None`` when it
#: cannot judge this turn (e.g. a grader that only scores a finished task). May be
#: sync or async — the runner adapts either.
VerifierFn = Callable[
    [str, Mapping[str, Any]], "float | bool | None | Awaitable[float | bool | None]"
]


class TierRunner:
    """Runs the tiered trace evaluation for one turn and reports the outcome."""

    def __init__(
        self,
        config: HarnessConfig,
        evaluator: MetricEvaluator,
        locator: TraceLocator,
        gate: TrajectoryGate,
        *,
        verifier: VerifierFn | None = None,
    ) -> None:
        self._config = config
        self._evaluator = evaluator
        self._locator = locator
        self._gate = gate
        self._verifier = verifier

    async def run(self, ctx: TurnContext) -> EvalReport:
        """Evaluate ``ctx``'s new traces and escalate as far as the gate warrants."""

        # The verifier depends only on the turn payload, so it runs alongside the
        # tier ladder rather than after it — on the barrier path its latency would
        # otherwise stack on top.
        tier_scores, outcome = await asyncio.gather(
            self._run_tiers(ctx), self._run_verifier(ctx)
        )
        scores = [*tier_scores, *([outcome] if outcome is not None else [])]
        return EvalReport.from_scores(ctx.session_id, ctx.turn_index, scores)

    # -- tiers -----------------------------------------------------------------

    async def _run_tiers(self, ctx: TurnContext) -> list[MetricScore]:
        traces = await self._locator.new_traces(ctx.session_id)
        if not traces:
            logger.info(
                "no new completed traces for session=%s turn=%s; tier 1 skipped",
                ctx.session_id,
                ctx.turn_index,
            )
            return []
        tier1 = await self._run_tier1(ctx.session_id, traces)
        if not any(score.gate_breached for score in tier1):
            return tier1
        return [*tier1, *await self._escalate(ctx.session_id, traces[-1])]

    async def _run_tier1(
        self, session_id: str, traces: list[TraceRef]
    ) -> list[MetricScore]:
        """Score every new trace, folding each into the gate in chronological order."""

        metrics = self._config.tier_metrics(1)
        if not metrics:
            return []
        # One platform run for the whole turn: the batch endpoint takes every
        # trace id at once and tags each score with its trace, so this is one
        # create + one poll loop instead of one per trace.
        raw = await self._evaluator.evaluate_traces(
            [ref.trace_id for ref in traces], metrics, tier=1
        )
        by_trace: dict[str | None, list[MetricScore]] = {}
        for score in raw:
            by_trace.setdefault(score.trace_id, []).append(score)
        # The gate's peak/stall fold is history-dependent, so it is applied one
        # trace at a time in chronological order — only the *evals* were batched.
        return await asyncio.to_thread(
            self._fold_gate, session_id, [by_trace.get(ref.trace_id, []) for ref in traces]
        )

    def _fold_gate(
        self, session_id: str, per_trace: list[list[MetricScore]]
    ) -> list[MetricScore]:
        """Blocking store I/O for the whole turn, in one thread hop."""

        out: list[MetricScore] = []
        for scores in per_trace:
            out.extend(self._gate.apply_all(session_id, scores))
        return out

    async def _escalate(self, session_id: str, last: TraceRef) -> list[MetricScore]:
        """Tier 2 on the last trace, then Tier 3 only if Tier 2 confirms a breach."""

        out: list[MetricScore] = []
        tier2_metrics = self._config.tier_metrics(2)
        if not tier2_metrics:
            return out
        tier2 = await self._evaluator.evaluate_trace(last.trace_id, tier2_metrics, tier=2)
        out.extend(tier2)
        if not any(score.breached for score in tier2):
            logger.info(
                "gate opened for session=%s but tier 2 is clean on trace=%s",
                session_id,
                last.trace_id,
            )
            return out
        tier3_metrics = self._config.tier_metrics(3)
        if tier3_metrics:
            out.extend(
                await self._evaluator.evaluate_trace(last.trace_id, tier3_metrics, tier=3)
            )
        return out

    # -- verifier --------------------------------------------------------------

    async def _run_verifier(self, ctx: TurnContext) -> MetricScore | None:
        """Run the developer verifier, if any, as a synthetic ``outcome_correct``."""

        value = await run_verifier(self._verifier, ctx.session_id, ctx.end_state)
        if value is None:
            return None
        return MetricScore(
            metric=Metric.OUTCOME,
            value=value,
            threshold=self._config.outcome_threshold,
            reason="developer-supplied outcome verifier",
            tier=0,
        )


async def run_verifier(
    verifier: VerifierFn | None, session_id: str, end_state: Mapping[str, Any]
) -> float | None:
    """Invoke a :data:`VerifierFn` and coerce its answer to a ``[0, 1]`` score.

    The whole calling convention in one place — sync or async, guarded, clamped —
    so the live turn path and replay validation cannot disagree about what a
    verifier is allowed to return. ``None`` means "no usable verdict", which is a
    normal answer: a grader may only be able to score a finished task.
    """

    if verifier is None:
        return None
    try:
        outcome = verifier(session_id, end_state)
        if inspect.isawaitable(outcome):
            outcome = await outcome
    except Exception:  # noqa: BLE001 - a broken verifier must not break the caller
        logger.exception("outcome verifier failed for session=%s", session_id)
        return None
    if isinstance(outcome, bool):
        return 1.0 if outcome else 0.0
    if isinstance(outcome, (int, float)):
        return max(0.0, min(1.0, float(outcome)))
    if outcome is not None:
        logger.warning(
            "outcome verifier returned %r for session=%s; expected float, bool or None",
            outcome,
            session_id,
        )
    return None
