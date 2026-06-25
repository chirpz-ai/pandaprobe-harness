"""Orchestrates per-turn metric evaluation against the PandaProbe platform.

The flow for a completed turn:

1. Resolve the turn's trace IDs via ``traces list --session-id``.
2. Launch the configured metric runs (``evals runs batch`` is asynchronous and
   returns a ``run_id``).
3. Poll ``evals runs scores <run-id>`` until terminal (bounded).
4. Compare each score to its threshold and assemble an ``EvalReport``.

Every CLI failure is caught and degrades to a *pending* (``None``) score so the
harness never raises into — or blocks — the host agent loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Sequence

from ..cli.client import CliClient
from ..cli.errors import CliError
from ..cli.models import RunCreated, RunScores
from ..config import HarnessConfig
from ..hook.turn import TurnContext
from .metrics import EvalReport, Metric, MetricScore

__all__ = ["MetricEvaluator"]

logger = logging.getLogger("pandaprobe_harness.evaluation")


class MetricEvaluator:
    """Computes ``agent_reliability`` / ``agent_consistency`` for a turn."""

    def __init__(self, cli: CliClient, config: HarnessConfig) -> None:
        self._cli = cli
        self._config = config

    async def evaluate_turn(self, ctx: TurnContext) -> EvalReport:
        """Evaluate the configured metrics for ``ctx`` and return a report."""

        coros: list[Awaitable[MetricScore]] = []

        if self._config.eval_reliability:
            coros.append(self._safe_metric(Metric.RELIABILITY, ctx))
        if self._config.eval_consistency:
            coros.append(self._safe_metric(Metric.CONSISTENCY, ctx))

        if not coros:
            return EvalReport.from_scores(ctx.session_id, ctx.turn_index, [])

        if self._config.concurrent_eval:
            scores = await asyncio.gather(*coros)
        else:
            scores = [await coro for coro in coros]

        return EvalReport.from_scores(ctx.session_id, ctx.turn_index, scores)

    # -- per-metric evaluation ------------------------------------------------

    async def _safe_metric(self, metric: Metric, ctx: TurnContext) -> MetricScore:
        """Evaluate one metric, degrading any CLI failure to a pending score."""

        threshold = self._threshold_for(metric)
        try:
            return await self._evaluate_metric(metric, ctx, threshold)
        except CliError as exc:
            logger.warning(
                "metric %s degraded for session=%s turn=%s: %s",
                metric,
                ctx.session_id,
                ctx.turn_index,
                exc,
            )
            return MetricScore(metric=metric, value=None, threshold=threshold)

    async def _evaluate_metric(
        self, metric: Metric, ctx: TurnContext, threshold: float
    ) -> MetricScore:
        if metric is Metric.RELIABILITY:
            trace_ids = await self._latest_trace_ids(ctx.session_id)
            if not trace_ids:
                logger.info("no traces found for session=%s; skipping %s", ctx.session_id, metric)
                return MetricScore(metric=metric, value=None, threshold=threshold)
            run = await self._create_run(metric, trace_ids=trace_ids)
        else:
            run = await self._create_run(metric, session_ids=[ctx.session_id])

        scores = await self._poll_scores(run.run_id)
        record = scores.by_name(str(metric))
        if record is None:
            return MetricScore(metric=metric, value=None, threshold=threshold)
        return MetricScore(
            metric=metric,
            value=record.value if record.is_terminal else None,
            threshold=threshold,
            reason=record.reason,
            metadata=record.metadata,
        )

    # -- CLI calls ------------------------------------------------------------

    async def _latest_trace_ids(self, session_id: str, limit: int = 25) -> list[str]:
        result = await self._cli.run(
            "traces", "list", "--session-id", session_id, "--limit", str(limit)
        )
        payload = result.json()
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        ids: list[str] = []
        for item in items or []:
            if isinstance(item, dict):
                trace_id = item.get("trace_id") or item.get("id")
                if trace_id:
                    ids.append(str(trace_id))
        return ids

    async def _create_run(
        self,
        metric: Metric,
        *,
        trace_ids: Sequence[str] | None = None,
        session_ids: Sequence[str] | None = None,
    ) -> RunCreated:
        args = ["evals", "runs", "batch", "--target", metric.target, "--metrics", str(metric)]
        if trace_ids is not None:
            args += ["--trace-ids", ",".join(trace_ids)]
        if session_ids is not None:
            args += ["--session-ids", ",".join(session_ids)]
        result = await self._cli.run(*args)
        return RunCreated.parse(result.json())

    async def _poll_scores(self, run_id: str) -> RunScores:
        """Poll ``evals runs scores`` until terminal or attempts exhausted."""

        last = RunScores(run_id=run_id, scores=())
        for attempt in range(self._config.poll_max_attempts):
            result = await self._cli.run("evals", "runs", "scores", run_id)
            last = RunScores.parse(run_id, result.json())
            if last.is_terminal():
                return last
            if attempt + 1 < self._config.poll_max_attempts:
                await asyncio.sleep(self._config.poll_interval_s)
        logger.info("run %s did not reach terminal state within poll budget", run_id)
        return last

    def _threshold_for(self, metric: Metric) -> float:
        if metric is Metric.RELIABILITY:
            return self._config.reliability_threshold
        return self._config.consistency_threshold
