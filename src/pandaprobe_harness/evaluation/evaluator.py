"""Orchestrates metric evaluation against the PandaProbe platform.

Both evaluation scopes run through one batch/poll pipeline, differing only in the
``--target`` and the id flag:

1. ``evals runs batch --target <trace|session>
   <--trace-ids|--session-ids> <ids> --metrics <m1,m2>``
   (asynchronous; returns a ``run_id``).
2. Poll ``evals runs scores <run_id> --target <trace|session>`` until terminal
   (bounded).
3. Map each score record to a ``MetricScore`` against its threshold.

* :meth:`MetricEvaluator.evaluate_trace` is the v2 path — one trace, the metrics
  of one tier.
* :meth:`MetricEvaluator.evaluate_turn` is the v1 session path, kept intact for
  ``trigger_mode="session"`` and the ablation.

``--signal-weights`` is session-only on the platform and is therefore sent only
on the session path.

Trace ingestion lags turn-end (the SDK flushes on a background thread), so the
run is retried with backoff while it looks transiently empty/not-found. Every
CLI failure ultimately degrades to a *pending* (``None``) score — the harness
never raises into, or blocks, the host agent loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence

from ..cli.client import CliClient
from ..cli.errors import CliError, is_transient
from ..cli.models import RunCreated, RunScores, ScoreRecord
from ..config import HarnessConfig
from ..hook.turn import TurnContext
from .metrics import EvalReport, Metric, MetricScore
from .traces import TraceLocator

__all__ = ["MetricEvaluator"]

logger = logging.getLogger("pandaprobe_harness.evaluation")


class MetricEvaluator:
    """Computes the configured session metrics for a turn."""

    def __init__(
        self,
        cli: CliClient,
        config: HarnessConfig,
        *,
        locator: TraceLocator | None = None,
    ) -> None:
        self._cli = cli
        self._config = config
        # Owned rather than injected so `score_last_trace` works for any caller
        # holding only an evaluator (the replay validator and regression runner).
        self._locator = locator or TraceLocator(cli, config)

    @property
    def locator(self) -> TraceLocator:
        return self._locator

    async def evaluate_turn(self, ctx: TurnContext) -> EvalReport:
        """Evaluate the configured session metrics for ``ctx`` (v1 session path)."""

        metrics = self._active_metrics()
        if not metrics:
            return EvalReport.from_scores(ctx.session_id, ctx.turn_index, [])
        scores = await self._run_batch("session", [ctx.session_id], metrics)
        return EvalReport.from_scores(ctx.session_id, ctx.turn_index, scores)

    async def evaluate_trace(
        self, trace_id: str, metric_names: Sequence[str], *, tier: int = 0
    ) -> list[MetricScore]:
        """Score one trace against ``metric_names``, stamping trace id and tier."""

        return await self.evaluate_traces([trace_id], metric_names, tier=tier)

    async def evaluate_traces(
        self, trace_ids: Sequence[str], metric_names: Sequence[str], *, tier: int = 0
    ) -> list[MetricScore]:
        """Score several traces in **one** platform run, attributing each score.

        One create + one poll loop regardless of how many traces, instead of a
        round-trip each: the batch endpoint takes a list of ids and every score
        comes back tagged with its ``trace_id``.
        """

        metrics = self._resolve(metric_names, target="trace")
        if not metrics or not trace_ids:
            return []
        return await self._run_batch("trace", trace_ids, metrics, tier=tier)

    async def score_last_trace(
        self, session_id: str, metric_names: Sequence[str]
    ) -> list[MetricScore]:
        """Score a session's most recent completed trace.

        The re-scoring primitive for rule validation and regression: a replay
        produces a fresh session, and its last trace carries the whole trajectory,
        so it is graded on the same signal that triggered in the first place.
        """

        ref = await self._locator.last_trace(session_id)
        if ref is None:
            logger.info("no completed trace to score for session=%s", session_id)
            return []
        return await self.evaluate_trace(ref.trace_id, metric_names)

    async def score_for_trigger(self, session_id: str) -> dict[str, float]:
        """Re-score a session on whichever signal the configured trigger uses.

        The one place that mapping lives, shared by candidate-rule validation and
        the regression runner: grading a replay on the *same* axis that fired is
        what makes a promote/retire verdict mean anything. v1 graded replays with
        the session composites — the metric that floors at roughly the same value
        for every session — which is why promotion was close to random.
        """

        if self._config.trigger_mode == "session":
            report = await self.evaluate_turn(
                TurnContext(session_id=session_id, turn_index=0)
            )
            scores: Sequence[MetricScore] = report.scores
        else:
            scores = await self.score_last_trace(
                session_id, self._config.replay_metrics()
            )
        return {
            str(score.metric): score.value
            for score in scores
            if score.value is not None
        }

    # -- orchestration --------------------------------------------------------

    async def _run_batch(
        self,
        target: str,
        ids: Sequence[str],
        metrics: list[Metric],
        *,
        tier: int = 0,
    ) -> list[MetricScore]:
        names = [str(m) for m in metrics]
        attempts = max(1, self._config.eval_retry_attempts)
        # Trace runs attribute each score to its trace; a session run has no
        # per-id dimension, so it collapses to one lookup by name.
        trace_ids: Sequence[str | None] = ids if target == "trace" else [None]
        last: list[MetricScore] | None = None

        for attempt in range(attempts):
            try:
                run = await self._create_run(target, ids, names)
                run_scores = await self._poll_scores(run.run_id, target)
            except CliError as exc:
                if is_transient(exc) and attempt + 1 < attempts:
                    await self._backoff(attempt)
                    continue
                logger.warning(
                    "%s eval degraded for %s: %s", target, ",".join(ids), exc
                )
                return self._all_pending(metrics, trace_ids, tier=tier)

            results = [
                self._score_for(
                    m,
                    (
                        run_scores.for_trace(str(m), trace_id)
                        if trace_id is not None
                        else run_scores.by_name(str(m))
                    ),
                    trace_id=trace_id,
                    tier=tier,
                )
                for trace_id in trace_ids
                for m in metrics
            ]
            # Retry only when the run never reached a terminal state (no scores
            # yet / still computing — i.e. trace-ingestion lag). A terminal run
            # whose scores FAILED is final and must NOT be retried.
            if not run_scores.is_terminal() and attempt + 1 < attempts:
                last = results
                await self._backoff(attempt)
                continue
            return results

        return last if last is not None else self._all_pending(metrics, trace_ids, tier=tier)

    async def _backoff(self, attempt: int) -> None:
        await asyncio.sleep(self._config.eval_retry_backoff_s * (attempt + 1))

    # -- CLI calls ------------------------------------------------------------

    async def _create_run(
        self, target: str, ids: Sequence[str], metric_names: list[str]
    ) -> RunCreated:
        id_flag = "--session-ids" if target == "session" else "--trace-ids"
        args = [
            "evals", "runs", "batch",
            "--target", target,
            id_flag, ",".join(ids),
            "--metrics", ",".join(metric_names),
        ]
        # `--signal-weights` is session-only on the platform; sending it on the
        # trace path is a validation error.
        if target == "session" and self._config.signal_weights:
            args += ["--signal-weights", json.dumps(self._config.signal_weights)]
        result = await self._cli.run(*args)
        return RunCreated.parse(result.json())

    async def _poll_scores(self, run_id: str, target: str) -> RunScores:
        """Poll ``evals runs scores --target <target>`` until terminal/exhausted."""

        last = RunScores(run_id=run_id, scores=())
        for attempt in range(self._config.poll_max_attempts):
            result = await self._cli.run(
                "evals", "runs", "scores", run_id, "--target", target
            )
            last = RunScores.parse(run_id, result.json())
            if last.is_terminal():
                return last
            if attempt + 1 < self._config.poll_max_attempts:
                await asyncio.sleep(self._config.poll_interval_s)
        logger.info("run %s did not reach terminal state within poll budget", run_id)
        return last

    # -- mapping --------------------------------------------------------------

    def _active_metrics(self) -> list[Metric]:
        return self._resolve(self._config.active_metrics(), target="session")

    def _resolve(self, names: Sequence[str], *, target: str) -> list[Metric]:
        """Map metric names to enum members, dropping anything not ``target``-runnable."""

        out: list[Metric] = []
        for name in names:
            try:
                metric = Metric(name)
            except ValueError:
                logger.warning("unknown %s metric %r; skipping", target, name)
                continue
            if metric.target != target:
                logger.warning(
                    "metric %r is %s-scoped, not %s-runnable; skipping",
                    name,
                    metric.target,
                    target,
                )
                continue
            out.append(metric)
        return out

    def _threshold_for(self, metric: Metric, record: ScoreRecord | None) -> float:
        """Local threshold, falling back to the one the platform reported."""

        name = str(metric)
        if name not in self._config.thresholds and record is not None:
            reported = record.metadata.get("threshold")
            if isinstance(reported, (int, float)):
                return float(reported)
        return self._config.threshold_for(name)

    def _score_for(
        self,
        metric: Metric,
        record: ScoreRecord | None,
        *,
        trace_id: str | None = None,
        tier: int = 0,
    ) -> MetricScore:
        threshold = self._threshold_for(metric, record)
        if record is None:
            return MetricScore(
                metric=metric,
                value=None,
                threshold=threshold,
                trace_id=trace_id,
                tier=tier,
            )
        return MetricScore(
            metric=metric,
            value=record.value if record.is_terminal else None,
            threshold=threshold,
            reason=record.reason,
            metadata=record.metadata,
            trace_id=trace_id or record.trace_id,
            tier=tier,
        )

    def _all_pending(
        self,
        metrics: list[Metric],
        trace_ids: Sequence[str | None] = (None,),
        *,
        tier: int = 0,
    ) -> list[MetricScore]:
        return [
            MetricScore(
                metric=m,
                value=None,
                threshold=self._config.threshold_for(str(m)),
                trace_id=trace_id,
                tier=tier,
            )
            for trace_id in trace_ids
            for m in metrics
        ]
