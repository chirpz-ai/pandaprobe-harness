"""Orchestrates per-turn metric evaluation against the PandaProbe platform.

``agent_reliability`` and ``agent_consistency`` are **session-level** metrics, so
a completed turn is evaluated by ``session_id`` in a single batch run that covers
all configured metrics at once:

1. ``evals runs batch --target session --session-ids <id> --metrics <m1,m2>``
   (asynchronous; returns a ``run_id``).
2. Poll ``evals runs scores <run_id> --target session`` until terminal (bounded).
3. Map each ``SessionScoreResponse`` to a ``MetricScore`` against its threshold.

Trace ingestion lags turn-end (the SDK flushes on a background thread), so the
run is retried with backoff while it looks transiently empty/not-found. Every
CLI failure ultimately degrades to a *pending* (``None``) score — the harness
never raises into, or blocks, the host agent loop.
"""

from __future__ import annotations

import asyncio
import json
import logging

from ..cli.client import CliClient
from ..cli.errors import (
    CliApiError,
    CliError,
    CliNotFoundError,
)
from ..cli.models import RunCreated, RunScores, ScoreRecord
from ..config import HarnessConfig
from ..hook.turn import TurnContext
from .metrics import EvalReport, Metric, MetricScore

__all__ = ["MetricEvaluator"]

logger = logging.getLogger("pandaprobe_harness.evaluation")

_TRANSIENT_HINTS = ("no traces", "no session", "not yet", "empty", "pending", "no data")


class MetricEvaluator:
    """Computes the configured session metrics for a turn."""

    def __init__(self, cli: CliClient, config: HarnessConfig) -> None:
        self._cli = cli
        self._config = config

    async def evaluate_turn(self, ctx: TurnContext) -> EvalReport:
        """Evaluate the configured session metrics for ``ctx``."""

        metrics = self._active_metrics()
        if not metrics:
            return EvalReport.from_scores(ctx.session_id, ctx.turn_index, [])
        scores = await self._run_session(ctx.session_id, metrics)
        return EvalReport.from_scores(ctx.session_id, ctx.turn_index, scores)

    # -- orchestration --------------------------------------------------------

    async def _run_session(
        self, session_id: str, metrics: list[Metric]
    ) -> list[MetricScore]:
        names = [str(m) for m in metrics]
        attempts = max(1, self._config.eval_retry_attempts)
        last: list[MetricScore] | None = None

        for attempt in range(attempts):
            try:
                run = await self._create_session_run(session_id, names)
                run_scores = await self._poll_scores(run.run_id)
            except CliError as exc:
                if self._is_transient(exc) and attempt + 1 < attempts:
                    await self._backoff(attempt)
                    continue
                logger.warning(
                    "session eval degraded for session=%s: %s", session_id, exc
                )
                return self._all_pending(metrics)

            results = [self._score_for(m, run_scores.by_name(str(m))) for m in metrics]
            # Retry only when the run never reached a terminal state (no scores
            # yet / still computing — i.e. trace-ingestion lag). A terminal run
            # whose scores FAILED is final and must NOT be retried.
            if not run_scores.is_terminal() and attempt + 1 < attempts:
                last = results
                await self._backoff(attempt)
                continue
            return results

        return last if last is not None else self._all_pending(metrics)

    async def _backoff(self, attempt: int) -> None:
        await asyncio.sleep(self._config.eval_retry_backoff_s * (attempt + 1))

    @staticmethod
    def _is_transient(exc: CliError) -> bool:
        # Not-found (404, eventual-consistency lag) and other server errors
        # (5xx / 429, exit code 5) are retry-worthy. Auth (2), validation (4),
        # general (1), timeout and output-parse failures are not.
        if isinstance(exc, (CliNotFoundError, CliApiError)):
            return True
        text = (exc.result.stderr if exc.result else "").lower()
        return any(hint in text for hint in _TRANSIENT_HINTS)

    # -- CLI calls ------------------------------------------------------------

    async def _create_session_run(
        self, session_id: str, metric_names: list[str]
    ) -> RunCreated:
        args = [
            "evals", "runs", "batch",
            "--target", "session",
            "--session-ids", session_id,
            "--metrics", ",".join(metric_names),
        ]
        if self._config.signal_weights:
            args += ["--signal-weights", json.dumps(self._config.signal_weights)]
        result = await self._cli.run(*args)
        return RunCreated.parse(result.json())

    async def _poll_scores(self, run_id: str) -> RunScores:
        """Poll ``evals runs scores --target session`` until terminal/exhausted."""

        last = RunScores(run_id=run_id, scores=())
        for attempt in range(self._config.poll_max_attempts):
            result = await self._cli.run(
                "evals", "runs", "scores", run_id, "--target", "session"
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
        out: list[Metric] = []
        for name in self._config.active_metrics():
            try:
                out.append(Metric(name))
            except ValueError:
                logger.warning("unknown session metric %r; skipping", name)
        return out

    def _score_for(self, metric: Metric, record: ScoreRecord | None) -> MetricScore:
        threshold = self._config.threshold_for(str(metric))
        if record is None:
            return MetricScore(metric=metric, value=None, threshold=threshold)
        return MetricScore(
            metric=metric,
            value=record.value if record.is_terminal else None,
            threshold=threshold,
            reason=record.reason,
            metadata=record.metadata,
        )

    def _all_pending(self, metrics: list[Metric]) -> list[MetricScore]:
        return [
            MetricScore(metric=m, value=None, threshold=self._config.threshold_for(str(m)))
            for m in metrics
        ]
