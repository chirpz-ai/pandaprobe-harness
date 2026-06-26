"""The lifecycle hook and context injector.

``PandaHarnessHook`` is non-blocking by design. ``on_turn_end`` schedules the
evaluation as a detached ``asyncio.Task`` (the producing turn never waits).
``drain_pending`` is the bounded await-barrier the adapter calls at the start of
the *next* turn: it joins the in-flight evaluation (up to ``drain_timeout_s``),
feeds each resolved score to the local EWMA **trend detector**, and — when an
alert condition fires and is not currently suppressed — dumps the telemetry
payload and injects a System/Trend alert into the agent's next-turn message
queue. It never mutates framework checkpoints, and never raises into the host
agent loop.

``startup_context()`` returns the living ``harness_rules.md`` as a system-prompt
preamble, closing the self-healing loop (learned rules are re-read each run).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ..cli.client import CliClient
from ..config import HarnessConfig
from ..evaluation.evaluator import MetricEvaluator
from ..evaluation.history import ScoreHistoryStore
from ..evaluation.metrics import EvalReport
from ..evaluation.trends import TrendDetector
from .alert import build_system_alert, build_trend_alert
from .context import compose_system_preamble

if TYPE_CHECKING:
    from ..adapters.protocol import FrameworkAdapter
    from ..filesystem.layout import HarnessFilesystem

__all__ = ["PandaHarnessHook"]

logger = logging.getLogger("pandaprobe_harness.hook")


@dataclass
class _SessionAlertState:
    signatures: set[str] = field(default_factory=set)
    cooldown: int = 0


class PandaHarnessHook:
    """Pluggable, framework-agnostic turn-completion hook."""

    def __init__(
        self,
        adapter: FrameworkAdapter,
        cli: CliClient,
        *,
        config: HarnessConfig | None = None,
        filesystem: HarnessFilesystem | None = None,
        evaluator: MetricEvaluator | None = None,
    ) -> None:
        self._adapter = adapter
        self._config = config or HarnessConfig()
        self._cli = cli
        self._evaluator = evaluator or MetricEvaluator(cli, self._config)
        if filesystem is None:
            # Imported lazily to avoid a hard cycle at module import time.
            from ..filesystem.layout import HarnessFilesystem

            filesystem = HarnessFilesystem(self._config)
        self._filesystem = filesystem

        self._detector: TrendDetector | None = None
        if self._config.enable_trend:
            store = ScoreHistoryStore(self._config)
            self._detector = TrendDetector(self._config, store)

        self._pending: dict[str, asyncio.Task[EvalReport]] = {}
        self._alert_state: dict[str, _SessionAlertState] = {}
        self._draining: set[str] = set()

    # -- startup context (rules -> agent context) ----------------------------

    def startup_context(self) -> str:
        """Living harness rules, for prepending to the agent's system prompt."""

        return compose_system_preamble(self._filesystem)

    # -- producing side (turn end) -------------------------------------------

    def on_turn_end(self, raw_turn: object) -> None:
        """Schedule evaluation for a completed turn. Returns immediately."""

        try:
            ctx = self._adapter.parse_turn(raw_turn)
        except Exception:  # noqa: BLE001 - never break the host loop
            logger.exception("failed to parse turn; skipping evaluation")
            return

        # If a prior turn's eval is still in flight for this session, cancel it;
        # the newest turn supersedes it (avoid orphaning a detached task).
        prev = self._pending.get(ctx.session_id)
        if prev is not None and not prev.done():
            prev.cancel()

        task: asyncio.Task[EvalReport] = asyncio.ensure_future(
            self._evaluator.evaluate_turn(ctx)
        )
        self._pending[ctx.session_id] = task

    # -- consuming side (next turn start) ------------------------------------

    async def drain_pending(self, session_id: str) -> EvalReport | None:
        """Await the in-flight eval for ``session_id`` and act on it.

        Bounded by ``drain_timeout_s``. On timeout the task is left in place to
        be drained on a later turn. Returns the (trend-annotated) report when one
        was processed, else ``None``.
        """

        # Guard against concurrent/re-entrant drains for the same session so a
        # resolved report is never handled (history-recorded, dedup-advanced)
        # more than once. Does not disturb the on-timeout "task remains" path.
        if session_id in self._draining:
            return None
        task = self._pending.get(session_id)
        if task is None:
            return None

        self._draining.add(session_id)
        try:
            try:
                report = await asyncio.wait_for(
                    asyncio.shield(task), self._config.drain_timeout_s
                )
            except TimeoutError:
                logger.info("eval for session=%s not ready within drain budget", session_id)
                return None  # task left in _pending for a later drain
            except Exception:  # noqa: BLE001 - degrade gracefully
                logger.exception("eval task for session=%s failed", session_id)
                self._pending.pop(session_id, None)
                return None

            self._pending.pop(session_id, None)
            return await self._handle_report(report)
        finally:
            self._draining.discard(session_id)

    async def _handle_report(self, report: EvalReport) -> EvalReport:
        # 1. Feed every resolved score to the trend detector (records history +
        #    sets trend/relative flags). Runs in a thread (sync store I/O).
        if self._detector is not None:
            report = await asyncio.to_thread(self._apply_trends, report)

        # 2. Decide whether to alert (dedup / cooldown / recovery).
        if not self._should_alert(report):
            return report

        # 3. Dump diagnostics + inject the appropriate alert flavor.
        try:
            payload = await self._build_dump(report)
            await asyncio.to_thread(self._filesystem.write_latest_eval, payload)
            alert = self._select_alert(report)
            self._adapter.inject_alert(alert)
        except Exception:  # noqa: BLE001 - never break the host loop
            logger.exception("failed to dump/inject alert for session=%s", report.session_id)
        return report

    # -- trend application ----------------------------------------------------

    def _apply_trends(self, report: EvalReport) -> EvalReport:
        assert self._detector is not None
        updated = []
        for score in report.scores:
            if not score.pending and score.value is not None:
                verdict = self._detector.update(
                    report.session_id, str(score.metric), score.value
                )
                score = replace(
                    score,
                    trend_declining=verdict.declining,
                    relative_breach=score.relative_breach or verdict.relative_breach,
                    percentile_breach=verdict.percentile_breach,
                )
            updated.append(score)
        return replace(report, scores=tuple(updated))

    # -- alert decisioning ----------------------------------------------------

    @staticmethod
    def _signatures(report: EvalReport) -> set[str]:
        sigs: set[str] = set()
        for score in report.scores:
            metric = str(score.metric)
            if score.breached:
                sigs.add(f"breach:{metric}")
            if score.relative_breach:
                sigs.add(f"relative:{metric}")
            if score.trend_declining:
                sigs.add(f"trend:{metric}")
            if score.percentile_breach:
                sigs.add(f"percentile:{metric}")
        return sigs

    def _should_alert(self, report: EvalReport) -> bool:
        """Dedup/cooldown gate; resets on recovery. Mutates per-session state."""

        session_id = report.session_id
        current = self._signatures(report)
        if not current:
            self._alert_state.pop(session_id, None)  # recovery
            return False

        state = self._alert_state.get(session_id, _SessionAlertState())
        new_conditions = current - state.signatures
        cooldown_turns = self._config.alert_cooldown_turns

        inject = bool(new_conditions) or (
            current == state.signatures and cooldown_turns > 0 and state.cooldown <= 0
        )

        if inject:
            self._alert_state[session_id] = _SessionAlertState(
                signatures=set(current), cooldown=cooldown_turns
            )
        else:
            self._alert_state[session_id] = _SessionAlertState(
                signatures=set(current), cooldown=max(0, state.cooldown - 1)
            )
        return inject

    def _select_alert(self, report: EvalReport) -> str:
        critical = any(s.breached or s.relative_breach for s in report.scores)
        if critical:
            return build_system_alert(report, self._config)
        return build_trend_alert(report, self._config)

    async def _build_dump(self, report: EvalReport) -> dict[str, Any]:
        payload = report.to_dump()
        if self._config.enrich_flagged_traces and report.flagged_traces:
            trace_id = report.flagged_traces[0]
            try:
                result = await self._cli.run("traces", "get", trace_id, "--kind", "TOOL")
                payload["flagged_trace_detail"] = result.json()
            except Exception:  # noqa: BLE001 - enrichment is best-effort
                logger.debug("flagged-trace enrichment failed for %s", trace_id)
        return payload
