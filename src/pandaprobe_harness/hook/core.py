"""Component 1: the lifecycle hook and context injector.

``PandaHarnessHook`` is non-blocking by design. ``on_turn_end`` schedules the
evaluation as a detached ``asyncio.Task`` (the producing turn never waits).
``drain_pending`` is the bounded await-barrier the adapter calls at the start of
the *next* turn: it joins the in-flight evaluation (up to ``drain_timeout_s``)
and, on a threshold breach, dumps the telemetry payload and injects a System
Alert into the agent's next-turn message queue. It never mutates framework
checkpoints, and never raises into the host agent loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ..cli.client import CliClient
from ..config import HarnessConfig
from ..evaluation.evaluator import MetricEvaluator
from ..evaluation.metrics import EvalReport
from .alert import build_system_alert

if TYPE_CHECKING:
    from ..adapters.protocol import FrameworkAdapter
    from ..filesystem.layout import HarnessFilesystem

__all__ = ["PandaHarnessHook"]

logger = logging.getLogger("pandaprobe_harness.hook")


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
        self._pending: dict[str, asyncio.Task[EvalReport]] = {}

    # -- producing side (turn end) -------------------------------------------

    def on_turn_end(self, raw_turn: object) -> None:
        """Schedule evaluation for a completed turn. Returns immediately."""

        try:
            ctx = self._adapter.parse_turn(raw_turn)
        except Exception:  # noqa: BLE001 - never break the host loop
            logger.exception("failed to parse turn; skipping evaluation")
            return

        task: asyncio.Task[EvalReport] = asyncio.ensure_future(
            self._evaluator.evaluate_turn(ctx)
        )
        # If a prior turn's eval is still pending for this session, drop it; the
        # newest turn supersedes it.
        self._pending[ctx.session_id] = task

    # -- consuming side (next turn start) ------------------------------------

    async def drain_pending(self, session_id: str) -> EvalReport | None:
        """Await the in-flight eval for ``session_id`` and act on a breach.

        Bounded by ``drain_timeout_s``. On timeout the task is left in place to
        be drained on a later turn (the alert simply arrives as soon as ready).
        Returns the report when one was processed, else ``None``.
        """

        task = self._pending.get(session_id)
        if task is None:
            return None

        try:
            report = await asyncio.wait_for(
                asyncio.shield(task), self._config.drain_timeout_s
            )
        except TimeoutError:
            logger.info("eval for session=%s not ready within drain budget", session_id)
            return None
        except Exception:  # noqa: BLE001 - degrade gracefully
            logger.exception("eval task for session=%s failed", session_id)
            self._pending.pop(session_id, None)
            return None

        self._pending.pop(session_id, None)
        await self._handle_report(report)
        return report

    async def _handle_report(self, report: EvalReport) -> None:
        if not report.any_breach:
            return
        try:
            await asyncio.to_thread(self._filesystem.write_latest_eval, report.to_dump())
            alert = build_system_alert(report, self._config)
            self._adapter.inject_alert(alert)
        except Exception:  # noqa: BLE001 - never break the host loop
            logger.exception("failed to dump/inject alert for session=%s", report.session_id)
