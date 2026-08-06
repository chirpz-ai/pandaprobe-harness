"""Single-flight notice assignment and settlement integration."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from contextvars import Context
from datetime import UTC, datetime
from functools import partial
from typing import Any, cast

from ..agent_tools.toolset import RepairToolset
from ..cli.client import CliClient
from ..config import HarnessConfig
from ..hook.turn import TurnContext
from ..workspace.journal import Journal
from ..workspace.mailbox import DiagnosticNotice, Mailbox
from ..workspace.rules import RulesStore
from ..workspace.sanitize import is_sensitive_key, sanitize_text
from .agent import ManagedRepairAgent
from .models import RepairAssignment, RepairResult

__all__ = ["ManagedRepairCoordinator"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ManagedRepairCoordinator:
    """Claim each notice once in-process and cache its structured outcome."""

    def __init__(
        self,
        *,
        config: HarnessConfig,
        cli: CliClient,
        mailbox: Mailbox,
        journal: Journal,
        rules: RulesStore,
        agent: ManagedRepairAgent,
    ) -> None:
        self._config = config
        self._cli = cli
        self._mailbox = mailbox
        self._journal = journal
        self._rules = rules
        self._agent = agent
        self._lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[RepairResult]] = {}
        self._results: dict[str, RepairResult] = {}
        self._started_at: dict[str, str] = {}
        self._latest_by_session: dict[str, str] = {}
        self._turns: dict[str, TurnContext] = {}

    def remember_turn(self, turn: TurnContext) -> None:
        self._turns[turn.session_id] = turn
        if len(self._turns) > 4096:
            self._turns.pop(next(iter(self._turns)), None)

    async def settle(self, session_id: str, *, timeout_s: float) -> RepairResult | None:
        notice = await self._pending_notice(session_id)
        if notice is None:
            latest = self._latest_by_session.get(session_id)
            if latest is None:
                return None
            result = self._results.get(latest)
            if result is not None:
                return result
            task = self._tasks.get(latest)
            if task is None:
                return None
            done, _ = await asyncio.wait({task}, timeout=max(0.0, timeout_s))
            return await task if task in done else None

        task = await self._single_flight(notice)
        if task.done():
            return await task
        done, _ = await asyncio.wait({task}, timeout=max(0.0, timeout_s))
        if task in done:
            return await task

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        result = RepairResult(
            task_session_id=session_id,
            repair_session_id=_repair_session_id(session_id, notice.id),
            notice_id=notice.id,
            status="timed_out",
            started_at=self._started_at.get(notice.id, _now()),
            ended_at=_now(),
            model=self._config.repair_model or "",
            provider=(self._config.repair_model or "").partition("/")[0],
            error_category="settlement_timeout",
            message="repair exceeded the settlement budget; notice remains pending",
            tracing_enabled=self._config.trace_repair_agent,
        )
        async with self._lock:
            self._results.setdefault(notice.id, result)
            self._latest_by_session[session_id] = notice.id
            self._tasks.pop(notice.id, None)
        await asyncio.to_thread(
            self._journal.record, {"type": "repair_timed_out", **result.to_json()}
        )
        return self._results[notice.id]

    async def _pending_notice(self, session_id: str) -> DiagnosticNotice | None:
        notices = await asyncio.to_thread(self._mailbox.pending)
        return next((notice for notice in notices if notice.session_id == session_id), None)

    async def _single_flight(self, notice: DiagnosticNotice) -> asyncio.Task[RepairResult]:
        async with self._lock:
            existing_result = self._results.get(notice.id)
            if existing_result is not None:
                return _completed_task(existing_result)
            existing_task = self._tasks.get(notice.id)
            if existing_task is not None:
                return existing_task

            assignment = self._assignment(notice)
            tools = RepairToolset(
                config=self._config,
                cli=self._cli,
                mailbox=self._mailbox,
                journal=self._journal,
                rules=self._rules,
                notice_id=notice.id,
                allowed_trace_ids=notice.flagged_traces,
            )
            # A new Context prevents an active task-agent trace from becoming the
            # repair completion's parent. The SDK session is set inside transport.
            task = asyncio.create_task(
                self._agent.run(assignment, tools),
                name=f"pandaprobe-repair:{notice.id}",
                context=Context(),
            )
            self._tasks[notice.id] = task
            self._started_at[notice.id] = _now()
            self._latest_by_session[notice.session_id] = notice.id
            task.add_done_callback(partial(self._capture_result, notice))
            return task

    def _assignment(self, notice: DiagnosticNotice) -> RepairAssignment:
        turn = self._turns.get(notice.session_id)
        descriptor: dict[str, Any] = {}
        if turn is not None and turn.end_state:
            # Sanitization and output bounds happen in the notice/prompt path;
            # retain only a shallow, JSON-oriented descriptor here.
            descriptor = _bounded_descriptor(
                turn.end_state, max_len=self._config.sanitize_max_len
            )
        return RepairAssignment(
            task_session_id=notice.session_id,
            repair_session_id=_repair_session_id(notice.session_id, notice.id),
            turn_index=notice.turn_index,
            notice=notice,
            task_descriptor=descriptor,
            domain_policy=sanitize_text(
                self._config.domain_policy, max_len=self._config.sanitize_max_len
            )
            or None,
        )

    def _capture_result(
        self, notice: DiagnosticNotice, task: asyncio.Task[RepairResult]
    ) -> None:
        if task.cancelled():
            return
        try:
            result = task.result()
        except Exception:  # pragma: no cover - agent folds ordinary exceptions
            return
        self._results.setdefault(notice.id, result)
        self._latest_by_session[notice.session_id] = notice.id
        self._tasks.pop(notice.id, None)


def _repair_session_id(task_session_id: str, notice_id: str) -> str:
    return f"repair-{task_session_id}-{notice_id}"


def _completed_task(result: RepairResult) -> asyncio.Task[RepairResult]:
    async def completed() -> RepairResult:
        return result

    return asyncio.create_task(completed())


def _bounded_descriptor(value: Any, *, max_len: int) -> dict[str, Any]:
    """Keep useful task metadata while excluding obvious credential fields."""

    def scrub(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                str(key): "[redacted]" if is_sensitive_key(key) else scrub(child)
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [scrub(child) for child in item[:50]]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        return str(item)

    scrubbed = cast(dict[str, Any], scrub(dict(value)))
    encoded = json.dumps(scrubbed, sort_keys=True, default=str)
    if len(encoded) <= max_len:
        return scrubbed
    return {"summary": encoded[:max_len]}
