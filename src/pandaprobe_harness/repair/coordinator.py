"""Single-flight notice assignment and settlement integration."""

from __future__ import annotations

import asyncio
import hashlib
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
from ..workspace.scopes import RESERVED_SCOPES
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
        self._turns: dict[tuple[str, int], TurnContext] = {}

    def remember_turn(self, turn: TurnContext) -> None:
        self._turns[(turn.session_id, turn.turn_index)] = turn
        if len(self._turns) > 4096:
            self._turns.pop(next(iter(self._turns)), None)

    async def settle(self, session_id: str, *, timeout_s: float) -> RepairResult | None:
        notices = await self._pending_episode(session_id)
        if not notices:
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

        episode_id = _episode_id(notices)
        task = await self._single_flight(notices)
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
            repair_session_id=_repair_session_id(session_id, episode_id),
            notice_id=notices[0].id,
            status="timed_out",
            started_at=self._started_at.get(episode_id, _now()),
            ended_at=_now(),
            model=self._config.repair_model or "",
            provider=(self._config.repair_model or "").partition("/")[0],
            episode_id=episode_id,
            notice_ids=tuple(notice.id for notice in notices),
            recommended_scope=_recommended_scope(notices),
            error_category="settlement_timeout",
            message="repair exceeded the settlement budget; notice remains pending",
            tracing_enabled=self._config.trace_repair_agent,
        )
        async with self._lock:
            self._latest_by_session[session_id] = episode_id
            self._tasks.pop(episode_id, None)
        await asyncio.to_thread(
            self._journal.record, {"type": "repair_timed_out", **result.to_json()}
        )
        return result

    async def _pending_episode(self, session_id: str) -> tuple[DiagnosticNotice, ...]:
        notices = await asyncio.to_thread(self._mailbox.pending)
        session_notices = [notice for notice in notices if notice.session_id == session_id]
        if not session_notices:
            return ()
        episode = [session_notices[0]]
        changed = True
        while changed:
            changed = False
            for notice in session_notices[1:]:
                if notice in episode or not any(_related(notice, item) for item in episode):
                    continue
                episode.append(notice)
                changed = True
        return tuple(episode)

    async def _single_flight(
        self, notices: tuple[DiagnosticNotice, ...]
    ) -> asyncio.Task[RepairResult]:
        episode_id = _episode_id(notices)
        async with self._lock:
            existing_result = self._results.get(episode_id)
            if existing_result is not None:
                return _completed_task(existing_result)
            existing_task = self._tasks.get(episode_id)
            if existing_task is not None:
                return existing_task

            assignment = self._assignment(notices)
            allowed_trace_ids = tuple(
                dict.fromkeys(
                    trace_id for notice in notices for trace_id in notice.flagged_traces
                )
            )
            tools = RepairToolset(
                config=self._config,
                cli=self._cli,
                mailbox=self._mailbox,
                journal=self._journal,
                rules=self._rules,
                notice_id=notices[0].id,
                notice_ids=assignment.notice_ids,
                episode_id=episode_id,
                recommended_scope=assignment.recommended_scope,
                scope_hints=assignment.scope_hints,
                generic_scopes=assignment.generic_scopes,
                allowed_trace_ids=allowed_trace_ids,
            )
            # A new Context prevents an active task-agent trace from becoming the
            # repair completion's parent. The SDK session is set inside transport.
            task = asyncio.create_task(
                self._agent.run(assignment, tools),
                name=f"pandaprobe-repair:{episode_id}",
                context=Context(),
            )
            self._tasks[episode_id] = task
            self._started_at[episode_id] = _now()
            self._latest_by_session[notices[0].session_id] = episode_id
            task.add_done_callback(
                partial(self._capture_result, notices[0].session_id, episode_id)
            )
            return task

    def _assignment(self, notices: tuple[DiagnosticNotice, ...]) -> RepairAssignment:
        notice = notices[0]
        episode_id = _episode_id(notices)
        turn = self._turns.get((notice.session_id, notice.turn_index))
        descriptor: dict[str, Any] = {}
        if turn is not None and turn.end_state:
            # Sanitization and output bounds happen in the notice/prompt path;
            # retain only a shallow, JSON-oriented descriptor here. `task_summary`
            # is dropped: it is already a first-class assignment field, sanitized
            # and bounded, and repeating it here would only spend prompt budget.
            descriptor = _bounded_descriptor(
                {
                    key: value
                    for key, value in turn.end_state.items()
                    if key != "task_summary"
                },
                max_len=self._config.sanitize_max_len,
            )
        scope_hints = (
            tuple(hint.to_json() for hint in turn.rule_scope_hints)
            if turn is not None
            else ()
        )
        if not scope_hints:
            scope_hints = tuple(
                dict(hint) for item in notices for hint in item.scope_hints
            )
        # Host identity labels — the integration's own name for itself. They
        # describe where the agent runs, never what failed, so they must not
        # become rule scopes. Collected here so the toolset can reject them.
        generic_scopes = tuple(
            str(descriptor[key])
            for key in ("benchmark", "integration", "host")
            if descriptor.get(key)
        )
        return RepairAssignment(
            task_session_id=notice.session_id,
            repair_session_id=_repair_session_id(notice.session_id, episode_id),
            turn_index=notice.turn_index,
            notice=notice,
            episode_id=episode_id,
            notices=notices,
            scope_hints=scope_hints,
            recommended_scope=_recommended_scope(notices),
            generic_scopes=generic_scopes,
            task_descriptor=descriptor,
            task_summary=turn.task_summary if turn is not None else "",
            domain_policy=sanitize_text(
                self._config.domain_policy, max_len=self._config.sanitize_max_len
            )
            or None,
        )

    def _capture_result(
        self, session_id: str, episode_id: str, task: asyncio.Task[RepairResult]
    ) -> None:
        if task.cancelled():
            return
        try:
            result = task.result()
        except Exception:  # pragma: no cover - agent folds ordinary exceptions
            return
        if result.succeeded:
            self._results.setdefault(episode_id, result)
        self._latest_by_session[session_id] = episode_id
        self._tasks.pop(episode_id, None)


def _repair_session_id(task_session_id: str, episode_id: str) -> str:
    return f"repair-{task_session_id}-{episode_id}"


def _episode_id(notices: tuple[DiagnosticNotice, ...]) -> str:
    identity = "|".join(sorted(notice.id for notice in notices))
    return "e-" + hashlib.sha256(identity.encode()).hexdigest()[:12]


def _recommended_scope(notices: tuple[DiagnosticNotice, ...]) -> str | None:
    """The most precise host recommendation across a coalesced episode.

    ``None`` when no notice carried one — the repair model then chooses freely.
    A reserved name is not a recommendation: it carries no topic information the
    model does not already have.
    """

    for notice in notices:
        scope = notice.recommended_scope
        if scope and scope not in RESERVED_SCOPES:
            return scope
    return None


def _related(left: DiagnosticNotice, right: DiagnosticNotice) -> bool:
    if left.session_id != right.session_id or left.turn_index != right.turn_index:
        return False
    if set(left.flagged_traces) & set(right.flagged_traces):
        return True
    if set(left.signatures) & set(right.signatures):
        return True
    return False


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
