"""Bounded package-owned repair-agent orchestration."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from ..agent_tools.toolset import RepairToolset
from ..config import HarnessConfig
from ..workspace.journal import Journal
from .completion import RepairCompletion
from .models import RepairAssignment, RepairResult, RepairStatus, RepairUsage
from .prompt import repair_messages

__all__ = ["ManagedRepairAgent"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ManagedRepairAgent:
    """Own the repair prompt and tool loop; callers supply only model transport."""

    def __init__(
        self,
        *,
        config: HarnessConfig,
        completion: RepairCompletion,
        journal: Journal,
    ) -> None:
        self._config = config
        self._completion = completion
        self._journal = journal

    async def run(
        self, assignment: RepairAssignment, tools: RepairToolset
    ) -> RepairResult:
        """Run one repair assignment inside one coherent agent trace."""

        import pandaprobe

        trace_context = self._trace_context(assignment)
        with pandaprobe.session(assignment.repair_session_id), trace_context as repair_trace:
            with repair_trace.span(
                "harness",
                kind="CHAIN",
                metadata={"role": "repair"},
            ) as repair_span:
                repair_span.set_input(_trace_assignment(assignment, self._config))
                result = await self._run_loop(assignment, tools, repair_trace)
                repair_span.set_output(result.to_json())
                repair_span.set_metadata({"repair_status": result.status})
            repair_trace.set_output(_trace_result_message(result))
            repair_trace.set_metadata(
                {
                    "repair_status": result.status,
                    "turns": result.turns,
                    "tool_calls": result.tool_calls,
                }
            )
            return result

    async def _run_loop(
        self,
        assignment: RepairAssignment,
        tools: RepairToolset,
        repair_trace: Any,
    ) -> RepairResult:
        started_at = _now()
        usage = RepairUsage()
        turns = 0
        tool_calls = 0
        await self._record(
            "repair_started",
            assignment,
            model=self._config.repair_model,
            provider=_provider(self._config.repair_model),
            tracing_enabled=self._config.trace_repair_agent,
        )
        messages: list[dict[str, Any]] = repair_messages(assignment)
        completed_tools: set[str] = set()
        deadline = asyncio.get_running_loop().time() + max(0.0, self._config.repair_timeout_s)

        try:
            for turns in range(1, max(1, self._config.repair_max_turns) + 1):
                remaining_time = deadline - asyncio.get_running_loop().time()
                remaining_tokens = max(0, self._config.repair_max_tokens - usage.output_tokens)
                if remaining_time <= 0:
                    return await self._finish(
                        assignment,
                        tools,
                        "timed_out",
                        started_at,
                        turns - 1,
                        tool_calls,
                        usage,
                        error_category="timeout",
                    )
                if remaining_tokens <= 0:
                    return await self._finish(
                        assignment,
                        tools,
                        "failed",
                        started_at,
                        turns - 1,
                        tool_calls,
                        usage,
                        error_category="token_limit",
                        message="repair output-token budget exhausted",
                    )

                with repair_trace.span(
                    "repair-agent",
                    kind="AGENT",
                    metadata={"role": "repair", "turn": turns},
                ) as model_span:
                    model_span.set_input(
                        {
                            "turn": turns,
                            "message_count": len(messages),
                            "remaining_output_tokens": remaining_tokens,
                        }
                    )
                    response = await asyncio.wait_for(
                        self._completion.complete(
                            model=self._config.repair_model or "",
                            messages=messages,
                            tools=_completion_tools(tools),
                            max_tokens=remaining_tokens,
                            temperature=self._config.repair_temperature,
                            reasoning_effort=self._config.repair_reasoning_effort,
                            timeout_s=remaining_time,
                        ),
                        timeout=remaining_time,
                    )
                    model_span.set_output(
                        {
                            "content_present": bool(response.content),
                            "tool_calls": [call.name for call in response.tool_calls],
                            "usage": response.usage.to_json(),
                        }
                    )
                usage = usage.plus(response.usage)
                await self._record(
                    "repair_model_turn",
                    assignment,
                    turn=turns,
                    tool_calls=len(response.tool_calls),
                    usage=response.usage.to_json(),
                )

                if not response.tool_calls:
                    if tools.resolved:
                        return await self._finish(
                            assignment, tools, _status(tools), started_at, turns, tool_calls, usage
                        )
                    return await self._finish(
                        assignment,
                        tools,
                        "failed",
                        started_at,
                        turns,
                        tool_calls,
                        usage,
                        error_category="empty_response",
                        message="repair response did not resolve the notice",
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            }
                            for call in response.tool_calls
                        ],
                    }
                )
                with repair_trace.span(
                    "tools",
                    kind="AGENT",
                    metadata={"role": "repair", "turn": turns},
                ) as tools_span:
                    tools_span.set_input(
                        {
                            "turn": turns,
                            "tool_calls": [call.name for call in response.tool_calls],
                        }
                    )
                    executed: list[dict[str, Any]] = []
                    for call in response.tool_calls:
                        tool_calls += 1
                        try:
                            raw_args = json.loads(call.arguments)
                        except (TypeError, ValueError):
                            tools_span.set_error("malformed_tool_arguments")
                            tools_span.set_output(
                                {"failed_tool": call.name, "reason": "invalid_json"}
                            )
                            return await self._finish(
                                assignment,
                                tools,
                                "failed",
                                started_at,
                                turns,
                                tool_calls,
                                usage,
                                error_category="malformed_tool_arguments",
                                message=f"invalid JSON arguments for {call.name}",
                            )
                        if not isinstance(raw_args, dict):
                            tools_span.set_error("malformed_tool_arguments")
                            tools_span.set_output(
                                {"failed_tool": call.name, "reason": "non_object"}
                            )
                            return await self._finish(
                                assignment,
                                tools,
                                "failed",
                                started_at,
                                turns,
                                tool_calls,
                                usage,
                                error_category="malformed_tool_arguments",
                                message=f"non-object arguments for {call.name}",
                            )
                        result = await self._call_tool(
                            repair_trace,
                            tools,
                            call.name,
                            raw_args,
                            turn=turns,
                            tool_call_id=call.id,
                        )
                        ok = result.get("ok") is True
                        if ok:
                            completed_tools.add(call.name)
                        executed.append({"tool": call.name, "ok": ok})
                        await self._record(
                            "repair_tool_call",
                            assignment,
                            turn=turns,
                            tool=call.name,
                            ok=ok,
                        )
                        if call.name == "harness_rule_add" and result.get("ok"):
                            rule = result.get("rule")
                            await self._record(
                                "repair_candidate_added",
                                assignment,
                                rule_id=rule.get("id") if isinstance(rule, dict) else None,
                                created=result.get("created"),
                            )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": call.name,
                                "content": json.dumps(result, sort_keys=True, default=str),
                            }
                        )
                        if tools.resolved:
                            break
                    tools_span.set_output(
                        {"tool_calls": executed, "notice_resolved": tools.resolved}
                    )

                if tools.resolved:
                    return await self._finish(
                        assignment, tools, _status(tools), started_at, turns, tool_calls, usage
                    )
                progress = _repair_progress(completed_tools, tools)
                if progress is not None:
                    messages.append({"role": "system", "content": progress})

            return await self._finish(
                assignment,
                tools,
                "failed",
                started_at,
                turns,
                tool_calls,
                usage,
                error_category="turn_limit",
                message="repair turn limit exhausted",
            )
        except TimeoutError:
            return await self._finish(
                assignment,
                tools,
                "timed_out",
                started_at,
                turns,
                tool_calls,
                usage,
                error_category="timeout",
            )
        except asyncio.CancelledError:
            await self._record(
                "repair_failed",
                assignment,
                status="cancelled",
                turns=turns,
                tool_calls=tool_calls,
                usage=usage.to_json(),
                error_category="cancelled",
            )
            raise
        except Exception as exc:  # noqa: BLE001 - repair never fails the developer task
            return await self._finish(
                assignment,
                tools,
                "failed",
                started_at,
                turns,
                tool_calls,
                usage,
                error_category=type(exc).__name__,
                message="repair provider or tool call failed",
            )

    async def _finish(
        self,
        assignment: RepairAssignment,
        tools: RepairToolset,
        status: RepairStatus,
        started_at: str,
        turns: int,
        tool_calls: int,
        usage: RepairUsage,
        *,
        error_category: str | None = None,
        message: str | None = None,
    ) -> RepairResult:
        result = RepairResult(
            task_session_id=assignment.task_session_id,
            repair_session_id=assignment.repair_session_id,
            notice_id=assignment.notice_id,
            status=status,
            started_at=started_at,
            ended_at=_now(),
            model=self._config.repair_model or "",
            provider=_provider(self._config.repair_model),
            episode_id=assignment.episode_id,
            notice_ids=assignment.notice_ids,
            turns=turns,
            tool_calls=tool_calls,
            candidate_rule_ids=tools.candidate_ids,
            existing_rule_id=tools.existing_rule_id,
            recommended_scope=assignment.recommended_scope,
            selected_scope=tools.selected_scope,
            scope_rationale=tools.scope_rationale,
            considered_rule_ids=tools.considered_rule_ids,
            resolution_kind=tools.resolution or status,
            candidate_suppression_reason=tools.suppression_reason,
            usage=usage,
            error_category=error_category,
            message=message,
            tracing_enabled=self._config.trace_repair_agent,
        )
        event = {
            "candidate_added": "repair_completed",
            "duplicate": "repair_duplicate",
            "already_covered": "repair_already_covered",
            "no_proposal": "repair_no_proposal",
            "unactionable": "repair_unactionable",
            "timed_out": "repair_timed_out",
            "failed": "repair_failed",
            "cancelled": "repair_failed",
        }[status]
        await asyncio.to_thread(self._journal.record, {"type": event, **result.to_json()})
        return result

    async def _record(
        self, event_type: str, assignment: RepairAssignment, **fields: Any
    ) -> None:
        await asyncio.to_thread(
            self._journal.record,
            {
                "type": event_type,
                "task_session_id": assignment.task_session_id,
                "repair_session_id": assignment.repair_session_id,
                "repair_episode_id": assignment.episode_id,
                "notice_id": assignment.notice_id,
                "notice_ids": list(assignment.notice_ids),
                "recommended_scope": assignment.recommended_scope,
                **fields,
            },
        )

    async def _call_tool(
        self,
        repair_trace: Any,
        tools: RepairToolset,
        name: str,
        args: dict[str, Any],
        *,
        turn: int,
        tool_call_id: str,
    ) -> dict[str, Any]:
        """Dispatch one restricted tool as a child of the active tool round."""

        with repair_trace.span(
            name,
            kind="TOOL",
            metadata={
                "role": "repair",
                "turn": turn,
                "tool_call_id": tool_call_id,
            },
        ) as tool_span:
            tool_span.set_input(args)
            result = await tools.call(name, args)
            tool_span.set_output(_safe_tool_result(name, result))
            ok = result.get("ok") is True
            tool_span.set_metadata({"ok": ok})
            if not ok:
                tool_span.set_error("tool call returned ok=false")
            return result

    def _trace_context(self, assignment: RepairAssignment) -> Any:
        """Return an exporting or SDK-native no-op trace for one assignment."""

        import pandaprobe

        kwargs: dict[str, Any] = {
            "input": _trace_assignment_message(assignment, self._config),
            "session_id": assignment.repair_session_id,
            "tags": ["pandaprobe-harness", "repair"],
            "metadata": {
                "role": "repair",
                "task_session_id": assignment.task_session_id,
                "notice_id": assignment.notice_id,
                "repair_episode_id": assignment.episode_id,
                "notice_ids": list(assignment.notice_ids),
                "model": self._config.repair_model,
            },
        }
        client = pandaprobe.get_client()
        if self._config.trace_repair_agent and client is not None and client.enabled:
            return pandaprobe.start_trace("pandaprobe", **kwargs)

        # Keep an inactive trace in context so the wrapped LiteLLM call never
        # creates a standalone exporting trace while repair tracing is disabled.
        kwargs["metadata"] = {**kwargs["metadata"], "export": False}
        return pandaprobe.Client(enabled=False).trace("pandaprobe", **kwargs)


def _status(tools: RepairToolset) -> RepairStatus:
    if tools.resolution == "duplicate":
        return "duplicate"
    if tools.resolution == "already_covered":
        return "already_covered"
    if tools.resolution == "no_proposal":
        return "no_proposal"
    if tools.resolution == "unactionable":
        return "unactionable"
    return "candidate_added"


def _provider(model: str | None) -> str:
    return (model or "").partition("/")[0]


def _repair_progress(completed_tools: set[str], tools: RepairToolset) -> str | None:
    """Keep provider-neutral tool loops moving after required evidence is read."""

    if tools.candidate_ids:
        return (
            "Candidate creation succeeded. Do not inspect or search again. "
            "On the next round, call harness_notice_ack with the candidate rule ID."
        )
    evidence_ready = {
        "harness_notice_read",
        "harness_trace_inspect",
        "harness_rules_search",
    }.issubset(completed_tools)
    if evidence_ready:
        return (
            "Evidence review and the required existing-rule search are complete. "
            "Do not call another read, inspect, list, status, or search tool. On the "
            "next round, either add one candidate with harness_rule_add, resolve as "
            "duplicate/already_covered, or resolve as no_proposal/unactionable."
        )
    return None


def _trace_assignment(
    assignment: RepairAssignment, config: HarnessConfig
) -> dict[str, Any]:
    """Bounded non-secret assignment identity for trace and root-span input."""

    return {
        "task_session_id": assignment.task_session_id,
        "repair_session_id": assignment.repair_session_id,
        "notice_id": assignment.notice_id,
        "repair_episode_id": assignment.episode_id,
        "notice_ids": list(assignment.notice_ids),
        "turn_index": assignment.turn_index,
        "severity": assignment.notice.severity,
        "signatures": list(assignment.notice.signatures),
        "model": config.repair_model,
    }


def _trace_assignment_message(
    assignment: RepairAssignment, config: HarnessConfig
) -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    _trace_assignment(assignment, config), sort_keys=True
                ),
            }
        ]
    }


def _trace_result_message(result: RepairResult) -> dict[str, list[dict[str, str]]]:
    summary = {
        "status": result.status,
        "notice_id": result.notice_id,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "candidate_rule_ids": list(result.candidate_rule_ids),
        "existing_rule_id": result.existing_rule_id,
        "recommended_scope": result.recommended_scope,
        "selected_scope": result.selected_scope,
        "resolution_kind": result.resolution_kind,
        "error_category": result.error_category,
    }
    return {
        "messages": [
            {
                "role": "assistant",
                "content": json.dumps(summary, sort_keys=True),
            }
        ]
    }


def _safe_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Trace tool outcomes without diagnostic payloads or complete rule bodies."""

    summary: dict[str, Any] = {"ok": result.get("ok") is True, "tool": name}
    for key in (
        "created", "suppressed", "suppression_reason", "recommended_resolution",
        "scope", "path", "episode_id", "notice_ids",
    ):
        if key in result:
            summary[key] = result[key]
    rule = result.get("rule")
    if isinstance(rule, dict):
        summary["rule_id"] = rule.get("id")
        summary["rule_status"] = rule.get("status")
    existing = result.get("existing_rule")
    if isinstance(existing, dict):
        summary["existing_rule_id"] = existing.get("id")
    rules = result.get("rules")
    if isinstance(rules, list):
        summary["rule_ids"] = [
            item.get("id") for item in rules if isinstance(item, dict) and item.get("id")
        ]
    if result.get("ok") is not True:
        summary["error"] = str(result.get("error") or "tool failed")[:240]
    return summary


def _completion_tools(tools: RepairToolset) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }
        for spec in tools.specs()
    ]
