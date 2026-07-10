"""The one shared tool-calling agent loop used by every benchmark.

The two study arms run *identical* code here; they differ only by whether a
:class:`HarnessWiring` is passed. Arm A (``wiring=None``) is a plain
call-model / run-tools / repeat loop. Arm B additionally prepends the harness
preamble each turn and exposes the harness's self-diagnostic tools alongside
the benchmark's tools. Session lifecycle (``on_turn_end`` / ``refresh`` /
``drain_validation``) is owned by the *runner*, not this loop, so the loop stays
a pure agent stepper (see ``runners/base.py``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..providers.litellm_client import ChatClient, ProviderError, ToolCall, Usage
from ..providers.models import ResolvedModel
from .harness_wiring import HarnessWiring

logger = logging.getLogger("pandabench.loop")

__all__ = ["LoopResult", "ToolExecutor", "run_agent_loop"]

# A benchmark's tool dispatcher: (tool_name, parsed_args) -> result payload.
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class LoopResult:
    final_message: dict[str, Any] | None
    messages: list[dict[str, Any]]
    turns: int
    usage: Usage
    stopped_reason: str  # "final" | "max_turns" | "error"
    error: str | None = None
    tool_call_count: int = 0
    harness_tool_calls: list[str] = field(default_factory=list)


async def run_agent_loop(
    *,
    client: ChatClient,
    model: ResolvedModel,
    session_id: str,
    system_prompt: str,
    tools: Sequence[dict[str, Any]],
    tool_executor: ToolExecutor,
    initial_messages: Sequence[dict[str, Any]],
    max_turns: int,
    wiring: HarnessWiring | None = None,
    task_hint: str = "",
    max_tokens: int | None = None,
) -> LoopResult:
    """Drive one task-trial to completion (final answer, cap, or error).

    ``tool_executor`` handles the benchmark's own tools; ``harness_*`` calls are
    routed to the harness in arm B. Never raises on a model failure — returns a
    partial result with ``stopped_reason="error"``.
    """

    convo: list[dict[str, Any]] = [dict(m) for m in initial_messages]
    total = Usage()
    turns = 0
    tool_calls_made = 0
    harness_calls: list[str] = []
    final_message: dict[str, Any] | None = None

    while True:
        if turns >= max_turns:
            return LoopResult(
                final_message, convo, turns, total, "max_turns",
                tool_call_count=tool_calls_made, harness_tool_calls=harness_calls,
            )

        if wiring is not None:
            system = wiring.system_preamble(task_hint) + "\n\n" + system_prompt
            call_tools: list[dict[str, Any]] = [*tools, *wiring.harness_tools()]
        else:
            system = system_prompt
            call_tools = list(tools)

        call_messages = [{"role": "system", "content": system}, *convo]

        try:
            result = await client.chat(
                model=model,
                messages=call_messages,
                tools=call_tools or None,
                session_id=session_id,
                max_tokens=max_tokens,
            )
        except ProviderError as exc:
            logger.warning("session %s: model error on turn %d: %s", session_id, turns + 1, exc)
            return LoopResult(
                final_message, convo, turns, total, "error", str(exc),
                tool_call_count=tool_calls_made, harness_tool_calls=harness_calls,
            )

        turns += 1
        total = total + result.usage
        convo.append(result.assistant_message)
        final_message = result.assistant_message

        if not result.tool_calls:
            return LoopResult(
                final_message, convo, turns, total, "final",
                tool_call_count=tool_calls_made, harness_tool_calls=harness_calls,
            )

        for tool_call in result.tool_calls:
            tool_calls_made += 1
            if wiring is not None and wiring.is_harness_tool(tool_call.name):
                harness_calls.append(tool_call.name)
            output = await _dispatch(tool_call, tool_executor, wiring)
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": _as_tool_content(output),
                }
            )


async def _dispatch(
    tool_call: ToolCall, tool_executor: ToolExecutor, wiring: HarnessWiring | None
) -> Any:
    """Route one tool call to the harness (``harness_*``) or the benchmark."""

    try:
        if wiring is not None and wiring.is_harness_tool(tool_call.name):
            return await wiring.dispatch(tool_call.name, tool_call.arguments)
        return await tool_executor(tool_call.name, tool_call.arguments)
    except Exception as exc:  # noqa: BLE001 - a bad tool call must not kill the trial
        logger.warning("tool %s failed: %s", tool_call.name, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


def _as_tool_content(output: Any) -> str:
    """Serialize a tool result to string content for the transcript."""

    if isinstance(output, str):
        return output
    try:
        return json.dumps(output)
    except (TypeError, ValueError):
        return str(output)
