"""The one shared tool-calling agent loop used by every benchmark.

The two study arms run *identical* code here; they differ only by whether a
:class:`HarnessWiring` is passed. Arm A (``wiring=None``) is a plain
call-model / run-tools / repeat loop. Arm B additionally prepends the harness
preamble each turn and exposes only read-only learned-rule tools alongside the
benchmark's tools. The runner owns final settlement and phase validation; the
loop settles continuing turns so package-owned repair can update next-turn
on-demand discovery in-session.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..providers.litellm_client import ChatClient, ProviderError, ToolCall, Usage
from ..providers.models import ResolvedModel
from .harness_wiring import AgentWiring

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
    wiring: AgentWiring | None = None,
    max_tokens: int | None = None,
    max_input_chars: int | None = None,
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
            system = wiring.system_preamble() + "\n\n" + system_prompt
            call_tools: list[dict[str, Any]] = [*tools, *wiring.harness_tools()]
        else:
            system = system_prompt
            call_tools = list(tools)

        call_messages = _bounded_call_messages(
            system=system,
            convo=convo,
            preserved_prefix=len(initial_messages),
            max_chars=max_input_chars,
        )

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

        # The per-turn barrier: block until the harness has evaluated this turn and
        # completed package-owned repair, so the next iteration's read-only rule
        # tools already reflect any provisional guidance. This is what makes
        # healing take effect *within* a session rather than after it.
        #
        # Only when another turn will actually follow. At the cap the next
        # iteration returns immediately, so this turn is the trial's last and the
        # runner settles it — settling here too would fire a second evaluation for
        # the same turn, which finds no new traces and so reports none of the tier
        # scores the runner then records as telemetry.
        if wiring is not None and wiring.settles_turns and turns < max_turns:
            await wiring.settle_turn(turns)


async def _dispatch(
    tool_call: ToolCall, tool_executor: ToolExecutor, wiring: AgentWiring | None
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


def _bounded_call_messages(
    *,
    system: str,
    convo: Sequence[dict[str, Any]],
    preserved_prefix: int,
    max_chars: int | None,
) -> list[dict[str, Any]]:
    """Keep a bounded, protocol-valid suffix for providers with finite context.

    The system prompt and original benchmark request are always retained. Older
    assistant turns are removed as whole blocks together with their tool results,
    so a Bedrock Converse request can never begin with an orphaned ``role=tool``
    message. ``convo`` itself stays complete for tracing and returned diagnostics;
    only the next provider request is compacted.

    The bound is deliberately in serialized characters rather than guessed model
    tokens. Terminal output includes code, logs, and binary-ish text whose token
    ratio varies substantially; a conservative character ceiling is predictable
    and provider-independent.
    """

    system_message: dict[str, Any] = {"role": "system", "content": system}
    if max_chars is None:
        return [system_message, *convo]
    if max_chars <= 0:
        raise ValueError("max_input_chars must be positive")

    prefix = [dict(message) for message in convo[:preserved_prefix]]
    blocks = _conversation_blocks(convo[preserved_prefix:])
    base_size = _messages_size([system_message, *prefix])
    remaining = max(0, max_chars - base_size)
    kept_reversed: list[list[dict[str, Any]]] = []
    omitted = 0
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        block_size = _messages_size(block)
        if block_size > remaining:
            omitted = index + 1
            break
        kept_reversed.append(block)
        remaining -= block_size

    if omitted:
        system_message["content"] = (
            f"{system}\n\n[Context notice: {omitted} older terminal turn(s) were "
            "omitted to stay within the model context window. Re-inspect any "
            "state you still need.]"
        )
        logger.info("compacted %d older conversation block(s)", omitted)

    kept = [message for block in reversed(kept_reversed) for message in block]
    return [system_message, *prefix, *kept]


def _conversation_blocks(
    messages: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group each assistant message with all tool results that answer it."""

    blocks: list[list[dict[str, Any]]] = []
    for original in messages:
        message = dict(original)
        if message.get("role") == "tool" and blocks:
            blocks[-1].append(message)
        else:
            blocks.append([message])
    return blocks


def _messages_size(messages: Sequence[dict[str, Any]]) -> int:
    return sum(len(json.dumps(message, ensure_ascii=False, default=str)) for message in messages)
