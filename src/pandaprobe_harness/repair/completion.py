"""PandaProbe-instrumented LiteLLM transport for managed repair."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .models import RepairUsage

__all__ = [
    "NormalizedRepairMessage",
    "NormalizedToolCall",
    "PandaProbeLiteLLMCompletion",
    "RepairCompletion",
]


@dataclass(frozen=True, slots=True)
class NormalizedToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class NormalizedRepairMessage:
    content: str | None = None
    tool_calls: tuple[NormalizedToolCall, ...] = ()
    usage: RepairUsage = RepairUsage()


class RepairCompletion(Protocol):
    """Transport-only seam; orchestration remains package-owned."""

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None,
        timeout_s: float,
    ) -> NormalizedRepairMessage: ...


class PandaProbeLiteLLMCompletion:
    """Call normalized LiteLLM through PandaProbe's official wrapper."""

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None,
        timeout_s: float,
    ) -> NormalizedRepairMessage:
        from pandaprobe.wrappers.litellm import wrap_litellm

        litellm = wrap_litellm()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "tools": list(tools),
            "max_tokens": max_tokens,
            "timeout": timeout_s,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort is not None and _supports_param(
            litellm, model, "reasoning_effort"
        ):
            kwargs["reasoning_effort"] = reasoning_effort

        # The managed agent owns the assignment-wide trace. The official
        # wrapper finds that active context and contributes one nested LLM span.
        response = await litellm.acompletion(**kwargs)
        return _normalize_response(response)


def _supports_param(litellm: Any, model: str, name: str) -> bool:
    """Ask LiteLLM's model registry instead of branching on provider names."""

    resolver = getattr(litellm, "get_supported_openai_params", None)
    if not callable(resolver):
        return True
    try:
        return name in (resolver(model=model) or ())
    except Exception:
        return False


def _normalize_response(response: Any) -> NormalizedRepairMessage:
    """Read LiteLLM's provider-neutral ModelResponse contract."""

    choices = getattr(response, "choices", None) or []
    if not choices:
        return NormalizedRepairMessage(usage=_usage(response))
    message = getattr(choices[0], "message", None)
    if message is None:
        return NormalizedRepairMessage(usage=_usage(response))

    calls: list[NormalizedToolCall] = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        if function is None:
            continue
        calls.append(
            NormalizedToolCall(
                id=str(getattr(call, "id", "")),
                name=str(getattr(function, "name", "")),
                arguments=str(getattr(function, "arguments", "{}") or "{}"),
            )
        )
    content = getattr(message, "content", None)
    return NormalizedRepairMessage(
        content=str(content) if content is not None else None,
        tool_calls=tuple(calls),
        usage=_usage(response),
    )


def _usage(response: Any) -> RepairUsage:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    hidden = getattr(response, "_hidden_params", None)
    raw_cost = hidden.get("response_cost") if isinstance(hidden, Mapping) else None
    cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else None
    return RepairUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
        cost=cost,
    )
