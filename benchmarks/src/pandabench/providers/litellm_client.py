"""The single LiteLLM provider wrapper — the only place LiteLLM is touched.

One async ``chat`` method: OpenAI-format messages + tool schemas in; a normalized
assistant message, parsed tool calls, and usage/cost out. Every model call in
every benchmark, both arms, the tau2 user simulator, and every ReplayFn goes
through here, so tool-calling semantics, usage accounting, retries, param
handling, and PandaProbe tracing stay identical everywhere.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import ResolvedModel
from .tracing import PandaTracer

logger = logging.getLogger("pandabench.provider")

__all__ = [
    "ChatClient",
    "ChatResult",
    "LiteLLMClient",
    "MockClient",
    "ProviderError",
    "ToolCall",
    "Usage",
]

# Sampler params we may set; each is forwarded only if the model's allowlist
# permits it (Claude 5 / GPT-5 400 on temperature/top_p/top_k).
_SAMPLER_PARAMS = ("temperature", "top_p", "top_k")


class ProviderError(RuntimeError):
    """A model call failed after exhausting retries — the trial is recorded as
    ``error`` rather than crashing the run."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A parsed tool call. ``arguments`` is always a dict (JSON-parsed), never a
    string — the loop must never string-match model output."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True, slots=True)
class ChatResult:
    assistant_message: dict[str, Any]
    """OpenAI-shape assistant message, ready to append to the transcript."""
    tool_calls: list[ToolCall]
    usage: Usage
    finish_reason: str
    resolved_model: str
    raw: Any = field(default=None, repr=False)
    """The underlying LiteLLM response (for raw/ dumps); None for the mock."""


class ChatClient(Protocol):
    """The single chat interface the agent loop depends on (mockable)."""

    async def chat(
        self,
        *,
        model: ResolvedModel,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        session_id: str | None = None,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> ChatResult: ...

    def flush(self) -> None:
        """Block until buffered traces are sent to the platform (no-op if untraced)."""
        ...


class LiteLLMClient:
    """Thin wrapper over ``litellm.acompletion`` with usage/cost + PandaProbe tracing."""

    def __init__(
        self,
        *,
        tracer: PandaTracer | None = None,
        num_retries: int = 2,
        timeout_s: float = 120.0,
        default_max_tokens: int = 4096,
        drop_params: bool = False,
    ) -> None:
        self._tracer = tracer or PandaTracer.disabled()
        self._num_retries = num_retries
        self._timeout_s = timeout_s
        self._default_max_tokens = default_max_tokens
        # We filter params by allowlist ourselves; drop_params is a belt-and-braces
        # backstop, off by default so we never silently drop something unexpected.
        self._drop_params = drop_params

    def _call_params(
        self, model: ResolvedModel, max_tokens: int | None, extra: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Build the allowlisted per-call params (drops disallowed samplers)."""

        params: dict[str, Any] = {}
        if "max_tokens" in model.param_allowlist:
            params["max_tokens"] = max_tokens or self._default_max_tokens
        for name in _SAMPLER_PARAMS:
            if name in model.param_allowlist and extra and name in extra:
                params[name] = extra[name]
        # Any explicitly-requested extra that is allowlisted passes through.
        for name, value in (extra or {}).items():
            if name in model.param_allowlist and name not in params:
                params[name] = value
        return params

    async def chat(
        self,
        *,
        model: ResolvedModel,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        session_id: str | None = None,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> ChatResult:
        import litellm

        call_params = self._call_params(model, max_tokens, extra_params)
        kwargs: dict[str, Any] = {
            "model": model.litellm_model,
            "messages": messages,
            "num_retries": self._num_retries,
            "timeout": self._timeout_s,
            **call_params,
        }
        if tools:
            kwargs["tools"] = list(tools)
        if self._drop_params:
            kwargs["drop_params"] = True

        # The native LiteLLM wrapper auto-produces the LLM span; binding the
        # session id is all we do here (usage/cost for our records come from the
        # parsed response below).
        with self._tracer.session(session_id):
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as exc:  # noqa: BLE001 - normalize any provider failure
                raise ProviderError(
                    f"{model.litellm_model} call failed after {self._num_retries} retries: {exc}"
                ) from exc
        return _parse_response(response, model)

    def flush(self) -> None:
        self._tracer.flush()


def _parse_response(response: Any, model: ResolvedModel) -> ChatResult:
    """Normalize a LiteLLM ModelResponse to a ChatResult."""

    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    raw_tool_calls = getattr(message, "tool_calls", None) or []

    tool_calls: list[ToolCall] = []
    tool_call_field: list[dict[str, Any]] = []
    for tc in raw_tool_calls:
        args_str = tc.function.arguments or "{}"
        try:
            parsed = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            logger.warning("un-parseable tool args for %s: %r", tc.function.name, args_str)
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=parsed))
        tool_call_field.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": args_str},
            }
        )

    assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_call_field:
        assistant_message["tool_calls"] = tool_call_field

    usage_obj = getattr(response, "usage", None)
    in_tok = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage_obj, "completion_tokens", 0) or 0)
    cost = _cost_of(response, model, in_tok, out_tok)

    return ChatResult(
        assistant_message=assistant_message,
        tool_calls=tool_calls,
        usage=Usage(in_tok, out_tok, cost),
        finish_reason=str(getattr(choice, "finish_reason", "") or ""),
        resolved_model=model.litellm_model,
        raw=response,
    )


def _cost_of(response: Any, model: ResolvedModel, in_tok: int, out_tok: int) -> float:
    import litellm

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:  # noqa: BLE001 - many providers lack a price; fall back
        cost = None
    if not cost:
        cost = model.cost_from_usage(in_tok, out_tok)
    return float(cost or 0.0)


class MockClient:
    """Deterministic, network-free client for ``--dry-run`` and unit tests.

    Default behavior: return a single final assistant message (no tool calls),
    so any loop terminates after one turn while exercising the full record /
    harness / report pipeline. Pass ``scripted`` for multi-turn/tool scenarios:
    a list of ``ChatResult`` returned in order (last one repeats).
    """

    def __init__(
        self,
        *,
        scripted: list[ChatResult] | None = None,
        final_text: str = "[mock] done",
        per_call_tokens: tuple[int, int] = (10, 5),
    ) -> None:
        self._scripted = list(scripted or [])
        self._final_text = final_text
        self._in, self._out = per_call_tokens
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        model: ResolvedModel,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        session_id: str | None = None,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> ChatResult:
        self.calls.append(
            {"model": model.key, "session_id": session_id, "n_messages": len(messages)}
        )
        if self._scripted:
            idx = min(len(self.calls) - 1, len(self._scripted) - 1)
            return self._scripted[idx]
        return ChatResult(
            assistant_message={"role": "assistant", "content": self._final_text},
            tool_calls=[],
            usage=Usage(self._in, self._out, 0.0),
            finish_reason="stop",
            resolved_model=model.litellm_model,
            raw=None,
        )

    def flush(self) -> None:
        pass
