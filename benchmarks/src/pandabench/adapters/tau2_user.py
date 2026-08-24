"""tau2 user simulator routed through PandaBench's provider wrapper.

tau2's stock :class:`UserSimulator` calls LiteLLM directly.  That bypasses the
resolved study model's backend, parameter allowlist, required defaults, retry
policy, and cost fallback.  This adapter preserves tau2's prompts, state
machine, role flipping, and user-tool behavior while delegating only the model
call to :class:`~pandabench.providers.litellm_client.ChatClient`.

The user call uses a session distinct from the task agent.  In the harness arm
this keeps simulated-user spans out of the trajectory the harness evaluates.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any

from tau2.data_model.message import MultiToolMessage, ToolCall, UserMessage
from tau2.user.user_simulator import UserSimulator

from ..providers.litellm_client import ChatClient, ChatResult, ProviderError, Usage
from ..providers.models import ResolvedModel

logger = logging.getLogger("pandabench.tau2.user")

_VISIBLE_RESPONSE_REQUIREMENT = (
    "On every turn, after any private reasoning, emit one non-empty "
    "customer-facing message or a native tool call. Private reasoning alone "
    "is not a response."
)
_EMPTY_RESPONSE_RETRY = (
    "The previous generation contained only private reasoning and no visible "
    "customer turn. Respond now with one non-empty customer-facing message or "
    "a native tool call."
)
_MAX_EMPTY_RETRIES = 1


class PandaBenchTau2User(UserSimulator):  # type: ignore[misc]
    """A stock-compatible tau2 user using the task model and backend."""

    def __init__(
        self,
        tools: Any,
        instructions: Any,
        *,
        client: ChatClient,
        model: ResolvedModel,
        session_id: str,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        # Keep tau2's conventional user temperature.  ChatClient applies it only
        # when the resolved model allowlists temperature; unsupported providers
        # never receive it.  set_seed() may similarly add a seed to llm_args.
        super().__init__(
            tools=tools,
            instructions=instructions,
            llm=model.litellm_model,
            llm_args={"temperature": 1.0},
        )
        self._client = client
        self._model = model
        self._session_id = f"{session_id}:tau2-user"
        self._loop = loop

    @property
    def system_prompt(self) -> str:
        """Retain tau2's prompt while requiring a protocol-valid user turn."""

        return f"{super().system_prompt}\n\n{_VISIBLE_RESPONSE_REQUIREMENT}"

    def _generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
        """Mirror tau2's state updates, replacing only its direct LiteLLM call."""

        from tau2.utils.llm_utils import to_litellm_messages

        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = to_litellm_messages(state.system_messages + state.flip_roles())
        tools = [tool.openai_schema for tool in self.tools] if self.tools else None
        total_usage = Usage()
        result: ChatResult | None = None
        for attempt in range(_MAX_EMPTY_RETRIES + 1):
            result = self._await(self._client.chat(
                model=self._model,
                messages=messages,
                tools=tools,
                session_id=self._session_id,
                extra_params=dict(self.llm_args),
            ))
            total_usage = total_usage + result.usage
            if _has_visible_turn(result):
                break
            logger.warning(
                "tau2 user model %s returned no content/tool call "
                "(finish_reason=%s, attempt=%d/%d)",
                self._model.key,
                result.finish_reason,
                attempt + 1,
                _MAX_EMPTY_RETRIES + 1,
            )
            messages = _with_empty_response_retry(messages)
        if result is None or not _has_visible_turn(result):
            raise ProviderError(
                f"{self._model.litellm_model} tau2 user returned no visible content "
                f"or tool call after {_MAX_EMPTY_RETRIES + 1} attempts"
            )
        result = replace(result, usage=total_usage)
        user_message = _to_tau2_user(result)
        state.messages.append(user_message)
        return user_message, state

    def _await(self, coro: Any) -> Any:
        """Submit provider calls to the runner loop from tau2's worker thread."""

        if self._loop is None:
            return asyncio.run(coro)
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


def _has_visible_turn(result: ChatResult) -> bool:
    """Whether tau2 can validate and route this simulated-user response."""

    content = result.assistant_message.get("content")
    return bool(result.tool_calls or (isinstance(content, str) and content.strip()))


def _with_empty_response_retry(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reinforce the system instruction without altering domain dialogue."""

    retried = [dict(message) for message in messages]
    for message in retried:
        if message.get("role") == "system":
            content = str(message.get("content") or "")
            message["content"] = f"{content}\n\n{_EMPTY_RESPONSE_RETRY}"
            break
    return retried


def _to_tau2_user(result: ChatResult) -> UserMessage:
    """Convert the normalized provider result into tau2's user message shape."""

    calls = [
        ToolCall(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            requestor="user",
        )
        for call in result.tool_calls
    ]
    return UserMessage(
        role="user",
        content=result.assistant_message.get("content"),
        tool_calls=calls or None,
        cost=result.usage.cost_usd,
        usage={
            "prompt_tokens": result.usage.input_tokens,
            "completion_tokens": result.usage.output_tokens,
        },
        raw_data=_raw_dict(result.raw),
    )


def _raw_dict(raw: Any) -> dict[str, Any] | None:
    """Best-effort raw response serialization matching tau2's message schema."""

    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    to_dict = getattr(raw, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value
    return None
