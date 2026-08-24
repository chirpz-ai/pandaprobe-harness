"""tau2-bench custom agent, routed through the pandabench LiteLLM wrapper.

tau2's orchestrator drives an agent incrementally: given one inbound message
(user/tool) + opaque state, ``generate_next_message`` returns the next
``AssistantMessage`` + state. We subclass ``LLMAgent`` and override that method to
call OUR wrapper (uniform usage/cost/tracing + model routing) instead of tau2's
own ``generate()``. The paired user adapter uses that same resolved model and
backend in a separate trace session.

Harness wiring keeps the developer-owned tau2 agent on domain work. The task
model sees a stable capability note plus optional read-only rule tools alongside
tau2 domain tools. After
each completed turn this adapter drives the per-turn barrier; evaluation and
notice handling then invoke the package-owned managed repair agent, whose calls
use a distinct repair session and never enter tau2's transcript.

THREADING: ``generate_next_message`` is synchronous and the runner drives
``Orchestrator.run()`` in a worker thread, but the Harness belongs to the runner's
event loop — so every coroutine (chat, harness dispatch, the barrier) is submitted
back to that loop via :meth:`PandaBenchTau2Agent._await`.

tau2's own ``run_task`` hardcodes the ``LLMAgent(tools, domain_policy, llm,
llm_args)`` constructor and cannot pass harness config, which is why our runner
drives ``tau2.orchestrator.Orchestrator`` directly. tau2's data is not shipped, so
``TAU2_DATA_DIR`` must point at a clone's ``data/`` **before the first tau2
import**. See IMPLEMENTATION_NOTES.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

# tau2 is an optional dependency; nothing in the core suite imports this module,
# so a top-level import is safe when the ``tau2`` extra is installed.
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall

from ..agents.harness_wiring import AgentWiring
from ..providers.litellm_client import ChatClient
from ..providers.models import ResolvedModel


class PandaBenchTau2Agent(LLMAgent):  # type: ignore[misc]
    """LLMAgent whose next-message generation routes through our wrapper."""

    def __init__(
        self,
        tools: Any,
        domain_policy: str,
        *,
        client: ChatClient,
        model: ResolvedModel,
        session_id: str,
        wiring: AgentWiring | None = None,
        max_tokens: int | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._client = client
        self._model = model
        self._session_id = session_id
        self._wiring = wiring
        self._max_tokens = max_tokens
        self._domain_tool_schemas = [t.openai_schema for t in tools]
        # tau2's orchestrator is synchronous, so the runner drives it in a worker
        # thread; every coroutine we need belongs to the loop that owns the
        # Harness. See _await.
        self._loop = loop
        self._turns = 0
        self._seed: int | None = None

    def set_seed(self, seed: int) -> None:
        """Record tau2's per-trial seed.

        The orchestrator calls this whenever a seed is given, and the inherited
        implementation raises ``ValueError("LLM is not set")`` because we
        deliberately never set ``self.llm`` — we route generation through our own
        client instead. Sampling determinism therefore comes from whatever the
        provider honours (GPT-5.x accepts no sampler params at all).
        """

        self._seed = seed

    def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
        # Mirror LLMAgent's state bookkeeping.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        assistant = _to_tau2_assistant(self._decide_domain(state))
        state.messages.append(assistant)

        # The per-turn barrier. tau2 has no shared agent loop to host it, so the
        # agent settles its own turns: without this the harness would score the
        # session once per trial and the trajectory gate would never accumulate a
        # series — the v1 inertness this release exists to fix.
        self._turns += 1
        if self._wiring is not None and self._wiring.settles_turns:
            self._await(self._wiring.settle_turn(self._turns))
        return assistant, state

    # -- internals ------------------------------------------------------------

    def _await(self, coro: Any) -> Any:
        """Run ``coro`` on the loop that owns the Harness, from this thread.

        ``generate_next_message`` is synchronous and runs in a worker thread, but
        the Harness (its task bookkeeping, stores and locks) belongs to the
        runner's loop — driving its coroutines on a fresh per-call loop would
        corrupt that state. With no owning loop (unit tests) fall back to
        ``asyncio.run``.
        """

        if self._loop is None:
            return asyncio.run(coro)
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _system_prompt(self) -> str:
        policy = str(self.domain_policy)
        if self._wiring is not None:
            return "\n\n".join(
                (
                    self._wiring.system_preamble(),
                    "PandaProbe owns workspace repair. Focus only on the tau2 "
                    "domain task. Learned guidance is optional and read-only.",
                    policy,
                )
            )
        return policy

    def _decide_domain(self, state: Any) -> Any:
        """Produce exactly one user-facing/domain action on the task session."""

        from tau2.utils.llm_utils import to_litellm_messages

        base = [{"role": "system", "content": self._system_prompt()}]
        convo = to_litellm_messages(state.system_messages + state.messages)
        # Drop any system messages already in convo; we prepend our own.
        convo = [m for m in convo if m.get("role") != "system"]
        return self._await(self._chat_with_rule_tools(base + convo))

    async def _chat_with_rule_tools(self, messages: list[dict[str, Any]]) -> Any:
        """Let tau2 agents choose read-only rule tools before a domain action."""

        tools = list(self._domain_tool_schemas)
        if self._wiring is not None:
            tools.extend(self._wiring.harness_tools())
        for _ in range(8):
            result = await self._client.chat(
                model=self._model,
                messages=messages,
                tools=tools,
                session_id=self._session_id,
                max_tokens=self._max_tokens,
            )
            harness_calls = [
                call
                for call in result.tool_calls
                if self._wiring is not None and self._wiring.is_harness_tool(call.name)
            ]
            if not harness_calls:
                return result
            messages.append(result.assistant_message)
            for call in result.tool_calls:
                if self._wiring is not None and self._wiring.is_harness_tool(call.name):
                    payload = await self._wiring.dispatch(call.name, call.arguments)
                else:
                    payload = {
                        "ok": False,
                        "error": "choose rule inspection or a domain action, not both",
                    }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(payload, sort_keys=True, default=str),
                    }
                )
        return result


#: Emitted when a turn produced neither text nor a domain tool call — tau2's
#: orchestrator calls AssistantMessage.validate(), which raises on an empty
#: message and would abort the whole run rather than the single trial.
_EMPTY_TURN_CONTENT = "(no action this turn)"


def _to_tau2_assistant(result: Any) -> Any:
    """Convert our ChatResult into a tau2 AssistantMessage (domain tools only)."""

    tool_calls = [
        ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments, requestor="assistant")
        for tc in result.tool_calls
        if not tc.name.startswith("harness_")
    ] or None
    content = result.assistant_message.get("content")
    if tool_calls is None and not content:
        content = _EMPTY_TURN_CONTENT
    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=result.usage.cost_usd,
        usage={
            "prompt_tokens": result.usage.input_tokens,
            "completion_tokens": result.usage.output_tokens,
        },
    )


__all__ = ["PandaBenchTau2Agent"]
