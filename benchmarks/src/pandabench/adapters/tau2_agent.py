"""tau2-bench custom agent, routed through the pandabench LiteLLM wrapper.

tau2's orchestrator drives an agent incrementally: given one inbound message
(user/tool) + opaque state, ``generate_next_message`` returns the next
``AssistantMessage`` + state. We subclass ``LLMAgent`` and override that method to
call OUR wrapper (uniform usage/cost/tracing + model routing) instead of tau2's
own ``generate()``, keeping the user simulator on tau2's stock path so its model
stays fixed and independent across arms.

Harness wiring: the arm-B preamble is prepended to the domain policy, and the 14
harness tools are offered alongside the domain tools; a ``harness_*`` call is
handled in an internal sub-loop (tau2's orchestrator only knows domain tools) and
never surfaces to the orchestrator.

ISOLATION: tau2-bench pins ``litellm<1.82.7`` (conflicts with the pandabench core's
1.91), and its data is not shipped — so this runs in tau2's own venv with
``pandabench`` co-installed and ``TAU2_DATA_DIR`` set. Constructed by our runner
driving ``tau2.orchestrator.Orchestrator`` directly (tau2's ``run_task`` hardcodes
the ``LLMAgent(tools, domain_policy, llm, llm_args)`` constructor and cannot pass
harness config). See IMPLEMENTATION_NOTES.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

# tau2 is only importable inside its isolated venv; nothing in the core suite
# imports this module, so a top-level import is safe there.
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall

from ..agents.harness_wiring import HarnessWiring
from ..providers.litellm_client import ChatClient
from ..providers.models import ResolvedModel

logger = logging.getLogger("pandabench.tau2")

_MAX_HARNESS_SUBSTEPS = 6


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
        wiring: HarnessWiring | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._client = client
        self._model = model
        self._session_id = session_id
        self._wiring = wiring
        self._max_tokens = max_tokens
        self._domain_tool_schemas = [t.openai_schema for t in tools]

    def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
        # Mirror LLMAgent's state bookkeeping.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        assistant = self._decide(state)
        state.messages.append(assistant)
        return assistant, state

    # -- internals ------------------------------------------------------------

    def _system_prompt(self) -> str:
        policy = str(self.domain_policy)
        if self._wiring is not None:
            return self._wiring.system_preamble(policy[:400]) + "\n\n" + policy
        return policy

    def _tool_schemas(self) -> list[dict[str, Any]]:
        if self._wiring is not None:
            return [*self._domain_tool_schemas, *self._wiring.harness_tools()]
        return list(self._domain_tool_schemas)

    def _decide(self, state: Any) -> Any:
        """Call our wrapper; resolve any harness_* tool calls internally, then
        return the first assistant message that is a user reply or a DOMAIN action."""

        from tau2.utils.llm_utils import to_litellm_messages

        base = [{"role": "system", "content": self._system_prompt()}]
        convo = to_litellm_messages(state.system_messages + state.messages)
        # Drop any system messages already in convo; we prepend our own.
        convo = [m for m in convo if m.get("role") != "system"]

        for _ in range(_MAX_HARNESS_SUBSTEPS):
            result = asyncio.run(
                self._client.chat(
                    model=self._model, messages=base + convo,
                    tools=self._tool_schemas(), session_id=self._session_id,
                    max_tokens=self._max_tokens,
                )
            )
            harness_calls = [
                tc for tc in result.tool_calls
                if self._wiring is not None and self._wiring.is_harness_tool(tc.name)
            ]
            if not harness_calls:
                return _to_tau2_assistant(result)
            # Execute harness tools in-band and let the model continue.
            base.append(result.assistant_message)
            for tc in harness_calls:
                out = asyncio.run(self._wiring.dispatch(tc.name, tc.arguments))  # type: ignore[union-attr]
                base.append({"role": "tool", "tool_call_id": tc.id, "content": str(out)})
        # Ran out of sub-steps: return whatever the last call produced.
        return _to_tau2_assistant(result)


def _to_tau2_assistant(result: Any) -> Any:
    """Convert our ChatResult into a tau2 AssistantMessage (domain tools only)."""

    tool_calls = [
        ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments, requestor="assistant")
        for tc in result.tool_calls
        if not tc.name.startswith("harness_")
    ] or None
    content = result.assistant_message.get("content")
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
