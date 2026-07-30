"""Regression tests for the tau2 adapter's benchmark-specific failure modes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

if not os.environ.get("TAU2_DATA_DIR"):
    pytest.skip("TAU2_DATA_DIR is required for tau2 adapter tests", allow_module_level=True)

pytest.importorskip("tau2")

from tau2.data_model.message import UserMessage
from tau2.registry import registry

from pandabench.adapters.tau2_agent import (
    _EMPTY_TURN_CONTENT,
    PandaBenchTau2Agent,
)
from pandabench.providers.litellm_client import (
    ChatResult,
    MockClient,
    ToolCall,
    Usage,
)
from pandabench.providers.models import ResolvedModel, load_registry

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _result(*, content: str | None = None, tool_name: str | None = None) -> ChatResult:
    calls: list[ToolCall] = []
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_name is not None:
        calls = [ToolCall(id="call_h", name=tool_name, arguments={})]
        message["tool_calls"] = [
            {
                "id": "call_h",
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ]
    return ChatResult(
        assistant_message=message,
        tool_calls=calls,
        usage=Usage(10, 5, 0.0),
        finish_reason="tool_calls" if calls else "stop",
        resolved_model="mock/mock",
    )


class RecordingMockClient(MockClient):
    def __init__(self, *, scripted: list[ChatResult]) -> None:
        super().__init__(scripted=scripted)
        self.message_batches: list[list[dict[str, Any]]] = []

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
        self.message_batches.append([dict(message) for message in messages])
        return await super().chat(
            model=model,
            messages=messages,
            tools=tools,
            session_id=session_id,
            max_tokens=max_tokens,
            extra_params=extra_params,
        )


class StubWiring:
    def __init__(self) -> None:
        self.settled: list[int] = []
        self.dispatched: list[str] = []

    def system_preamble(self) -> str:
        return "harness preamble"

    def harness_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "harness_observe",
                    "description": "Observe the harness state.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def is_harness_tool(self, name: str) -> bool:
        return name.startswith("harness_")

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, bool]:
        self.dispatched.append(name)
        return {"ok": True}

    async def settle_turn(self, turn_index: int) -> None:
        self.settled.append(turn_index)


@pytest.fixture(scope="module")
def retail() -> tuple[list[Any], str]:
    environment = registry.get_env_constructor("retail")()
    return environment.get_tools(), environment.get_policy()


@pytest.fixture(scope="module")
def mock_model() -> ResolvedModel:
    return load_registry(CONFIGS / "models.yaml").resolve("mock")


def _agent(
    retail: tuple[list[Any], str],
    mock_model: ResolvedModel,
    client: MockClient,
    wiring: StubWiring | None = None,
) -> PandaBenchTau2Agent:
    tools, policy = retail
    return PandaBenchTau2Agent(
        tools,
        policy,
        client=client,
        model=mock_model,
        session_id="tau2-test",
        wiring=wiring,  # type: ignore[arg-type]
    )


def test_set_seed_does_not_require_tau2_llm(retail, mock_model):
    agent = _agent(retail, mock_model, MockClient())
    agent.set_seed(7)
    assert agent._seed == 7


def test_harness_substeps_follow_the_conversation(retail, mock_model):
    client = RecordingMockClient(
        scripted=[_result(tool_name="harness_observe"), _result(content="done")]
    )
    wiring = StubWiring()
    agent = _agent(retail, mock_model, client, wiring)

    assistant, _ = agent.generate_next_message(
        UserMessage(role="user", content="help with my order"), agent.get_init_state()
    )

    assert assistant.content == "done"
    assert wiring.dispatched == ["harness_observe"]
    second_call = client.message_batches[1]
    roles = [message["role"] for message in second_call]
    assert roles[-3:] == ["user", "assistant", "tool"]
    assert second_call[-1]["tool_call_id"] == "call_h"


def test_all_harness_turn_survives_tau2_validation(retail, mock_model):
    client = RecordingMockClient(scripted=[_result(tool_name="harness_observe")])
    agent = _agent(retail, mock_model, client, StubWiring())

    assistant, _ = agent.generate_next_message(
        UserMessage(role="user", content="check yourself"), agent.get_init_state()
    )

    assistant.validate()
    assert assistant.content == _EMPTY_TURN_CONTENT
    assert len(client.calls) == 6


def test_settle_once_per_completed_turn_with_increasing_indices(retail, mock_model):
    wiring = StubWiring()
    agent = _agent(retail, mock_model, MockClient(), wiring)
    state = agent.get_init_state()

    agent.generate_next_message(UserMessage(role="user", content="first"), state)
    agent.generate_next_message(UserMessage(role="user", content="second"), state)

    assert wiring.settled == [1, 2]
