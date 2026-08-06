"""Regression tests for the tau2 adapter's benchmark-specific failure modes."""

from __future__ import annotations

import json
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
from pandabench.agents.frozen_wiring import FrozenEvalWiring
from pandabench.frozen_rules import FrozenRulesSnapshot
from pandabench.providers.litellm_client import (
    ChatResult,
    MockClient,
    ToolCall,
    Usage,
)
from pandabench.providers.models import ResolvedModel, load_registry
from pandabench.runners.tau2 import Tau2Runner

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _result(
    *,
    content: str | None = None,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
    call_id: str = "call_h",
) -> ChatResult:
    calls: list[ToolCall] = []
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_name is not None:
        arguments = tool_args or {}
        calls = [ToolCall(id=call_id, name=tool_name, arguments=arguments)]
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(arguments)},
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
        self.tool_batches: list[list[str]] = []
        self.flushes = 0

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
        self.tool_batches.append(
            [
                str((schema.get("function") or {}).get("name", ""))
                for schema in tools or []
            ]
        )
        return await super().chat(
            model=model,
            messages=messages,
            tools=tools,
            session_id=session_id,
            max_tokens=max_tokens,
            extra_params=extra_params,
        )

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


class StubWiring:
    def __init__(self) -> None:
        self.settled: list[int] = []
        self.dispatched: list[str] = []

    @property
    def settles_turns(self) -> bool:
        return True

    def system_preamble(self) -> str:
        return "harness preamble"

    def harness_tools(self) -> list[dict[str, Any]]:
        names = [
            "harness_rules_list",
            "harness_rules_read",
            "harness_rules_search",
            "harness_rule_status",
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Test tool {name}.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in names
        ]

    def is_harness_tool(self, name: str) -> bool:
        return name.startswith("harness_")

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
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


@pytest.mark.parametrize(
    ("domain", "task_count"),
    [("airline", 50), ("retail", 114), ("telecom", 114)],
)
async def test_runner_selects_tasks_and_environment_for_every_domain(
    domain: str,
    task_count: int,
    mock_model: ResolvedModel,
):
    runner = Tau2Runner()
    task_ids = runner.list_tasks(domain)

    orchestrator, task = runner._build(
        task_id=task_ids[0],
        session_id=f"tau2-{domain}-test",
        model=mock_model,
        client=MockClient(),
        max_turns=10,
        wiring=None,
        preamble=None,
    )

    assert len(task_ids) == task_count
    assert orchestrator.domain == domain
    assert orchestrator.user.llm_args["temperature"] == 1.0
    assert str(task.id) == task_ids[0]


def test_runner_rejects_non_benchmark_tau2_domain():
    with pytest.raises(ValueError, match="unsupported tau2 domain"):
        Tau2Runner(domain="telecom-workflow")


def test_set_seed_does_not_require_tau2_llm(retail, mock_model):
    agent = _agent(retail, mock_model, MockClient())
    agent.set_seed(7)
    assert agent._seed == 7


def test_tau2_task_agent_never_runs_workspace_repair(retail, mock_model):
    client = RecordingMockClient(scripted=[_result(content="done")])
    wiring = StubWiring()
    agent = _agent(retail, mock_model, client, wiring)

    assistant, _ = agent.generate_next_message(
        UserMessage(role="user", content="help with my order"), agent.get_init_state()
    )

    assert assistant.content == "done"
    assert wiring.dispatched == []
    assert wiring.settled == [1]
    assert [call["session_id"] for call in client.calls] == ["tau2-test"]
    assert all(not name.startswith("harness_") for name in client.tool_batches[0])
    assert "PandaProbe owns workspace repair" in client.message_batches[0][0]["content"]
    assert assistant.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert client.flushes == 0


def test_tau2_domain_turn_settles_without_model_driven_notice_phase(retail, mock_model):
    client = RecordingMockClient(
        scripted=[_result(tool_name="get_user_details", call_id="call_domain")]
    )
    wiring = StubWiring()
    agent = _agent(retail, mock_model, client, wiring)

    assistant, _ = agent.generate_next_message(
        UserMessage(role="user", content="help with my flight"), agent.get_init_state()
    )

    assert [call.id for call in assistant.tool_calls or []] == ["call_domain"]
    assert wiring.dispatched == []
    assert wiring.settled == [1]
    assert [call["session_id"] for call in client.calls] == ["tau2-test"]
    assert assistant.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert client.flushes == 0


def test_unexpected_harness_call_in_domain_phase_survives_tau2_validation(
    retail, mock_model
):
    client = RecordingMockClient(scripted=[_result(tool_name="harness_mailbox_list")])
    agent = _agent(retail, mock_model, client, StubWiring())

    assistant, _ = agent.generate_next_message(
        UserMessage(role="user", content="check yourself"), agent.get_init_state()
    )

    assistant.validate()
    assert assistant.content == _EMPTY_TURN_CONTENT
    assert len(client.calls) == 1
    assert all(not name.startswith("harness_") for name in client.tool_batches[0])


def test_settle_once_per_completed_turn_with_increasing_indices(retail, mock_model):
    wiring = StubWiring()
    agent = _agent(retail, mock_model, MockClient(), wiring)
    state = agent.get_init_state()

    agent.generate_next_message(UserMessage(role="user", content="first"), state)
    agent.generate_next_message(UserMessage(role="user", content="second"), state)

    assert wiring.settled == [1, 2]


def test_frozen_eval_injects_learning_rules_without_repair_or_settlement(
    retail, mock_model
):
    class NoSettleFrozenWiring(FrozenEvalWiring):
        async def settle_turn(self, turn_index: int) -> None:
            raise AssertionError(f"frozen tau2 eval settled turn {turn_index}")

    snapshot = FrozenRulesSnapshot.create(
        [{
            "id": "r-tau2",
            "created_at": "2026-08-05T00:00:00+00:00",
            "rule": "Verify the reservation owner before cancellation.",
            "rationale": "Learned from a training failure.",
            "source_notice_id": "n-tau2",
            "metric": "tool_correctness",
            "status": "active",
            "tags": ["reservation", "cancel"],
            "trial": None,
            "scope": "global",
        }],
        created_at="2026-08-05T01:00:00+00:00",
    )
    wiring = NoSettleFrozenWiring(snapshot)
    client = RecordingMockClient(scripted=[_result(content="I can help with that.")])
    agent = _agent(retail, mock_model, client, wiring)  # type: ignore[arg-type]

    assistant, _ = agent.generate_next_message(
        UserMessage(role="user", content="cancel my reservation"), agent.get_init_state()
    )

    assert assistant.content == "I can help with that."
    assert len(client.calls) == 1  # no task-agent notice-repair model phase
    assert all(not name.startswith("harness_") for name in client.tool_batches[0])
    assert "Verify the reservation owner" in client.message_batches[0][0]["content"]
