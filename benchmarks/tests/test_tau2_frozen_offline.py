"""Frozen tau2 adapter coverage that does not require the benchmark data tree."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
pytest.importorskip("tau2")

from tau2.data_model.message import UserMessage

from pandabench.adapters.tau2_agent import PandaBenchTau2Agent
from pandabench.agents.frozen_wiring import FrozenEvalWiring
from pandabench.frozen_rules import FrozenRulesSnapshot
from pandabench.providers.litellm_client import ChatResult, MockClient
from pandabench.providers.models import ResolvedModel, load_registry
from pandabench.runners.tau2 import Tau2Runner

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class RecordingClient(MockClient):
    def __init__(self) -> None:
        super().__init__(final_text="I can help with that.")
        self.message_batches: list[list[dict[str, Any]]] = []
        self.tool_batches: list[list[str]] = []

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
        self.tool_batches.append([
            str((tool.get("function") or {}).get("name", "")) for tool in tools or []
        ])
        return await super().chat(
            model=model, messages=messages, tools=tools, session_id=session_id,
            max_tokens=max_tokens, extra_params=extra_params,
        )


class NoSettleFrozenWiring(FrozenEvalWiring):
    async def settle_turn(self, turn_index: int) -> None:
        raise AssertionError(f"frozen tau2 eval settled turn {turn_index}")


def test_tau2_frozen_context_skips_repair_and_settlement() -> None:
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
    client = RecordingClient()
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    agent = PandaBenchTau2Agent(
        [],
        "Follow the domain policy.",
        client=client,
        model=model,
        session_id="tau2-frozen",
        wiring=wiring,
    )

    assistant, _ = agent.generate_next_message(
        UserMessage(role="user", content="cancel my reservation"), agent.get_init_state()
    )

    assert assistant.content == "I can help with that."
    assert len(client.calls) == 1
    assert client.tool_batches == [[]]
    assert wiring.pending_notice_ids() == ()
    assert "Verify the reservation owner" in client.message_batches[0][0]["content"]


async def test_tau2_frozen_run_still_records_native_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeGradeRunner(Tau2Runner):
        def __init__(self) -> None:
            super().__init__()
            self.grade_calls = 0

        def _build(self, **kwargs: Any) -> tuple[Any, Any]:
            del kwargs
            simulation = SimpleNamespace(messages=[], termination_reason="done")
            return SimpleNamespace(run=lambda: simulation), SimpleNamespace(id="task-1")

        async def _grade(self, simulation: Any, task: Any) -> tuple[float, dict[str, Any]]:
            del simulation, task
            self.grade_calls += 1
            return 1.0, {"reward": 1.0, "grader": "native_tau2"}

    monkeypatch.setattr("pandabench.runners.tau2._require_tau2", lambda: None)
    runner = NativeGradeRunner()
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    wiring = FrozenEvalWiring(
        FrozenRulesSnapshot.create((), created_at="2026-08-05T01:00:00+00:00")
    )

    outcome = await runner.run_once(
        task_id="task-1", session_id="tau2-native-grade", model=model,
        client=MockClient(), max_turns=5, wiring=wiring,
    )

    assert runner.grade_calls == 1
    assert outcome.passed is True
    assert outcome.native_metrics == {"reward": 1.0, "grader": "native_tau2"}
    assert runner.outcome_for("task-1", "tau2-native-grade") == 1.0
