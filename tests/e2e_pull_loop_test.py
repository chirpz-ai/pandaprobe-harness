"""Arbitrary developer task loop with package-owned managed repair."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pandaprobe_harness import Harness, HarnessConfig
from pandaprobe_harness.repair.completion import (
    NormalizedRepairMessage,
    NormalizedToolCall,
)
from pandaprobe_harness.repair.models import RepairUsage
from tests.fakes.fake_cli_client import FakeCliClient

SESSION = "developer-task-session"
GUIDANCE = "Verify the exact transaction status before retrying a payment mutation."


class CandidateRepairCompletion:
    """Script the normalized PandaProbe/LiteLLM tool-call contract."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": list(tools),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "timeout_s": timeout_s,
            }
        )
        assignment = json.loads(str(messages[1]["content"]).split("\n", 1)[1])
        if len(self.calls) == 1:
            return NormalizedRepairMessage(
                tool_calls=(
                    NormalizedToolCall(
                        "read",
                        "harness_notice_read",
                        json.dumps({"notice_id": assignment["notice_id"]}),
                    ),
                    NormalizedToolCall(
                        "trace",
                        "harness_trace_inspect",
                        json.dumps({"trace_id": assignment["flagged_traces"][0]}),
                    ),
                    NormalizedToolCall(
                        "search",
                        "harness_rules_search",
                        json.dumps({"query": "payment retry"}),
                    ),
                    NormalizedToolCall(
                        "add",
                        "harness_rule_add",
                        json.dumps(
                            {
                                "rule": GUIDANCE,
                                "rationale": "The failed turn repeated a mutation blindly.",
                                "metric": "tool_correctness",
                                "scope": "payments",
                            }
                        ),
                    ),
                ),
                usage=RepairUsage(input_tokens=100, output_tokens=40, total_tokens=140),
            )
        add_result = next(
            json.loads(str(message["content"]))
            for message in messages
            if message.get("role") == "tool" and message.get("name") == "harness_rule_add"
        )
        return NormalizedRepairMessage(
            tool_calls=(
                NormalizedToolCall(
                    "ack",
                    "harness_notice_ack",
                    json.dumps(
                        {
                            "notice_id": assignment["notice_id"],
                            "rule_id": add_result["rule"]["id"],
                        }
                    ),
                ),
            ),
            usage=RepairUsage(input_tokens=80, output_tokens=10, total_tokens=90),
        )


class DeveloperTaskAgent:
    """The host owns this object, its prompt handling, and its domain action."""

    def __init__(self) -> None:
        self.actions: list[str] = []

    async def turn(self, context: str) -> dict[str, Any]:
        action = "check_status_then_charge" if GUIDANCE in context else "charge_twice"
        self.actions.append(action)
        return {"action": action, "task": "charge transaction tx-1"}


async def test_managed_repair_reaches_same_session_next_turn(tmp_path: Path) -> None:
    config = HarnessConfig(
        harness_root=tmp_path / "harness",
        repair_model="openai/test-repair-model",
        repair_timeout_s=5,
        repair_max_turns=4,
        poll_interval_s=0,
        eval_retry_backoff_s=0,
        gate_window=1,
        health_check=False,
    )
    cli = FakeCliClient()
    completion = CandidateRepairCompletion()
    harness = Harness.create(config, cli=cli, _repair_completion=completion)
    task_agent = DeveloperTaskAgent()

    first_context = harness.system_context(SESSION, task_hint="payment")
    first_end_state = await task_agent.turn(first_context)
    for _ in range(2):
        cli.script_trace(
            SESSION,
            task_completion=0.2,
            coherence=0.3,
            tool_correctness=0.1,
            argument_correctness=0.1,
        )
    harness.on_turn_end(
        {"session_id": SESSION, "turn_index": 1, "end_state": first_end_state}
    )
    settled = await harness.settle(SESSION)

    assert settled.report is not None and settled.report.any_alert
    assert settled.repair is not None and settled.repair.status == "completed"
    assert len(settled.repair.candidate_rule_ids) == 1
    assert len(completion.calls) == 2
    assert all(call["model"] == config.repair_model for call in completion.calls)
    assert settled.repair.repair_session_id != SESSION
    assert settled.repair.repair_session_id.startswith(f"repair-{SESSION}-")
    assert harness.mailbox.pending() == []

    candidate = harness.rules.candidates()[0]
    second_context = harness.system_context(SESSION, task_hint="payment")
    assert f"[candidate {candidate.id}, added after turn 1]" in second_context
    assert GUIDANCE in second_context
    second_end_state = await task_agent.turn(second_context)
    assert task_agent.actions == ["charge_twice", "check_status_then_charge"]

    cli.script_trace(
        SESSION,
        task_completion=0.9,
        coherence=0.9,
        tool_correctness=0.9,
        argument_correctness=0.9,
    )
    harness.on_turn_end(
        {"session_id": SESSION, "turn_index": 2, "end_state": second_end_state}
    )
    await harness.settle(SESSION)
    assert len(completion.calls) == 2, "repair did not recurse or run twice"
    listed_sessions = [
        call[call.index("--session-id") + 1]
        for call in cli.calls
        if call[:2] == ("traces", "list") and "--session-id" in call
    ]
    assert listed_sessions and set(listed_sessions) == {SESSION}


def test_persisted_workspace_is_reused_without_task_administration(tmp_path: Path) -> None:
    config = HarnessConfig(
        harness_root=tmp_path / "harness",
        repair_model="test/fake",
        health_check=False,
    )
    first = Harness.create(
        config, cli=FakeCliClient(), _repair_completion=CandidateRepairCompletion()
    )
    rule = first.rules.add("Paginate all result pages.", "avoid incomplete totals")
    second = Harness.create(
        config, cli=FakeCliClient(), _repair_completion=CandidateRepairCompletion()
    )
    assert rule.rule in second.system_context("new-session")
    assert {spec.name for spec in second.task_tools.specs()} == {
        "harness_rules_read",
        "harness_rules_search",
        "harness_rules_list",
        "harness_rule_status",
    }
