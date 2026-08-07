"""Credential-free task, CLI, and completion fakes used by offline examples."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pandaprobe_harness.cli.client import CliResult
from pandaprobe_harness.repair.completion import (
    NormalizedRepairMessage,
    NormalizedToolCall,
)

GUIDANCE = "Verify the exact transaction status before retrying a payment mutation."


@dataclass
class OfflineCli:
    traces: dict[str, list[str]] = field(default_factory=dict)
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    runs: dict[str, tuple[list[str], list[str]]] = field(default_factory=dict)
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def script_trace(self, session_id: str, **scores: float) -> str:
        ids = self.traces.setdefault(session_id, [])
        trace_id = f"{session_id}-trace-{len(ids) + 1}"
        ids.append(trace_id)
        self.scores[trace_id] = scores
        return trace_id

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        del timeout
        self.calls.append(args)
        payload: Any = {}
        if args[:1] == ("version",):
            payload = {"version": "offline"}
        elif args[:2] == ("auth", "status"):
            payload = {"authenticated": True}
        elif args[:2] == ("traces", "list"):
            session_id = _flag(args, "--session-id") or ""
            payload = {
                "items": [
                    {
                        "trace_id": trace_id,
                        "status": "COMPLETED",
                        "started_at": f"2026-08-05T00:00:0{index}+00:00",
                        "session_id": session_id,
                    }
                    for index, trace_id in reversed(
                        list(enumerate(self.traces.get(session_id, [])))
                    )
                ]
            }
        elif args[:3] == ("evals", "runs", "batch"):
            run_id = f"run-{len(self.runs) + 1}"
            trace_ids = (_flag(args, "--trace-ids") or "").split(",")
            metrics = (_flag(args, "--metrics") or "").split(",")
            self.runs[run_id] = (trace_ids, metrics)
            payload = {"id": run_id, "status": "PENDING"}
        elif args[:3] == ("evals", "runs", "scores"):
            trace_ids, metrics = self.runs[args[3]]
            payload = [
                {
                    "name": metric,
                    "value": str(self.scores.get(trace_id, {}).get(metric, 0.9)),
                    "status": "SUCCESS",
                    "reason": f"offline {metric} score",
                    "trace_id": trace_id,
                }
                for trace_id in trace_ids
                for metric in metrics
            ]
        elif args[:2] == ("traces", "get"):
            payload = {"trace_id": args[2], "spans": []}
        elif args[:2] == ("traces", "spans"):
            payload = {"trace_id": args[2], "spans": []}
        elif args[:3] == ("evals", "scores", "get"):
            payload = {"trace_id": args[3], "scores": []}
        return CliResult(args=args, exit_code=0, stdout=json.dumps(payload), stderr="")


class OfflineRepairCompletion:
    """Fake only the wrapped model completion; package orchestration stays real."""

    def __init__(self) -> None:
        self.calls = 0

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
        del model, tools, max_tokens, temperature, reasoning_effort
        del timeout_s
        self.calls += 1
        assignment = json.loads(str(messages[1]["content"]).split("\n", 1)[1])
        notice_id = assignment["notice_id"]
        if self.calls == 1:
            return NormalizedRepairMessage(
                tool_calls=(
                    NormalizedToolCall(
                        "read", "harness_notice_read", json.dumps({"notice_id": notice_id})
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
                                "rationale": "The failed turn repeated a mutation.",
                                "metric": "tool_correctness",
                                "scope": "payments",
                            }
                        ),
                    ),
                )
            )
        added = next(
            json.loads(str(message["content"]))["rule"]["id"]
            for message in messages
            if message.get("role") == "tool" and message.get("name") == "harness_rule_add"
        )
        return NormalizedRepairMessage(
            tool_calls=(
                NormalizedToolCall(
                    "ack",
                    "harness_notice_ack",
                    json.dumps({"notice_id": notice_id, "rule_id": added}),
                ),
            )
        )


class DeveloperTaskAgent:
    def __init__(self) -> None:
        self.actions: list[str] = []

    async def run_turn(self, context: str, task_tools: Any) -> dict[str, str]:
        """Choose whether to discover and read optional learned guidance."""

        assert GUIDANCE not in context
        guidance = ""
        index = await task_tools.call("harness_rules_list", {})
        if any(scope["scope"] == "payments" for scope in index["scopes"]):
            scoped = await task_tools.call(
                "harness_rules_read", {"scope": "payments"}
            )
            guidance = str(scoped["content"])
        action = "check_status_then_charge" if GUIDANCE in guidance else "charge_twice"
        self.actions.append(action)
        return {"task": "charge tx-1", "action": action}


def _flag(args: Sequence[str], name: str) -> str | None:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None
