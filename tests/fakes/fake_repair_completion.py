"""Deterministic fake for the PandaProbe/LiteLLM completion seam."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pandaprobe_harness.repair.completion import (
    NormalizedRepairMessage,
    NormalizedToolCall,
)
from pandaprobe_harness.repair.models import RepairUsage


class FakeRepairCompletion:
    """Resolve assigned notices without network calls; retain every invocation."""

    def __init__(self, responses: Sequence[NormalizedRepairMessage] = ()) -> None:
        self.responses = list(responses)
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
        if self.responses:
            return self.responses.pop(0)
        assignment = json.loads(str(messages[1]["content"]).split("\n", 1)[1])
        notice_id = str(assignment["notice_id"])
        return NormalizedRepairMessage(
            tool_calls=(
                NormalizedToolCall(
                    id="read",
                    name="harness_notice_read",
                    arguments=json.dumps({"notice_id": notice_id}),
                ),
                NormalizedToolCall(
                    id="search",
                    name="harness_rules_search",
                    arguments=json.dumps({"query": assignment["summary"] or "failure"}),
                ),
                NormalizedToolCall(
                    id="resolve",
                    name="harness_notice_resolve",
                    arguments=json.dumps(
                        {
                            "notice_id": notice_id,
                            "resolution": "no_proposal",
                            "note": "deterministic offline test resolution",
                        }
                    ),
                ),
            ),
            usage=RepairUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
