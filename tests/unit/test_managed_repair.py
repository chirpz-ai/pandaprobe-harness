"""Managed repair orchestration, settlement, and idempotency."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from pandaprobe_harness import Harness, HarnessConfig, RuleScopeHint
from pandaprobe_harness.repair.completion import (
    NormalizedRepairMessage,
    NormalizedToolCall,
    PandaProbeLiteLLMCompletion,
)
from pandaprobe_harness.repair.models import RepairUsage
from pandaprobe_harness.repair.prompt import REPAIR_SYSTEM_PROMPT
from pandaprobe_harness.workspace.mailbox import DiagnosticNotice
from tests.fakes.fake_cli_client import FakeCliClient

_REAL_REPAIR_COMPLETE = PandaProbeLiteLLMCompletion.complete


def test_repair_prompt_allows_universal_bounded_scope_names() -> None:
    assert "scope catalog is not fixed" in REPAIR_SYSTEM_PROMPT
    assert "Custom names have no required category prefix" in REPAIR_SYSTEM_PROMPT
    assert "`scoped` is the default" in REPAIR_SYSTEM_PROMPT
    assert "never supply a path" in REPAIR_SYSTEM_PROMPT


class TraceCaptureClient:
    """Minimal SDK client that retains finalized traces for hierarchy assertions."""

    enabled = True

    def __init__(self) -> None:
        self.traces: list[Any] = []

    def trace(self, name: str, **kwargs: Any) -> Any:
        from pandaprobe.tracing.context import TraceContext

        return TraceContext(client=cast(Any, self), name=name, **kwargs)

    def log_trace(self, trace: Any) -> None:
        self.traces.append(trace)


class ResolvingLiteLLM:
    """Provider-neutral fake exercised through PandaProbe's real LiteLLM wrapper."""

    def get_supported_openai_params(self, *, model: str) -> list[str]:
        del model
        return ["reasoning_effort"]

    async def acompletion(self, **kwargs: Any) -> Any:
        assignment = json.loads(str(kwargs["messages"][1]["content"]).split("\n", 1)[1])
        notice_id = str(assignment["notice_id"])
        calls = [
            SimpleNamespace(
                id="read",
                function=SimpleNamespace(
                    name="harness_notice_read",
                    arguments=json.dumps({"notice_id": notice_id}),
                ),
            ),
            SimpleNamespace(
                id="search",
                function=SimpleNamespace(
                    name="harness_rules_search",
                    arguments=json.dumps({"query": "payment retry"}),
                ),
            ),
            SimpleNamespace(
                id="resolve",
                function=SimpleNamespace(
                    name="harness_notice_resolve",
                    arguments=json.dumps(
                        {
                            "notice_id": notice_id,
                            "resolution": "no_proposal",
                            "note": "no transferable guidance",
                        }
                    ),
                ),
            ),
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="diagnosis", tool_calls=calls)
                )
            ],
            model=str(kwargs["model"]),
            usage=SimpleNamespace(
                prompt_tokens=11, completion_tokens=7, total_tokens=18
            ),
            _hidden_params={"response_cost": 0.01},
        )


class ScriptedCompletion:
    def __init__(self, mode: str = "candidate") -> None:
        self.mode = mode
        self.calls: list[dict[str, Any]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

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
        del tools, max_tokens, temperature, reasoning_effort, timeout_s
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
            }
        )
        assignment = json.loads(str(messages[1]["content"]).split("\n", 1)[1])
        notice_id = str(assignment["notice_id"])
        if self.mode == "exception":
            raise RuntimeError("provider failed with secret=do-not-log")
        if self.mode in {"block", "cancel"}:
            self.started.set()
            await self.release.wait()
            return _resolve(notice_id, "no_proposal")
        if self.mode == "malformed":
            return NormalizedRepairMessage(
                tool_calls=(NormalizedToolCall("bad", "harness_notice_read", "{"),)
            )
        if self.mode == "empty":
            return NormalizedRepairMessage()
        if self.mode == "consume_tokens":
            return NormalizedRepairMessage(
                tool_calls=(
                    NormalizedToolCall(
                        "read",
                        "harness_notice_read",
                        json.dumps({"notice_id": notice_id}),
                    ),
                ),
                usage=RepairUsage(output_tokens=5, total_tokens=5),
            )
        if self.mode == "read_forever":
            return NormalizedRepairMessage(
                tool_calls=(
                    NormalizedToolCall(
                        f"read-{len(self.calls)}",
                        "harness_notice_read",
                        json.dumps({"notice_id": notice_id}),
                    ),
                )
            )
        if self.mode == "search_loop":
            prior_tools = {
                str(message.get("name"))
                for message in messages
                if message.get("role") == "tool"
            }
            if "harness_notice_read" not in prior_tools:
                return NormalizedRepairMessage(
                    tool_calls=(
                        NormalizedToolCall(
                            "read",
                            "harness_notice_read",
                            json.dumps({"notice_id": notice_id}),
                        ),
                    )
                )
            if "harness_trace_inspect" not in prior_tools:
                return NormalizedRepairMessage(
                    tool_calls=(
                        NormalizedToolCall(
                            "inspect",
                            "harness_trace_inspect",
                            json.dumps({"trace_id": assignment["flagged_traces"][0]}),
                        ),
                    )
                )
            if "harness_rules_search" not in prior_tools:
                return NormalizedRepairMessage(
                    tool_calls=(
                        NormalizedToolCall(
                            "search",
                            "harness_rules_search",
                            json.dumps({"query": "payment retry"}),
                        ),
                    )
                )
            if any(
                message.get("role") == "system"
                and "Evidence review" in str(message.get("content"))
                for message in messages
            ):
                return _resolve(notice_id, "no_proposal")
            return NormalizedRepairMessage(
                tool_calls=(
                    NormalizedToolCall(
                        "search-again",
                        "harness_rules_search",
                        json.dumps({"query": "retry"}),
                    ),
                )
            )
        if self.mode == "no_proposal":
            return _resolve(notice_id, "no_proposal")
        if self.mode == "duplicate":
            existing = next(
                json.loads(str(message["content"]))["rules"][0]["id"]
                for message in messages
                if message.get("role") == "tool"
                and message.get("name") == "harness_rules_search"
            ) if any(
                message.get("role") == "tool"
                and message.get("name") == "harness_rules_search"
                for message in messages
            ) else None
            if existing is None:
                return NormalizedRepairMessage(
                    tool_calls=(
                        NormalizedToolCall(
                            "search",
                            "harness_rules_search",
                            json.dumps({"query": "payment retry"}),
                        ),
                    )
                )
            return NormalizedRepairMessage(
                tool_calls=(
                    NormalizedToolCall(
                        "duplicate",
                        "harness_notice_resolve",
                        json.dumps(
                            {
                                "notice_id": notice_id,
                                "resolution": "duplicate",
                                "existing_rule_id": existing,
                                "note": "existing rule covers this failure",
                            }
                        ),
                    ),
                )
            )
        if len(self.calls) == 1:
            candidate_args: dict[str, Any] = {
                "rule": "Verify payment status before retrying.",
                "rationale": "Avoid duplicate mutations.",
            }
            if self.mode == "candidate_appworld":
                candidate_args["scope"] = "appworld"
            if self.mode == "candidate_global":
                candidate_args.update({"scope": "global", "applicability": "global"})
            return NormalizedRepairMessage(
                tool_calls=(
                    NormalizedToolCall(
                        "read",
                        "harness_notice_read",
                        json.dumps({"notice_id": notice_id}),
                    ),
                    NormalizedToolCall(
                        "search",
                        "harness_rules_search",
                        json.dumps({"query": "payment retry"}),
                    ),
                    NormalizedToolCall(
                        "add",
                        "harness_rule_add",
                        json.dumps(candidate_args),
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


def _resolve(notice_id: str, resolution: str) -> NormalizedRepairMessage:
    return NormalizedRepairMessage(
        tool_calls=(
            NormalizedToolCall(
                "resolve",
                "harness_notice_resolve",
                json.dumps(
                    {
                        "notice_id": notice_id,
                        "resolution": resolution,
                        "note": "no transferable guidance",
                    }
                ),
            ),
        )
    )


def _config(tmp_path: Path, **overrides: object) -> HarnessConfig:
    return HarnessConfig(
        harness_root=tmp_path / "h",
        repair_model="anthropic/test-repair",
        poll_interval_s=0,
        eval_retry_backoff_s=0,
        gate_window=1,
        health_check=False,
        domain_policy="Supervisor authentication is authorized.",
        **{
            "repair_timeout_s": 1,
            "repair_max_turns": 4,
            **overrides,
        },  # type: ignore[arg-type]
    )


def _failing_cli(session: str = "task") -> FakeCliClient:
    cli = FakeCliClient()
    for _ in range(2):
        cli.script_trace(
            session,
            task_completion=0.2,
            coherence=0.2,
            tool_correctness=0.1,
            argument_correctness=0.1,
        )
    return cli


def _turn(harness: Harness, session: str = "task") -> None:
    harness.on_turn_end(
        {
            "session_id": session,
            "turn_index": 1,
            "end_state": {
                "task": "charge tx",
                "api_key": "must-not-escape",
                "authToken": "also-must-not-escape",
            },
        }
    )


async def test_healthy_report_creates_no_notice_or_repair(tmp_path: Path) -> None:
    completion = ScriptedCompletion()
    harness = Harness.create(
        _config(tmp_path), cli=FakeCliClient(), _repair_completion=completion
    )
    _turn(harness)
    result = await harness.settle("task")
    assert result.report is not None and not result.report.any_alert
    assert result.repair is None
    assert completion.calls == []


async def test_candidate_repair_is_single_flight_and_cached(tmp_path: Path) -> None:
    completion = ScriptedCompletion()
    harness = Harness.create(
        _config(tmp_path), cli=_failing_cli(), _repair_completion=completion
    )
    _turn(harness)
    first, concurrent = await asyncio.gather(
        harness.settle("task"), harness.settle("task")
    )
    repeated = await harness.settle("task")
    assert first.repair is not None and first.repair.status == "candidate_added"
    assert concurrent.repair == first.repair == repeated.repair
    assert len(completion.calls) == 2  # two model turns, one repair run
    assert len(harness.journal.recent(types=("repair_started",))) == 1
    assert len(harness.rules.candidates()) == 1
    assert harness.mailbox.pending() == []

    assignment = json.loads(
        str(completion.calls[0]["messages"][1]["content"]).split("\n", 1)[1]
    )
    assert assignment["domain_policy"].startswith("Supervisor authentication")
    assert assignment["task_descriptor"]["api_key"] == "[redacted]"
    assert assignment["task_descriptor"]["authToken"] == "[redacted]"


async def test_precise_host_scope_overrides_generic_benchmark_scope(tmp_path: Path) -> None:
    completion = ScriptedCompletion("candidate_appworld")
    harness = Harness.create(
        _config(tmp_path), cli=_failing_cli(), _repair_completion=completion
    )
    hint = RuleScopeHint(
        key="spotify",
        description="Spotify search, playlist, library, and playback workflows.",
    )
    harness.on_turn_end(
        {
            "session_id": "task",
            "turn_index": 1,
            "end_state": {"benchmark": "appworld", "task_id": "opaque-id"},
            "rule_scope_hints": [hint.to_json()],
        }
    )

    result = await harness.settle("task")

    assert result.repair is not None and result.repair.status == "candidate_added"
    assert result.repair.recommended_scope == "spotify"
    assert result.repair.selected_scope == "spotify"
    assert harness.rules.candidates()[0].scope == "spotify"
    assignment = json.loads(
        str(completion.calls[0]["messages"][1]["content"]).split("\n", 1)[1]
    )
    assert assignment["scope_hints"][0]["key"] == "spotify"
    assert assignment["recommended_scope"] == "spotify"


async def test_topical_and_cross_domain_repairs_choose_distinct_scopes(
    tmp_path: Path,
) -> None:
    venmo = Harness.create(
        _config(tmp_path / "venmo"),
        cli=_failing_cli(),
        _repair_completion=ScriptedCompletion(),
    )
    venmo_hint = RuleScopeHint(
        key="venmo",
        description="Venmo authentication, payment, reminder, and transaction workflows.",
    )
    venmo.on_turn_end(
        {
            "session_id": "task",
            "turn_index": 1,
            "end_state": {"benchmark": "appworld", "task_id": "opaque-id"},
            "rule_scope_hints": [venmo_hint.to_json()],
        }
    )
    venmo_result = await venmo.settle("task")
    assert venmo_result.repair is not None
    assert venmo_result.repair.selected_scope == "venmo"
    assert venmo.config.rules_scope_file("venmo").is_file()

    cross_domain = Harness.create(
        _config(tmp_path / "global"),
        cli=_failing_cli(),
        _repair_completion=ScriptedCompletion("candidate_global"),
    )
    cross_domain.on_turn_end(
        {
            "session_id": "task",
            "turn_index": 1,
            "end_state": {"benchmark": "appworld", "task_id": "opaque-id"},
            "rule_scope_hints": [venmo_hint.to_json()],
        }
    )
    global_result = await cross_domain.settle("task")
    assert global_result.repair is not None
    assert global_result.repair.selected_scope == "global"
    assert cross_domain.rules.candidates()[0].applicability == "global"


async def test_related_notices_form_one_repair_episode(tmp_path: Path) -> None:
    completion = ScriptedCompletion()
    harness = Harness.create(
        _config(tmp_path), cli=_failing_cli(), _repair_completion=completion
    )
    _turn(harness)
    await harness.refresh("task")
    first = harness.mailbox.pending()[0]
    second = replace(
        first,
        id=DiagnosticNotice.new_id(),
        summary="the outcome verifier observed the same failure",
    )
    harness.mailbox.post(second)

    result = await harness.settle("task")

    assert result.repair is not None and result.repair.status == "candidate_added"
    assert set(result.repair.notice_ids) == {first.id, second.id}
    assert len(result.repair.candidate_rule_ids) == 1
    assert len(harness.rules.candidates()) == 1
    assert harness.mailbox.pending() == []
    assert len(harness.journal.recent(types=("repair_started",))) == 1


async def test_distinct_failures_in_one_session_are_not_coalesced(tmp_path: Path) -> None:
    completion = ScriptedCompletion("no_proposal")
    harness = Harness.create(
        _config(tmp_path), cli=FakeCliClient(), _repair_completion=completion
    )
    notices = (
        DiagnosticNotice.from_json(
            {
                "id": "n-payment",
                "created_at": "2026-08-06T00:00:00+00:00",
                "session_id": "task",
                "turn_index": 1,
                "severity": "breach",
                "signatures": ["breach:tool_correctness"],
                "flagged_traces": ["trace-payment"],
            }
        ),
        DiagnosticNotice.from_json(
            {
                "id": "n-auth",
                "created_at": "2026-08-06T00:00:01+00:00",
                "session_id": "task",
                "turn_index": 1,
                "severity": "breach",
                "signatures": ["breach:argument_correctness"],
                "flagged_traces": ["trace-auth"],
            }
        ),
    )
    for notice in notices:
        harness.mailbox.post(notice)

    first = await harness.settle("task")
    second = await harness.settle("task")

    assert first.repair is not None and second.repair is not None
    assert first.repair.notice_ids != second.repair.notice_ids
    assert len(completion.calls) == 2
    assert harness.mailbox.pending() == []


async def test_duplicate_and_no_proposal_need_no_new_rule(tmp_path: Path) -> None:
    duplicate = ScriptedCompletion("duplicate")
    duplicate_harness = Harness.create(
        _config(tmp_path), cli=_failing_cli(), _repair_completion=duplicate
    )
    existing = duplicate_harness.rules.add(
        "Verify payment status before retrying.", "existing guidance"
    )
    duplicate_harness.rules.promote(existing.id, reason="fixture", validator="test")
    _turn(duplicate_harness)
    duplicate_result = await duplicate_harness.settle("task")
    assert duplicate_result.repair is not None
    assert duplicate_result.repair.status == "duplicate"
    assert duplicate_result.repair.existing_rule_id == existing.id
    assert len(duplicate_harness.rules.live()) == 1

    other_root = tmp_path / "other"
    no_proposal = ScriptedCompletion("no_proposal")
    no_proposal_harness = Harness.create(
        _config(other_root), cli=_failing_cli(), _repair_completion=no_proposal
    )
    _turn(no_proposal_harness)
    no_proposal_result = await no_proposal_harness.settle("task")
    assert no_proposal_result.repair is not None
    assert no_proposal_result.repair.status == "no_proposal"
    assert no_proposal_harness.rules.all() == []


async def test_exception_and_malformed_response_leave_notice_recoverable(
    tmp_path: Path,
) -> None:
    for index, mode in enumerate(("exception", "malformed", "empty")):
        completion = ScriptedCompletion(mode)
        harness = Harness.create(
            _config(tmp_path / str(index)),
            cli=_failing_cli(),
            _repair_completion=completion,
        )
        _turn(harness)
        result = await harness.settle("task")
        assert result.report is not None
        assert result.repair is not None and result.repair.status == "failed"
        assert len(harness.mailbox.pending()) == 1
        journal_text = harness.config.journal_file.read_text(encoding="utf-8")
        assert "must-not-log" not in journal_text
        assert "do-not-log" not in journal_text


async def test_timeout_does_not_fail_task_and_remains_recoverable(tmp_path: Path) -> None:
    completion = ScriptedCompletion("block")
    harness = Harness.create(
        _config(tmp_path, repair_timeout_s=0.01),
        cli=_failing_cli(),
        _repair_completion=completion,
    )
    _turn(harness)
    result = await harness.settle("task")
    repeated = await harness.settle("task")
    assert result.report is not None
    assert result.repair is not None and result.repair.status == "timed_out"
    assert repeated.repair is not None and repeated.repair.status == "timed_out"
    assert len(completion.calls) == 2
    assert len(harness.mailbox.pending()) == 1


async def test_turn_limit_is_bounded(tmp_path: Path) -> None:
    completion = ScriptedCompletion("read_forever")
    harness = Harness.create(
        _config(tmp_path, repair_max_turns=2),
        cli=_failing_cli(),
        _repair_completion=completion,
    )
    _turn(harness)
    result = await harness.settle("task")
    assert result.repair is not None
    assert result.repair.error_category == "turn_limit"
    assert len(completion.calls) == 2
    assert len(harness.mailbox.pending()) == 1


async def test_output_token_budget_is_enforced_across_turns(tmp_path: Path) -> None:
    completion = ScriptedCompletion("consume_tokens")
    harness = Harness.create(
        _config(tmp_path, repair_max_tokens=5),
        cli=_failing_cli(),
        _repair_completion=completion,
    )
    _turn(harness)
    result = await harness.settle("task")
    assert result.repair is not None
    assert result.repair.error_category == "token_limit"
    assert result.repair.usage.output_tokens == 5
    assert len(completion.calls) == 1
    assert len(harness.mailbox.pending()) == 1


async def test_progress_instruction_stops_repeated_read_only_searches(
    tmp_path: Path,
) -> None:
    completion = ScriptedCompletion("search_loop")
    harness = Harness.create(
        _config(tmp_path, repair_max_turns=4),
        cli=_failing_cli(),
        _repair_completion=completion,
    )
    _turn(harness)
    result = await harness.settle("task")
    assert result.repair is not None
    assert result.repair.status == "no_proposal"
    assert result.repair.turns == 4
    assert result.repair.tool_calls == 4
    assert [
        message["name"]
        for call in completion.calls
        for message in call["messages"]
        if message.get("role") == "tool"
    ].count("harness_rules_search") == 1


async def test_repair_uses_one_agent_trace_with_nested_model_and_tool_spans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pandaprobe.wrappers.litellm.wrapper import wrap_litellm

    capture = TraceCaptureClient()
    wrapped = wrap_litellm(ResolvingLiteLLM())
    monkeypatch.setattr(PandaProbeLiteLLMCompletion, "complete", _REAL_REPAIR_COMPLETE)
    monkeypatch.setattr("pandaprobe.get_client", lambda: capture)
    monkeypatch.setattr("pandaprobe.wrappers.litellm.wrap_litellm", lambda: wrapped)
    harness = Harness.create(
        _config(tmp_path, trace_repair_agent=True),
        cli=_failing_cli(),
    )
    _turn(harness)
    result = await harness.settle("task")
    assert result.repair is not None and result.repair.status == "no_proposal"
    assert len(capture.traces) == 1

    trace = capture.traces[0]
    assert trace.name == "pandaprobe"
    assert trace.session_id == result.repair.repair_session_id
    assert trace.metadata["role"] == "repair"
    assert trace.output["messages"][0]["role"] == "assistant"

    spans = {span.name: span for span in trace.spans}
    assert set(spans) == {
        "harness",
        "repair-agent",
        "litellm-chat",
        "tools",
        "harness_notice_read",
        "harness_rules_search",
        "harness_notice_resolve",
    }
    assert spans["harness"].kind.value == "CHAIN"
    assert spans["harness"].parent_span_id is None
    assert spans["repair-agent"].kind.value == "AGENT"
    assert spans["repair-agent"].parent_span_id == spans["harness"].span_id
    assert spans["litellm-chat"].kind.value == "LLM"
    assert spans["litellm-chat"].parent_span_id == spans["repair-agent"].span_id
    assert spans["tools"].kind.value == "AGENT"
    assert spans["tools"].parent_span_id == spans["harness"].span_id
    for name in (
        "harness_notice_read",
        "harness_rules_search",
        "harness_notice_resolve",
    ):
        assert spans[name].kind.value == "TOOL"
        assert spans[name].parent_span_id == spans["tools"].span_id


async def test_disabled_repair_tracing_suppresses_wrapper_standalone_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pandaprobe.wrappers.litellm.wrapper import wrap_litellm

    capture = TraceCaptureClient()
    wrapped = wrap_litellm(ResolvingLiteLLM())
    monkeypatch.setattr(PandaProbeLiteLLMCompletion, "complete", _REAL_REPAIR_COMPLETE)
    monkeypatch.setattr("pandaprobe.get_client", lambda: capture)
    monkeypatch.setattr("pandaprobe.wrappers.litellm.wrap_litellm", lambda: wrapped)
    harness = Harness.create(_config(tmp_path), cli=_failing_cli())
    _turn(harness)
    result = await harness.settle("task")
    assert result.repair is not None and result.repair.status == "no_proposal"
    assert capture.traces == []


async def test_caller_cancellation_does_not_strand_claim(tmp_path: Path) -> None:
    completion = ScriptedCompletion("cancel")
    harness = Harness.create(
        _config(tmp_path), cli=_failing_cli(), _repair_completion=completion
    )
    _turn(harness)
    settling = asyncio.create_task(harness.settle("task"))
    await completion.started.wait()
    settling.cancel()
    try:
        await settling
    except asyncio.CancelledError:
        pass
    completion.release.set()
    for _ in range(100):
        if not harness.mailbox.pending():
            break
        await asyncio.sleep(0)
    result = await harness.settle("task")
    assert result.repair is not None and result.repair.status == "no_proposal"
    assert len(completion.calls) == 1
