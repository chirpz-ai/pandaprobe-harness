"""Capability-boundary tests for task and managed-repair tools."""

from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.agent_tools.toolset import (
    REPAIR_OP_SCHEMAS,
    TASK_OP_SCHEMAS,
    RepairToolset,
    TaskToolset,
)
from pandaprobe_harness.workspace.journal import Journal
from pandaprobe_harness.workspace.mailbox import DiagnosticNotice, Mailbox
from pandaprobe_harness.workspace.rules import RulesStore
from tests.fakes.fake_cli_client import FakeCliClient


def _workspace(tmp_path: Path):  # type: ignore[no-untyped-def]
    config = HarnessConfig(harness_root=tmp_path / "h", repair_model="test/fake")
    journal = Journal(config)
    mailbox = Mailbox(config)
    mailbox.provision()
    rules = RulesStore(config, journal=journal)
    notice = DiagnosticNotice.from_json(
        {
            "id": "n-assigned",
            "created_at": "2026-08-05T00:00:00+00:00",
            "session_id": "task-1",
            "turn_index": 2,
            "severity": "breach",
            "summary": "payment mutation repeated",
            "flagged_traces": ["trace-task"],
            "signatures": ["breach:tool_correctness"],
            "scope_hint": "scoped",
        }
    )
    mailbox.post(notice)
    return config, journal, mailbox, rules, notice


async def test_task_surface_is_exactly_read_only(tmp_path: Path) -> None:
    config, _, _, rules, _ = _workspace(tmp_path)
    tools = TaskToolset(config=config, rules=rules)
    assert {spec.name for spec in tools.specs()} == set(TASK_OP_SCHEMAS) == {
        "harness_rules_read",
        "harness_rules_search",
        "harness_rules_list",
        "harness_rule_status",
    }


async def test_hallucinated_task_administration_is_rejected(tmp_path: Path) -> None:
    config, _, mailbox, rules, _ = _workspace(tmp_path)
    tools = TaskToolset(config=config, rules=rules)
    for name in (
        "harness_mailbox_list",
        "harness_notice_read",
        "harness_trace_inspect",
        "harness_rule_add",
        "harness_rule_retire",
        "harness_notice_ack",
        "harness_validate",
    ):
        result = await tools.call(name, {})
        assert result["ok"] is False
    assert len(mailbox.pending()) == 1
    assert rules.all() == []


async def test_task_can_list_search_read_and_inspect_status(tmp_path: Path) -> None:
    config, _, _, rules, _ = _workspace(tmp_path)
    rule = rules.add("Verify payment status before retrying.", "Avoid duplicate writes")
    tools = TaskToolset(config=config, rules=rules)
    index = await tools.call("harness_rules_list", {})
    assert index["scopes"][0]["scope"] == "scoped"
    assert index["scopes"][0]["provisional"] == 1
    assert rule.rule not in index["content"]
    assert (await tools.call("harness_rules_search", {"query": "payment"}))["rules"]
    assert rule.rule in (await tools.call("harness_rules_read", {}))["content"]
    assert (await tools.call("harness_rule_status", {"rule_id": rule.id}))["ok"] is True


async def test_repair_surface_is_restricted_and_assignment_scoped(tmp_path: Path) -> None:
    config, journal, mailbox, rules, notice = _workspace(tmp_path)
    tools = RepairToolset(
        config=config,
        cli=FakeCliClient(),
        mailbox=mailbox,
        journal=journal,
        rules=rules,
        notice_id=notice.id,
        allowed_trace_ids=notice.flagged_traces,
    )
    assert {spec.name for spec in tools.specs()} == set(REPAIR_OP_SCHEMAS)
    for forbidden in (
        "execute",
        "bash",
        "harness_rule_promote",
        "harness_rule_retire",
        "harness_validate",
        "harness_regression_run",
    ):
        assert (await tools.call(forbidden, {}))["ok"] is False
    wrong = await tools.call("harness_notice_read", {"notice_id": "n-other"})
    assert wrong["ok"] is False
    outside = await tools.call("harness_trace_inspect", {"trace_id": "repair-trace"})
    assert outside["ok"] is False


async def test_repair_adds_candidate_then_acknowledges(tmp_path: Path) -> None:
    config, journal, mailbox, rules, notice = _workspace(tmp_path)
    tools = RepairToolset(
        config=config,
        cli=FakeCliClient(),
        mailbox=mailbox,
        journal=journal,
        rules=rules,
        notice_id=notice.id,
        allowed_trace_ids=notice.flagged_traces,
    )
    added = await tools.call(
        "harness_rule_add",
        {"rule": "Verify status before retrying.", "rationale": "Avoid duplicate writes"},
    )
    assert added["ok"] is True
    assert added["rule"]["status"] == "candidate"
    premature = await tools.call(
        "harness_notice_ack", {"notice_id": notice.id, "rule_id": "r-other"}
    )
    assert premature["ok"] is False
    ack = await tools.call(
        "harness_notice_ack",
        {"notice_id": notice.id, "rule_id": added["rule"]["id"]},
    )
    assert ack["ok"] is True
    assert mailbox.pending() == []
    assert rules.candidates()[0].id == added["rule"]["id"]


async def test_duplicate_and_no_proposal_are_explicit(tmp_path: Path) -> None:
    config, journal, mailbox, rules, notice = _workspace(tmp_path)
    existing = rules.add("Verify status before retrying.", "Existing guidance")
    tools = RepairToolset(
        config=config,
        cli=FakeCliClient(),
        mailbox=mailbox,
        journal=journal,
        rules=rules,
        notice_id=notice.id,
        allowed_trace_ids=notice.flagged_traces,
    )
    result = await tools.call(
        "harness_notice_resolve",
        {
            "notice_id": notice.id,
            "resolution": "already_covered",
            "existing_rule_id": existing.id,
            "note": "already covered",
        },
    )
    assert result["ok"] is True
    processed = mailbox.read(notice.id)
    assert processed is not None and processed.resolution is not None
    assert processed.resolution.kind == "already_covered"
