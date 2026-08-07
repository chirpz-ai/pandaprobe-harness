"""The task-facing read-only learned-rules context."""

from __future__ import annotations

from pathlib import Path

import pytest

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.agent_tools.toolset import TaskToolset
from pandaprobe_harness.hook.context import compose_system_preamble
from pandaprobe_harness.workspace.journal import Journal
from pandaprobe_harness.workspace.mailbox import DiagnosticNotice, Mailbox
from pandaprobe_harness.workspace.rules import RulesStore


def _stores(tmp_path: Path, *, topk: int = 3):  # type: ignore[no-untyped-def]
    config = HarnessConfig(
        harness_root=tmp_path / "h", rules_context_topk=topk, repair_model="test/fake"
    )
    journal = Journal(config)
    mailbox = Mailbox(config)
    mailbox.provision()
    return RulesStore(config, journal=journal), mailbox


def _notice(notice_id: str, session_id: str, turn: int) -> DiagnosticNotice:
    return DiagnosticNotice.from_json(
        {
            "id": notice_id,
            "created_at": "2026-08-05T00:00:00+00:00",
            "session_id": session_id,
            "turn_index": turn,
            "severity": "breach",
            "summary": "failure",
        }
    )


def test_context_has_no_mailbox_or_repair_instructions(tmp_path: Path) -> None:
    rules, mailbox = _stores(tmp_path)
    mailbox.post(_notice("n-1", "s-1", 1))
    context = compose_system_preamble(rules, mailbox, "s-1")
    assert "PANDAPROBE HARNESS" in context
    assert "Learned rules are available" in context
    assert "does not automatically insert rule contents" in context
    assert "harness_rules_list" in context
    assert "rules.md" in context
    assert "load" in context
    # Product terminology: these are learned RULES, not "optional guidance".
    assert "optional" not in context.casefold()
    assert "guidance" not in context.casefold()
    assert "mailbox" not in context.casefold()
    assert "acknowledge" not in context.casefold()
    assert "inspect diagnostic" not in context.casefold()
    assert "write a rule" not in context.casefold()


def test_current_session_candidate_is_not_injected_next_turn(tmp_path: Path) -> None:
    rules, mailbox = _stores(tmp_path, topk=0)
    notice = _notice("n-current", "task-session", 2)
    mailbox.post(notice)
    candidate = rules.add(
        "Verify exact target IDs before mutation.",
        "Avoid broad writes",
        source_notice_id=notice.id,
    )
    context = compose_system_preamble(rules, mailbox, "task-session")
    assert candidate.id not in context
    assert candidate.rule not in context
    assert "rules/global.md" not in context


def test_other_session_candidates_are_not_injected(tmp_path: Path) -> None:
    rules, mailbox = _stores(tmp_path, topk=1)
    for index in range(3):
        notice = _notice(f"n-{index}", "other-session", index)
        mailbox.post(notice)
        rules.add(f"Other guidance {index}", "x", source_notice_id=notice.id)
    context = compose_system_preamble(rules, mailbox, "task-session", task_hint="Other")
    assert all(f"Other guidance {index}" not in context for index in range(3))


def test_all_current_session_candidates_are_not_injected(tmp_path: Path) -> None:
    rules, mailbox = _stores(tmp_path, topk=2)
    for index in range(4):
        notice = _notice(f"n-current-{index}", "task-session", index)
        mailbox.post(notice)
        rules.add(f"Current guidance {index}", "x", source_notice_id=notice.id)
    context = compose_system_preamble(rules, mailbox, "task-session")
    assert all(f"Current guidance {index}" not in context for index in range(4))


def test_active_candidate_and_retired_bodies_are_all_excluded(
    tmp_path: Path,
) -> None:
    config = HarnessConfig(
        harness_root=tmp_path / "h", rule_validation=False, repair_model="test/fake"
    )
    mailbox = Mailbox(config)
    mailbox.provision()
    rules = RulesStore(config, journal=Journal(config))
    active = rules.add("Paginate all pages before totals.", "complete totals")
    retired = rules.add("Use the old endpoint.", "obsolete")
    rules.retire(retired.id)
    context = compose_system_preamble(rules, mailbox, "s")
    assert active.rule not in context
    assert retired.rule not in context


def test_task_hint_does_not_trigger_automatic_search(tmp_path: Path) -> None:
    rules, mailbox = _stores(tmp_path)
    notice = _notice("n-current", "s", 1)
    mailbox.post(notice)
    rule = rules.add("Check payment status.", "avoid retry", source_notice_id=notice.id)
    context = compose_system_preamble(rules, mailbox, "s", task_hint="payment")
    assert rule.rule not in context


def test_context_does_not_read_rule_or_mailbox_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules, mailbox = _stores(tmp_path)

    def unexpected(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("context construction performed automatic retrieval")

    monkeypatch.setattr(rules, "render_root", unexpected)
    monkeypatch.setattr(rules, "search", unexpected)
    monkeypatch.setattr(rules, "live", unexpected)
    monkeypatch.setattr(mailbox, "read", unexpected)
    context = compose_system_preamble(rules, mailbox, "s", task_hint="payment")
    assert "Learned rules are available" in context


def test_context_survives_unprovisioned_workspace(tmp_path: Path) -> None:
    config = HarnessConfig(harness_root=tmp_path / "missing", repair_model="test/fake")
    context = compose_system_preamble(
        RulesStore(config, journal=Journal(config)), Mailbox(config), "s"
    )
    assert "PANDAPROBE HARNESS" in context


async def test_a_new_candidate_becomes_discoverable_for_the_next_turn(
    tmp_path: Path,
) -> None:
    """The pull contract: a rule learned this turn is readable next turn, on demand.

    Discoverable, never injected — the preamble is identical before and after.
    """

    rules, mailbox = _stores(tmp_path)
    config = rules.config
    tools = TaskToolset(config=config, rules=rules)

    before = compose_system_preamble(rules, mailbox, "s-next")
    assert (await tools.call("harness_rules_read", {}))["content"].find(
        "Verify the recipient"
    ) == -1

    candidate = rules.add("Verify the recipient before sending.", "avoid misdelivery")

    after = compose_system_preamble(rules, mailbox, "s-next")
    assert after == before  # the context did not grow by one rule
    assert candidate.rule not in after
    read = await tools.call("harness_rules_read", {"scope": candidate.scope})
    assert candidate.rule in read["content"]
    listed = await tools.call("harness_rules_list", {})
    assert [entry["scope"] for entry in listed["scopes"]] == [candidate.scope]
    assert candidate.rule not in listed["content"]  # the index carries no bodies


async def test_the_four_read_only_tools_are_the_whole_task_surface(
    tmp_path: Path,
) -> None:
    rules, _ = _stores(tmp_path)
    tools = TaskToolset(config=rules.config, rules=rules)

    assert {spec.name for spec in tools.specs()} == {
        "harness_rules_read",
        "harness_rules_search",
        "harness_rules_list",
        "harness_rule_status",
    }
    # Administration and mutation fail at dispatch, however plausibly named.
    for name in (
        "harness_rule_add",
        "harness_rule_promote",
        "harness_rule_retire",
        "harness_notice_ack",
        "harness_notice_read",
        "harness_mailbox_list",
        "harness_validate",
        "harness_trace_inspect",
    ):
        result = await tools.call(name, {})
        assert result["ok"] is False
        assert "unsupported capability" in result["error"]
    assert rules.all() == []  # nothing was created by trying


async def test_no_rule_tool_is_called_while_building_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Context construction must not perform discovery on the agent's behalf."""

    rules, mailbox = _stores(tmp_path)
    rules.add("Verify the recipient before sending.", "avoid misdelivery")
    tools = TaskToolset(config=rules.config, rules=rules)

    async def unexpected(name: str, args: object) -> object:
        del args
        raise AssertionError(f"context construction called {name}")

    monkeypatch.setattr(tools, "call", unexpected)

    context = compose_system_preamble(rules, mailbox, "s", task_hint="recipient")

    assert "Learned rules are available" in context
    assert tools.surfaced_rule_ids == frozenset()  # nothing was read
