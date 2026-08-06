"""Task-facing read-only learned-guidance context."""

from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig
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
    assert "Relevant learned guidance" in context
    assert "mailbox" not in context.casefold()
    assert "acknowledge" not in context.casefold()
    assert "inspect diagnostic" not in context.casefold()
    assert "write a rule" not in context.casefold()


def test_current_session_candidate_is_visible_next_turn(tmp_path: Path) -> None:
    rules, mailbox = _stores(tmp_path, topk=0)
    notice = _notice("n-current", "task-session", 2)
    mailbox.post(notice)
    candidate = rules.add(
        "Verify exact target IDs before mutation.",
        "Avoid broad writes",
        source_notice_id=notice.id,
    )
    context = compose_system_preamble(rules, mailbox, "task-session")
    assert f"[candidate {candidate.id}, added after turn 2]" in context
    assert candidate.rule in context


def test_other_session_candidates_respect_the_context_bound(tmp_path: Path) -> None:
    rules, mailbox = _stores(tmp_path, topk=1)
    for index in range(3):
        notice = _notice(f"n-{index}", "other-session", index)
        mailbox.post(notice)
        rules.add(f"Other guidance {index}", "x", source_notice_id=notice.id)
    context = compose_system_preamble(rules, mailbox, "task-session", task_hint="Other")
    assert sum(f"Other guidance {index}" in context for index in range(3)) == 1


def test_current_session_candidates_also_respect_the_context_bound(tmp_path: Path) -> None:
    rules, mailbox = _stores(tmp_path, topk=2)
    for index in range(4):
        notice = _notice(f"n-current-{index}", "task-session", index)
        mailbox.post(notice)
        rules.add(f"Current guidance {index}", "x", source_notice_id=notice.id)
    context = compose_system_preamble(rules, mailbox, "task-session")
    assert sum(f"Current guidance {index}" in context for index in range(4)) == 2
    assert "Current guidance 3" in context


def test_active_and_candidate_labels_are_preserved_and_retired_excluded(
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
    assert f"[active {active.id}]" in context
    assert retired.rule not in context


def test_context_deduplicates_a_current_relevant_candidate(tmp_path: Path) -> None:
    rules, mailbox = _stores(tmp_path)
    notice = _notice("n-current", "s", 1)
    mailbox.post(notice)
    rule = rules.add("Check payment status.", "avoid retry", source_notice_id=notice.id)
    context = compose_system_preamble(rules, mailbox, "s", task_hint="payment")
    assert context.count(rule.rule) == 1


def test_context_survives_unprovisioned_workspace(tmp_path: Path) -> None:
    config = HarnessConfig(harness_root=tmp_path / "missing", repair_model="test/fake")
    context = compose_system_preamble(
        RulesStore(config, journal=Journal(config)), Mailbox(config), "s"
    )
    assert "PANDAPROBE HARNESS" in context
