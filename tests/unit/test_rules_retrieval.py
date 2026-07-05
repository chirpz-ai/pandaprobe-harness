"""Unit tests for task-conditioned lexical rule retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from pandaprobe_harness import HarnessConfig, HarnessToolset, Journal, Mailbox, RulesStore
from pandaprobe_harness.evaluation.history import ScoreHistoryStore
from pandaprobe_harness.hook.context import compose_system_preamble
from pandaprobe_harness.workspace.rules import _tokenize
from tests.fakes.fake_cli_client import FakeCliClient


def _store(tmp_path: Path, *, topk: int = 2, retrieval: bool = True) -> RulesStore:
    config = HarnessConfig(
        harness_root=tmp_path / "harness",
        rule_validation=False,  # adds are active immediately; retrieval is the subject
        rule_retrieval=retrieval,
        rules_context_topk=topk,
    )
    return RulesStore(config, journal=Journal(config))


def test_tokenizer_splits_signatures_and_keeps_metric_names() -> None:
    tokens = _tokenize("breach:agent_reliability Charge the payment!")
    assert "breach" in tokens
    assert "agent_reliability" in tokens  # underscore keeps the metric whole
    assert "charge" in tokens
    assert "payment" in tokens
    assert "the" in tokens  # 3 chars, kept
    assert "a" not in tokens  # single chars dropped


def test_tag_match_outranks_text_match_outranks_unrelated(tmp_path: Path) -> None:
    store = _store(tmp_path, topk=3)
    tagged = store.add(
        "verify transactions first", "x", tags=["breach:agent_reliability", "payment"]
    )
    text_only = store.add("check the payment ledger twice", "x", tags=["misc"])
    unrelated = store.add("prefer smaller diffs", "x", tags=["style"])

    results = store.search("payment breach", limit=10)
    ordered = [rule.id for rule, _ in results]
    assert ordered[0] == tagged.id
    assert ordered[1] == text_only.id
    assert ordered[2] == unrelated.id
    scores = {rule.id: score for rule, score in results}
    assert scores[tagged.id] > scores[text_only.id] > scores[unrelated.id] == 0.0


def test_relevant_keeps_globals_and_caps_tagged(tmp_path: Path) -> None:
    store = _store(tmp_path, topk=1)
    global_rule = store.add("always read before writing", "x")  # untagged = global
    relevant = store.add("verify payments", "x", tags=["payment"])
    other_a = store.add("rule about databases", "x", tags=["database"])
    other_b = store.add("rule about emails", "x", tags=["email"])

    selected = store.relevant("payment failed", k=1)
    ids = [rule.id for rule in selected]
    assert global_rule.id in ids  # globals always render, exempt from k
    assert relevant.id in ids
    assert len(ids) == 2  # 1 global + top-1 tagged
    assert other_a.id not in ids and other_b.id not in ids


def test_relevant_falls_back_to_recency_without_overlap(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("older tagged rule", "x", tags=["alpha"])
    newer = store.add("newer tagged rule", "x", tags=["beta"])

    selected = store.relevant("zzz-no-overlap", k=1)
    assert [rule.id for rule in selected] == [newer.id]


def test_query_none_renders_everything(tmp_path: Path) -> None:
    store = _store(tmp_path, topk=1)
    a = store.add("first tagged", "x", tags=["alpha"])
    b = store.add("second tagged", "x", tags=["beta"])
    assert {rule.id for rule in store.relevant(None, k=1)} == {a.id, b.id}


def test_render_markdown_notes_omitted_rules(tmp_path: Path) -> None:
    store = _store(tmp_path, topk=1)
    store.add("rule about payments", "x", tags=["payment"])
    store.add("rule about databases", "x", tags=["database"])
    store.add("rule about emails", "x", tags=["email"])

    markdown = store.render_markdown(query="payment")
    assert "rule about payments" in markdown
    assert "rule about databases" not in markdown
    assert "2 more active rule(s) available" in markdown
    assert "harness_rules_search" in markdown

    # The on-disk artifact is always the full render.
    full = store.render_markdown()
    assert "rule about databases" in full and "more active rule(s)" not in full


def test_retrieval_off_reproduces_v05_rendering(tmp_path: Path) -> None:
    store = _store(tmp_path, retrieval=False)
    store.add("rule about payments", "x", tags=["payment"])
    store.add("rule about databases", "x", tags=["database"])

    markdown = store.render_markdown(query="payment")
    assert "rule about payments" in markdown
    assert "rule about databases" in markdown  # nothing filtered
    assert "more active rule(s)" not in markdown


def test_search_filters_by_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    keep = store.add("about payments", "x", tags=["payment"])
    gone = store.add("retired payment rule", "x", tags=["payment"])
    store.retire(gone.id)

    active_only = store.search("payment", limit=10)
    assert [rule.id for rule, _ in active_only] == [keep.id]

    retired_only = store.search("payment", limit=10, statuses=("retired",))
    assert [rule.id for rule, _ in retired_only] == [gone.id]


def test_preamble_task_hint_drives_retrieval(tmp_path: Path) -> None:
    config = HarnessConfig(
        harness_root=tmp_path / "harness",
        rule_validation=False,
        rule_retrieval=True,
        rules_context_topk=1,
    )
    journal = Journal(config)
    rules = RulesStore(config, journal=journal)
    mailbox = Mailbox(config)
    mailbox.provision()
    rules.add("verify payment status first", "x", tags=["payment"])
    rules.add("email retries must back off", "x", tags=["email"])

    hinted = compose_system_preamble(rules, mailbox, task_hint="charge a payment")
    assert "verify payment status first" in hinted
    assert "email retries must back off" not in hinted
    assert "1 more active rule(s) available" in hinted

    # No hint and no pending notices → no signal → everything renders.
    unhinted = compose_system_preamble(rules, mailbox)
    assert "verify payment status first" in unhinted
    assert "email retries must back off" in unhinted


async def test_toolset_search_and_list_ops(tmp_path: Path) -> None:
    config = HarnessConfig(
        harness_root=tmp_path / "harness",
        rule_validation=False,
        rule_retrieval=True,
    )
    journal = Journal(config)
    rules = RulesStore(config, journal=journal)
    mailbox = Mailbox(config)
    mailbox.provision()
    toolset = HarnessToolset(
        config=config,
        cli=FakeCliClient(),
        mailbox=mailbox,
        journal=journal,
        rules=rules,
        history=ScoreHistoryStore(config),
    )
    payment = rules.add("verify payments", "x", tags=["payment"])
    retired = rules.add("old email rule", "x", tags=["email"])
    rules.retire(retired.id)

    found = await toolset.call("harness_rules_search", {"query": "payment", "limit": 5})
    assert found["ok"] is True
    assert found["rules"][0]["id"] == payment.id
    assert found["rules"][0]["score"] == pytest.approx(2.0)

    everything = await toolset.call("harness_rules_list", {})
    assert {r["id"] for r in everything["rules"]} == {payment.id, retired.id}

    retired_only = await toolset.call("harness_rules_list", {"status": "retired"})
    assert [r["id"] for r in retired_only["rules"]] == [retired.id]

    searched_retired = await toolset.call(
        "harness_rules_search", {"query": "email", "status": "retired"}
    )
    assert [r["id"] for r in searched_retired["rules"]] == [retired.id]


def test_pending_notice_signatures_drive_retrieval_without_a_hint(
    tmp_path: Path,
) -> None:
    """The retrieval query must come from the mailbox too: with no task hint,
    a pending notice's signatures/metrics select the matching rule."""

    from pandaprobe_harness.workspace.mailbox import DiagnosticNotice, NoticeMetric

    config = HarnessConfig(
        harness_root=tmp_path / "harness",
        rule_validation=False,
        rule_retrieval=True,
        rules_context_topk=1,
    )
    journal = Journal(config)
    rules = RulesStore(config, journal=journal)
    mailbox = Mailbox(config)
    mailbox.provision()
    rules.add("verify payment status first", "x", tags=["breach:agent_reliability"])
    rules.add("email retries must back off", "x", tags=["email"])
    mailbox.post(
        DiagnosticNotice(
            id="n-retr-1",
            created_at="2026-07-03T00:00:00+00:00",
            session_id="s-1",
            turn_index=1,
            severity="breach",
            metrics=(NoticeMetric(name="agent_reliability", value=0.3, threshold=0.5),),
            signatures=("breach:agent_reliability",),
        )
    )

    preamble = compose_system_preamble(rules, mailbox)  # no task_hint
    assert "verify payment status first" in preamble
    assert "email retries must back off" not in preamble
    assert "1 more active rule(s) available" in preamble


def test_candidates_render_even_when_query_filters_actives(tmp_path: Path) -> None:
    """Retrieval must never starve a trial: candidates render in full under
    any query, outside the top-k budget."""

    from pandaprobe_harness.workspace.rules import PROVISIONAL_HEADING

    config = HarnessConfig(
        harness_root=tmp_path / "harness",
        rule_validation=True,
        rule_retrieval=True,
        rules_context_topk=1,
    )
    store = RulesStore(config, journal=Journal(config))
    candidate = store.add("candidate about databases", "x", tags=["database"])
    active_a = store.add("payment rule", "x", tags=["payment"])
    store.promote(active_a.id)
    active_b = store.add("email rule", "x", tags=["email"])
    store.promote(active_b.id)

    markdown = store.render_markdown(query="payment")
    assert "payment rule" in markdown
    assert "email rule" not in markdown  # trimmed by top-k
    assert PROVISIONAL_HEADING in markdown
    assert candidate.rule in markdown  # the candidate survives the filter
