"""Unit tests for task-conditioned lexical rule retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from pandaprobe_harness import HarnessConfig, HarnessToolset, Journal, Mailbox, RulesStore
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


def test_relevant_keeps_globals_and_caps_scoped(tmp_path: Path) -> None:
    """Eligibility is decided by *scope*, not by whether a rule carries tags —
    v1 conflated the two, so any rule added without a notice became a permanent
    global by accident."""

    store = _store(tmp_path, topk=1)
    global_rule = store.add("always read before writing", "x", scope="global")
    relevant = store.add("verify payments", "x", tags=["payment"], scope="scoped")
    other_a = store.add("rule about databases", "x", tags=["database"], scope="scoped")
    other_b = store.add("rule about emails", "x", tags=["email"], scope="scoped")

    selected = store.relevant("payment failed", k=1)
    ids = [rule.id for rule in selected]
    assert global_rule.id in ids  # globals are always eligible, exempt from k
    assert relevant.id in ids
    assert len(ids) == 2  # 1 global + top-1 scoped
    assert other_a.id not in ids and other_b.id not in ids


def test_a_tagged_rule_filed_as_global_is_still_always_eligible(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, topk=1)
    tagged_global = store.add("hold the invariant", "x", tags=["email"], scope="global")

    assert tagged_global.id in {r.id for r in store.relevant("payment", k=1)}


def test_relevant_falls_back_to_recency_without_overlap(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("older scoped rule", "x", tags=["alpha"], scope="scoped")
    newer = store.add("newer scoped rule", "x", tags=["beta"], scope="scoped")

    selected = store.relevant("zzz-no-overlap", k=1)
    assert [rule.id for rule in selected] == [newer.id]


def test_query_none_renders_everything(tmp_path: Path) -> None:
    store = _store(tmp_path, topk=1)
    a = store.add("first tagged", "x", tags=["alpha"])
    b = store.add("second tagged", "x", tags=["beta"])
    assert {rule.id for rule in store.relevant(None, k=1)} == {a.id, b.id}


def test_render_scope_notes_omitted_rules(tmp_path: Path) -> None:
    store = _store(tmp_path, topk=1)
    store.add("rule about payments", "x", tags=["payment"], scope="scoped")
    store.add("rule about databases", "x", tags=["database"], scope="scoped")
    store.add("rule about emails", "x", tags=["email"], scope="scoped")

    narrowed = store.render_scope("scoped", query="payment")
    assert "rule about payments" in narrowed
    assert "rule about databases" not in narrowed
    assert "2 more active rule(s) available" in narrowed
    assert "harness_rules_search" in narrowed

    # The on-disk artifact is always the full render.
    full = store.render_scope("scoped")
    assert "rule about databases" in full and "more active rule(s)" not in full


def test_render_markdown_covers_every_scope(tmp_path: Path) -> None:
    """The full-corpus render is the *replay* context, so it must reach across
    every scope — a rule is only validated fairly if it was actually in force."""

    store = _store(tmp_path)
    store.add("global lesson", "x", scope="global")
    store.add("scoped lesson", "x", scope="scoped")
    store.add("payments lesson", "x", scope="payments")

    full = store.render_markdown()

    assert "global lesson" in full
    assert "scoped lesson" in full
    assert "payments lesson" in full


def test_retrieval_off_renders_every_active_rule(tmp_path: Path) -> None:
    store = _store(tmp_path, retrieval=False)
    store.add("rule about payments", "x", tags=["payment"], scope="scoped")
    store.add("rule about databases", "x", tags=["database"], scope="scoped")

    rendered = store.render_scope("scoped", query="payment")
    assert "rule about payments" in rendered
    assert "rule about databases" in rendered  # nothing filtered
    assert "more active rule(s)" not in rendered


def test_search_filters_by_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    keep = store.add("about payments", "x", tags=["payment"])
    gone = store.add("retired payment rule", "x", tags=["payment"])
    store.retire(gone.id)

    active_only = store.search("payment", limit=10)
    assert [rule.id for rule, _ in active_only] == [keep.id]

    retired_only = store.search("payment", limit=10, statuses=("retired",))
    assert [rule.id for rule, _ in retired_only] == [gone.id]


def test_preamble_never_carries_rule_text_however_it_is_hinted(tmp_path: Path) -> None:
    """v1 selected rules with a task hint and inlined them. v2 does not inline any
    rule text at all, so the hint has nothing to select — the agent conditions its
    own retrieval through `harness_rules_search` / `harness_rules_read`."""

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
    rules.add("verify payment status first", "x", tags=["payment"], scope="scoped")
    rules.add("email retries must back off", "x", tags=["email"], scope="scoped")

    for preamble in (
        compose_system_preamble(rules, mailbox),
        compose_system_preamble(rules, mailbox),
    ):
        assert "verify payment status first" not in preamble
        assert "email retries must back off" not in preamble
        assert "rules/scoped.md" in preamble  # named, so it is one call away


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


def test_notice_signature_tokens_still_rank_the_matching_rule(tmp_path: Path) -> None:
    """A notice's signatures are auto-derived into a rule's tags, so a later query
    built from those signatures still finds it — the ranking is unchanged, only
    the *caller* moved from the preamble to the agent's own search."""

    store = _store(tmp_path, topk=1)
    matching = store.add(
        "verify payment status first",
        "x",
        tags=["breach:tool_correctness"],
        scope="scoped",
    )
    store.add("email retries must back off", "x", tags=["email"], scope="scoped")

    selected = store.relevant("breach:tool_correctness tool_correctness", k=1)

    assert [rule.id for rule in selected] == [matching.id]


def test_candidates_render_even_when_a_query_filters_actives(tmp_path: Path) -> None:
    """Retrieval must never starve a trial: candidates render in full under any
    query, outside the top-k budget, because they have to be in force to be
    measurable."""

    from pandaprobe_harness.workspace.rules import PROVISIONAL_HEADING

    config = HarnessConfig(
        harness_root=tmp_path / "harness",
        rule_validation=True,
        rule_retrieval=True,
        rules_context_topk=1,
    )
    store = RulesStore(config, journal=Journal(config))
    store.add("candidate about databases", "x", tags=["database"], scope="scoped")
    active_a = store.add("payment rule", "x", tags=["payment"], scope="scoped")
    store.promote(active_a.id)
    active_b = store.add("email rule", "x", tags=["email"], scope="scoped")
    store.promote(active_b.id)

    rendered = store.render_scope("scoped", query="payment")

    assert "payment rule" in rendered
    assert "email rule" not in rendered  # trimmed by top-k
    assert PROVISIONAL_HEADING in rendered
    assert "candidate about databases" in rendered  # survives the filter
