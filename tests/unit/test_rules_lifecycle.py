"""Unit tests for the candidate → active | retired rule lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pandaprobe_harness import HarnessConfig, Journal, RulesStore
from pandaprobe_harness.workspace.mailbox import DiagnosticNotice, NoticeMetric
from pandaprobe_harness.workspace.rules import (
    PROVISIONAL_HEADING,
    Rule,
    TrialState,
    _as_status,
    derive_notice_tags,
)


@pytest.fixture
def validating_config(tmp_path: Path) -> HarnessConfig:
    return HarnessConfig(harness_root=tmp_path / "harness", rule_validation=True)


@pytest.fixture
def validating_journal(validating_config: HarnessConfig) -> Journal:
    return Journal(validating_config)


@pytest.fixture
def validating_rules(validating_config: HarnessConfig, validating_journal: Journal) -> RulesStore:
    return RulesStore(validating_config, journal=validating_journal)


# -- status parsing -------------------------------------------------------------


def test_as_status_forgiving_parse() -> None:
    assert _as_status("candidate") == "candidate"
    assert _as_status("retired") == "retired"
    assert _as_status("active") == "active"
    assert _as_status("bogus") == "active"
    assert _as_status(None) == "active"


def test_persisted_candidate_round_trips(validating_rules: RulesStore) -> None:
    """A candidate must survive from_json — not silently coerce to active."""

    rule = validating_rules.add("Check the ledger before charging", "avoid double charges")
    assert rule.status == "candidate"

    reloaded = validating_rules.all()
    assert len(reloaded) == 1
    assert reloaded[0].status == "candidate"
    assert validating_rules.active() == []
    assert [r.id for r in validating_rules.candidates()] == [rule.id]
    assert [r.id for r in validating_rules.live()] == [rule.id]


def test_legacy_v05_record_parses_as_active(tmp_path: Path) -> None:
    config = HarnessConfig(harness_root=tmp_path / "harness")
    config.rules_store_file.parent.mkdir(parents=True, exist_ok=True)
    legacy = {
        "id": "r-legacy",
        "created_at": "2026-01-01T00:00:00+00:00",
        "rule": "A v0.5 rule",
        "rationale": "written before the lifecycle existed",
        "source_notice_id": None,
        "metric": None,
        "status": "active",
    }
    config.rules_store_file.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    store = RulesStore(config)
    (rule,) = store.all()
    assert rule.status == "active"
    assert rule.tags == ()
    assert rule.trial is None


# -- add ------------------------------------------------------------------------


def test_add_is_candidate_only_when_validation_enabled(tmp_path: Path) -> None:
    plain = RulesStore(
        HarnessConfig(harness_root=tmp_path / "plain", rule_validation=False)
    )
    assert plain.add("rule one", "r").status == "active"  # the v0.5-compat switch

    validating = RulesStore(
        HarnessConfig(harness_root=tmp_path / "validating", rule_validation=True)
    )
    assert validating.add("rule one", "r").status == "candidate"


def test_add_captures_baseline_from_journal(
    validating_rules: RulesStore, validating_journal: Journal
) -> None:
    for session, signatures in (
        ("s-1", ["breach:agent_reliability"]),
        ("s-2", ["trend:agent_consistency"]),
        ("s-3", ["relative:agent_reliability"]),
    ):
        validating_journal.record(
            {"type": "notice", "session_id": session, "signatures": signatures}
        )
    # A recovered session widens the denominator (it had an incident but is
    # not currently breaching this metric family).
    validating_journal.record({"type": "recovery", "session_id": "s-4"})

    rule = validating_rules.add("verify before retry", "x", metric="agent_reliability")
    trial = rule.trial
    assert trial is not None
    assert trial.baseline_sessions == 4
    # s-1 (breach) and s-3 (relative) are in agent_reliability's family;
    # s-2 is another metric, s-4 only recovered.
    assert trial.baseline_breached_sessions == 2
    assert trial.baseline_window == 4
    assert trial.trial_started_at
    assert trial.baseline_rate == pytest.approx(0.5)


def test_baseline_rate_defaults_to_one_with_no_history(validating_rules: RulesStore) -> None:
    rule = validating_rules.add("verify before retry", "x", metric="agent_reliability")
    assert rule.trial is not None
    assert rule.trial.baseline_sessions == 0
    assert rule.trial.baseline_rate == 1.0


def test_add_dedups_against_candidates(validating_rules: RulesStore) -> None:
    first = validating_rules.add("Always verify inputs.", "one")
    duplicate = validating_rules.add("  always VERIFY inputs!!", "two")
    assert duplicate.id == first.id
    assert len(validating_rules.all()) == 1


def test_cap_counts_candidates(tmp_path: Path) -> None:
    config = HarnessConfig(
        harness_root=tmp_path / "capped", rule_validation=True, max_active_rules=2
    )
    store = RulesStore(config)
    store.add("rule number one", "r1")
    store.add("rule number two", "r2")
    from pandaprobe_harness import RulesCapError

    with pytest.raises(RulesCapError):
        store.add("rule number three", "r3")


def test_add_cleans_explicit_tags(validating_rules: RulesStore) -> None:
    rule = validating_rules.add(
        "tag me",
        "x",
        tags=["  Breach:Agent_Reliability ", "breach:agent_reliability", "", "x" * 100],
    )
    assert rule.tags[0] == "breach:agent_reliability"
    assert len(rule.tags) == 2  # dedup + empty dropped
    assert len(rule.tags[1]) == 48  # length-capped


# -- promote / retire / trial ----------------------------------------------------


def test_promote_moves_candidate_to_active_and_journals(
    validating_rules: RulesStore, validating_journal: Journal
) -> None:
    rule = validating_rules.add("verify before retry", "x", metric="agent_reliability")
    promoted = validating_rules.promote(
        rule.id, reason="replay improved agent_reliability by 0.62", validator="replay"
    )
    assert promoted.status == "active"
    assert [r.id for r in validating_rules.active()] == [rule.id]
    assert validating_rules.candidates() == []

    (event,) = validating_journal.recent(types=("rule_promote",))
    assert event["id"] == rule.id
    assert event["validator"] == "replay"
    assert "improved" in event["reason"]


def test_promote_rejects_non_candidates(validating_rules: RulesStore) -> None:
    rule = validating_rules.add("verify before retry", "x")
    validating_rules.promote(rule.id)
    with pytest.raises(KeyError):
        validating_rules.promote(rule.id)  # already active
    with pytest.raises(KeyError):
        validating_rules.promote("r-missing")


def test_retire_candidate_with_reason(
    validating_rules: RulesStore, validating_journal: Journal
) -> None:
    rule = validating_rules.add("verify before retry", "x")
    retired = validating_rules.retire(rule.id, reason="forward-trial: no improvement")
    assert retired.status == "retired"
    assert validating_rules.live() == []

    (event,) = validating_journal.recent(types=("rule_retire",))
    assert event["id"] == rule.id
    assert event["reason"] == "forward-trial: no improvement"


def test_update_trial_mutates_fresh_state_under_the_lock(
    validating_rules: RulesStore,
) -> None:
    from dataclasses import replace

    rule = validating_rules.add("verify before retry", "x")
    assert rule.trial is not None

    validating_rules.update_trial(
        rule.id,
        lambda t: replace(
            t, observed_sessions=("s-1", "s-2"), breached_sessions=("s-2",), replay_attempts=1
        ),
    )

    (reloaded,) = validating_rules.candidates()
    assert reloaded.trial is not None
    assert reloaded.trial.observed_sessions == ("s-1", "s-2")
    assert reloaded.trial.breached_sessions == ("s-2",)
    assert reloaded.trial.replay_attempts == 1
    assert reloaded.trial.trial_rate == pytest.approx(0.5)

    # The mutate closure receives the FRESH trial (not a caller snapshot):
    # a second update sees the first one's sessions.
    seen: list[tuple[str, ...]] = []

    def _second(trial: TrialState) -> TrialState:
        seen.append(trial.observed_sessions)
        return replace(trial, observed_sessions=(*trial.observed_sessions, "s-3"))

    validating_rules.update_trial(rule.id, _second)
    assert seen == [("s-1", "s-2")]
    (reloaded,) = validating_rules.candidates()
    assert reloaded.trial is not None
    assert reloaded.trial.observed_sessions == ("s-1", "s-2", "s-3")

    # Returning the same object signals "no change" — nothing is appended.
    before = validating_rules._config.rules_store_file.read_text(encoding="utf-8")
    validating_rules.update_trial(rule.id, lambda t: t)
    after = validating_rules._config.rules_store_file.read_text(encoding="utf-8")
    assert before == after

    validating_rules.promote(rule.id)
    with pytest.raises(KeyError):
        validating_rules.update_trial(rule.id, lambda t: t)


# -- rendering --------------------------------------------------------------------


def test_candidate_renders_in_provisional_section(
    validating_rules: RulesStore, validating_config: HarnessConfig
) -> None:
    candidate = validating_rules.add("Never charge twice without verifying", "duplicates seen")
    markdown = validating_config.rules_file.read_text(encoding="utf-8")

    assert PROVISIONAL_HEADING in markdown
    assert f"**{candidate.id}** (candidate): Never charge twice" in markdown
    assert "trial: 0/5 sessions observed" in markdown
    assert "_No learned rules yet._" not in markdown

    validating_rules.promote(candidate.id)
    markdown = validating_config.rules_file.read_text(encoding="utf-8")
    assert PROVISIONAL_HEADING not in markdown
    assert f"**{candidate.id}**: Never charge twice" in markdown


def test_active_rules_render_before_provisional_section(validating_rules: RulesStore) -> None:
    first = validating_rules.add("first learned rule", "a")
    validating_rules.promote(first.id)
    second = validating_rules.add("second provisional rule", "b")

    markdown = validating_rules.render_markdown()
    assert markdown.index("first learned rule") < markdown.index(PROVISIONAL_HEADING)
    assert markdown.index(PROVISIONAL_HEADING) < markdown.index("second provisional rule")
    assert second.id in markdown


def test_v05_rendering_unchanged_without_candidates(tmp_path: Path) -> None:
    """With validation off the rendered markdown matches the v0.5 shape."""

    store = RulesStore(
        HarnessConfig(harness_root=tmp_path / "plain", rule_validation=False)
    )
    rule = store.add("plain active rule", "why")
    markdown = store.render_markdown()
    assert PROVISIONAL_HEADING not in markdown
    assert f"- **{rule.id}**: plain active rule" in markdown


# -- tag derivation ---------------------------------------------------------------


def test_derive_notice_tags_collects_signatures_metrics_signals() -> None:
    notice = DiagnosticNotice(
        id="n-1",
        created_at="2026-07-01T00:00:00+00:00",
        session_id="s-1",
        turn_index=3,
        severity="breach",
        metrics=(
            NoticeMetric(name="agent_reliability", value=0.3, threshold=0.5),
            NoticeMetric(name="agent_consistency", value=0.4, threshold=0.5),
        ),
        signal_breakdown={
            "trace-1": {"loop_detection": 0.1, "tool_correctness": 0.2},
        },
        signatures=("breach:agent_reliability", "breach:agent_consistency"),
    )

    tags = derive_notice_tags(notice)
    assert "breach:agent_reliability" in tags
    assert "agent_consistency" in tags
    assert "loop_detection" in tags
    assert "tool_correctness" in tags
    assert "breach" in tags
    assert len(tags) == len(set(tags))


def test_rule_json_round_trip_with_lifecycle_fields() -> None:
    rule = Rule(
        id="r-abc",
        created_at="2026-07-01T00:00:00+00:00",
        rule="text",
        rationale="why",
        status="candidate",
        tags=("breach:agent_reliability",),
        trial=TrialState(baseline_sessions=4, baseline_breached_sessions=3),
    )
    parsed = Rule.from_json(rule.to_json())
    assert parsed == rule
