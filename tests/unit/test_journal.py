"""Unit tests for the append-only diagnostic journal."""

from __future__ import annotations

from pandaprobe_harness import HarnessConfig, Journal


def test_record_defaults_ts_and_type_and_returns_stored_dict(
    journal: Journal, config: HarnessConfig
) -> None:
    stored = journal.record({"detail": "hello"})
    assert stored["type"] == "unknown"
    assert stored["ts"]
    assert stored["detail"] == "hello"
    # The returned dict is exactly what was persisted.
    assert journal.recent() == [stored]
    assert config.journal_file.exists()


def test_record_preserves_explicit_ts_and_type(journal: Journal) -> None:
    stored = journal.record({"type": "notice", "ts": "2026-01-01T00:00:00+00:00"})
    assert stored["type"] == "notice"
    assert stored["ts"] == "2026-01-01T00:00:00+00:00"


def test_recent_filters_by_type_preserves_order_and_honors_limit(journal: Journal) -> None:
    journal.record({"type": "notice", "n": 1})
    journal.record({"type": "rule_add", "n": 2})
    journal.record({"type": "notice", "n": 3})
    journal.record({"type": "ack", "n": 4})
    journal.record({"type": "notice", "n": 5})

    notices = journal.recent(types=("notice",))
    assert [e["n"] for e in notices] == [1, 3, 5]

    limited = journal.recent(limit=2, types=("notice",))
    assert [e["n"] for e in limited] == [3, 5]

    everything = journal.recent(limit=0)
    assert [e["n"] for e in everything] == [1, 2, 3, 4, 5]


def test_recent_on_missing_journal_is_empty(journal: Journal) -> None:
    assert journal.recent() == []


def test_notices_for_matches_on_notice_metric_names(journal: Journal) -> None:
    journal.record({"type": "notice", "n": 1, "metrics": [{"name": "agent_reliability"}]})
    journal.record({"type": "notice", "n": 2, "metrics": [{"name": "agent_consistency"}]})
    # Non-notice events never match, even with a matching metric.
    journal.record({"type": "rule_add", "n": 3, "metrics": [{"name": "agent_reliability"}]})
    # Notice events without metrics are skipped.
    journal.record({"type": "notice", "n": 4})
    journal.record(
        {
            "type": "notice",
            "n": 5,
            "metrics": [{"name": "agent_consistency"}, {"name": "agent_reliability"}],
        }
    )

    matches = journal.notices_for("agent_reliability")
    assert [e["n"] for e in matches] == [1, 5]
    assert journal.notices_for("no_such_metric") == []


def test_corrupt_and_blank_lines_are_skipped(journal: Journal, config: HarnessConfig) -> None:
    journal.record({"type": "notice", "n": 1})
    with config.journal_file.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write("not json at all\n")
        handle.write("[1, 2, 3]\n")
    journal.record({"type": "notice", "n": 2})

    events = journal.recent(limit=0)
    assert [e["n"] for e in events] == [1, 2]
