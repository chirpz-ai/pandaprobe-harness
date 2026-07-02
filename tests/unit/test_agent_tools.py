"""Unit tests for the agent-facing ``HarnessToolset`` operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pandaprobe_harness import (
    DiagnosticNotice,
    HarnessConfig,
    HarnessFilesystem,
    HarnessToolset,
    Journal,
    Mailbox,
    RulesStore,
    ScoreHistoryStore,
)
from pandaprobe_harness.cli.errors import CliError
from tests.fakes.fake_cli_client import FakeCliClient


def _post_notice(
    config: HarnessConfig,
    mailbox: Mailbox,
    *,
    notice_id: str = "n-20260701T000000000000-abcd1234",
    session_id: str = "s-1",
) -> tuple[DiagnosticNotice, dict[str, Any]]:
    """Write a per-notice dump, build the notice pointing at it, and post it."""

    fs = HarnessFilesystem(config)
    fs.provision()
    dump = {
        "session_id": session_id,
        "turn_index": 1,
        "scores": [{"name": "agent_reliability", "value": 0.2, "status": "SUCCESS"}],
    }
    fs.write_trace_dump(notice_id, dump)
    notice = DiagnosticNotice.from_json(
        {
            "id": notice_id,
            "created_at": "2026-07-01T00:00:00+00:00",
            "session_id": session_id,
            "turn_index": 1,
            "severity": "breach",
            "metrics": [
                {
                    "name": "agent_reliability",
                    "value": 0.2,
                    "threshold": 0.5,
                    "reason": "score below threshold",
                    "conditions": ["breach"],
                }
            ],
            "flagged_traces": ["t-1"],
            "summary": "agent_reliability breached its threshold",
            "signatures": ["breach:agent_reliability"],
            "dump_path": str(config.traces_dir / f"{notice_id}.json"),
        }
    )
    mailbox.post(notice)
    return notice, dump


# -- mailbox ---------------------------------------------------------------------


async def test_mailbox_list_reports_status_and_pending_summaries(
    toolset: HarnessToolset, config: HarnessConfig, mailbox: Mailbox
) -> None:
    notice, _ = _post_notice(config, mailbox)

    result = await toolset.call("harness_mailbox_list", {})

    assert result["ok"] is True
    assert result["status"]["pending_count"] == 1
    assert result["status"]["max_severity"] == "breach"
    assert result["status"]["latest_id"] == notice.id
    assert result["pending"] == [
        {
            "id": notice.id,
            "severity": "breach",
            "session_id": "s-1",
            "metrics": ["agent_reliability"],
            "summary": notice.summary,
        }
    ]


async def test_mailbox_read_returns_full_notice_and_dump(
    toolset: HarnessToolset, config: HarnessConfig, mailbox: Mailbox
) -> None:
    notice, dump = _post_notice(config, mailbox)

    result = await toolset.call("harness_mailbox_read", {"notice_id": notice.id})

    assert result["ok"] is True
    assert result["notice"] == notice.to_json()
    assert result["notice"]["metrics"][0]["name"] == "agent_reliability"
    assert result["dump"] == dump


async def test_mailbox_read_unknown_notice_is_error_envelope(
    toolset: HarnessToolset,
) -> None:
    result = await toolset.call("harness_mailbox_read", {"notice_id": "n-missing"})
    assert result["ok"] is False
    assert "n-missing" in result["error"]


async def test_mailbox_ack_moves_notice_and_journals(
    toolset: HarnessToolset, config: HarnessConfig, mailbox: Mailbox, journal: Journal
) -> None:
    notice, _ = _post_notice(config, mailbox)

    result = await toolset.call(
        "harness_mailbox_ack",
        {"notice_id": notice.id, "rule_id": "r-1", "note": "mitigated"},
    )

    assert result["ok"] is True
    assert mailbox.pending() == []
    read = await toolset.call("harness_mailbox_read", {"notice_id": notice.id})
    assert read["ok"] is True
    assert read["notice"]["status"] == "acknowledged"
    assert read["notice"]["resolution"]["rule_id"] == "r-1"
    assert read["notice"]["resolution"]["note"] == "mitigated"

    events = journal.recent(types=("ack",))
    assert len(events) == 1
    assert events[0]["notice_id"] == notice.id
    assert events[0]["session_id"] == "s-1"

    # Acking a notice that is no longer pending degrades to an envelope.
    again = await toolset.call("harness_mailbox_ack", {"notice_id": notice.id})
    assert again["ok"] is False
    assert notice.id in again["error"]


# -- platform introspection --------------------------------------------------------


async def test_trace_inspect_returns_trace_spans_and_scores(
    toolset: HarnessToolset,
) -> None:
    result = await toolset.call("harness_trace_inspect", {"trace_id": "t-1"})

    assert result["ok"] is True
    assert result["trace_id"] == "t-1"
    assert result["trace"] == {"trace_id": "t-1", "spans": []}
    assert result["tool_spans"] == {"trace_id": "t-1", "spans": []}
    assert result["scores"]["id"] == "t-1"
    assert {s["name"] for s in result["scores"]["scores"]} == {
        "agent_reliability",
        "agent_consistency",
    }


async def test_trace_inspect_degrades_partially_on_cli_error(
    toolset: HarnessToolset, fake_cli: FakeCliClient
) -> None:
    fake_cli.error_on_prefix = {("traces", "get"): CliError("x")}

    result = await toolset.call("harness_trace_inspect", {"trace_id": "t-1"})

    assert result["ok"] is True
    assert result["trace"] is None
    assert result["tool_spans"] is not None
    assert result["scores"] is not None


# -- history -----------------------------------------------------------------------


async def test_history_with_session_returns_local_and_backend(
    toolset: HarnessToolset, config: HarnessConfig, fake_cli: FakeCliClient
) -> None:
    store = ScoreHistoryStore(config)
    store.record("s-1", "agent_reliability", 0.4, run_id="run-1", ts="2026-07-01T00:00:00+00:00")
    store.record("s-1", "agent_reliability", 0.6, run_id="run-2", ts="2026-07-01T00:01:00+00:00")
    backend_items = [{"metric_name": "agent_reliability", "value": "0.5", "run_id": "run-0"}]
    fake_cli.session_scores_list["s-1"] = backend_items

    result = await toolset.call(
        "harness_history", {"metric": "agent_reliability", "session_id": "s-1"}
    )

    assert result["ok"] is True
    assert result["metric"] == "agent_reliability"
    assert result["session_id"] == "s-1"
    assert result["local"] == [
        {"value": 0.4, "ts": "2026-07-01T00:00:00+00:00", "run_id": "run-1"},
        {"value": 0.6, "ts": "2026-07-01T00:01:00+00:00", "run_id": "run-2"},
    ]
    assert result["backend"] == {"items": backend_items}


async def test_history_without_session_is_empty_local_and_no_backend(
    toolset: HarnessToolset,
) -> None:
    result = await toolset.call("harness_history", {"metric": "agent_reliability"})

    assert result["ok"] is True
    assert result["session_id"] is None
    assert result["local"] == []
    assert result["backend"] is None


# -- journal -----------------------------------------------------------------------


async def test_journal_honors_limit_and_types(
    toolset: HarnessToolset, journal: Journal
) -> None:
    journal.record({"type": "notice", "id": "n-1"})
    journal.record({"type": "ack", "notice_id": "n-1"})
    journal.record({"type": "notice", "id": "n-2"})
    journal.record({"type": "notice", "id": "n-3"})

    limited = await toolset.call("harness_journal", {"limit": 2, "types": ["notice"]})
    assert limited["ok"] is True
    assert [e["id"] for e in limited["events"]] == ["n-2", "n-3"]

    acks = await toolset.call("harness_journal", {"types": ["ack"]})
    assert [e["type"] for e in acks["events"]] == ["ack"]

    everything = await toolset.call("harness_journal", {})
    assert len(everything["events"]) == 4


# -- rules -------------------------------------------------------------------------


async def test_rule_add_records_provenance(toolset: HarnessToolset) -> None:
    result = await toolset.call(
        "harness_rule_add",
        {
            "rule": "Always validate tool arguments before calling.",
            "rationale": "Repeated reliability breaches from malformed calls.",
            "notice_id": "n-1",
            "metric": "agent_reliability",
        },
    )

    assert result["ok"] is True
    rule = result["rule"]
    assert rule["id"].startswith("r-")
    assert rule["rule"] == "Always validate tool arguments before calling."
    assert rule["rationale"] == "Repeated reliability breaches from malformed calls."
    assert rule["source_notice_id"] == "n-1"
    assert rule["metric"] == "agent_reliability"
    assert rule["status"] == "active"
    assert rule["created_at"]


async def test_rule_add_at_cap_is_error_mentioning_retire(tmp_path: Path) -> None:
    cfg = HarnessConfig(harness_root=tmp_path / "capped", max_active_rules=1)
    journal = Journal(cfg)
    capped = HarnessToolset(
        config=cfg,
        cli=FakeCliClient(),
        mailbox=Mailbox(cfg),
        journal=journal,
        rules=RulesStore(cfg, journal=journal),
        history=ScoreHistoryStore(cfg),
    )

    first = await capped.call(
        "harness_rule_add", {"rule": "first rule", "rationale": "why"}
    )
    assert first["ok"] is True

    second = await capped.call(
        "harness_rule_add", {"rule": "second rule", "rationale": "why"}
    )
    assert second["ok"] is False
    assert "retire" in second["error"]


async def test_rule_retire_ok_and_unknown_is_error(toolset: HarnessToolset) -> None:
    added = await toolset.call(
        "harness_rule_add", {"rule": "some rule", "rationale": "why"}
    )
    rule_id = added["rule"]["id"]

    retired = await toolset.call("harness_rule_retire", {"rule_id": rule_id})
    assert retired["ok"] is True
    assert retired["rule"]["id"] == rule_id
    assert retired["rule"]["status"] == "retired"

    unknown = await toolset.call("harness_rule_retire", {"rule_id": "r-missing"})
    assert unknown["ok"] is False
    assert "r-missing" in unknown["error"]


# -- reflection ----------------------------------------------------------------------


async def test_reflect_assembles_context_and_journals(
    toolset: HarnessToolset, journal: Journal
) -> None:
    journal.record(
        {"type": "notice", "id": "n-1", "metrics": [{"name": "agent_reliability"}]}
    )
    added = await toolset.call(
        "harness_rule_add",
        {"rule": "reflect on failures", "rationale": "why", "metric": "agent_reliability"},
    )
    rule_id = added["rule"]["id"]

    result = await toolset.call("harness_reflect", {})

    assert result["ok"] is True
    assert [e["id"] for e in result["recent_notices"]] == ["n-1"]
    assert [r["id"] for r in result["active_rules"]] == [rule_id]
    assert rule_id in result["effectiveness"]
    assert result["effectiveness"][rule_id]["metric"] == "agent_reliability"

    reflections = journal.recent(types=("reflect",))
    assert len(reflections) == 1
    assert reflections[0]["notice_count"] == 1
    assert reflections[0]["active_rule_count"] == 1


# -- dispatch ------------------------------------------------------------------------


async def test_unknown_tool_is_error_envelope(toolset: HarnessToolset) -> None:
    result = await toolset.call("nope", {})
    assert result["ok"] is False
    assert "unknown tool" in result["error"]


async def test_missing_argument_never_raises(toolset: HarnessToolset) -> None:
    # ``notice_id`` is required; the KeyError inside the handler is enveloped.
    result = await toolset.call("harness_mailbox_read", {})
    assert result["ok"] is False
    assert "KeyError" in result["error"]
