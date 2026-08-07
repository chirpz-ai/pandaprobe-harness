"""On-demand task access, compact indexing, and novelty suppression."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.agent_tools.toolset import RepairToolset, TaskToolset
from pandaprobe_harness.workspace.journal import Journal
from pandaprobe_harness.workspace.mailbox import DiagnosticNotice, Mailbox
from pandaprobe_harness.workspace.rules import RulesStore
from tests.fakes.fake_cli_client import FakeCliClient


def _workspace(tmp_path: Path) -> tuple[HarnessConfig, Journal, RulesStore]:
    config = HarnessConfig(harness_root=tmp_path / "h", repair_model="test/fake")
    journal = Journal(config)
    return config, journal, RulesStore(config, journal=journal)


async def test_compact_index_and_scope_reads_follow_live_lifecycle(tmp_path: Path) -> None:
    config, _, rules = _workspace(tmp_path)
    rules.register_scope_metadata(
        "spotify", "Spotify search, playlist, library, and playback workflows."
    )
    global_rule = rules.add("Verify irreversible writes.", "cross-domain", scope="global")
    rules.promote(global_rule.id, reason="fixture", validator="test")
    spotify = rules.add("Paginate liked songs.", "complete library", scope="spotify")
    venmo = rules.add("Confirm payment status.", "avoid duplicates", scope="venmo")
    rules.retire(venmo.id)
    tools = TaskToolset(config=config, rules=rules)

    listed = await tools.call("harness_rules_list", {})
    assert [entry["scope"] for entry in listed["scopes"]] == ["global", "spotify"]
    assert listed["scopes"][0]["active"] == 1
    assert listed["scopes"][1]["provisional"] == 1
    assert listed["scopes"][1]["description"].startswith("Spotify search")
    assert listed["path"] == "harness_guide.md"
    assert listed["content"].startswith("---\nname: pandaprobe-learned-rules")
    assert "allowed-tools:" in listed["content"]
    assert "## Workflow" in listed["content"]
    assert global_rule.rule not in listed["content"]
    assert spotify.rule not in listed["content"]
    assert venmo.rule not in listed["content"]

    spotify_read = await tools.call("harness_rules_read", {"scope": "spotify"})
    assert spotify.rule in spotify_read["content"]
    assert "candidate" in spotify_read["content"].casefold()
    assert global_rule.rule not in spotify_read["content"]
    assert venmo.rule not in spotify_read["content"]

    rules.promote(spotify.id, reason="fixture", validator="test")
    refreshed = await tools.call("harness_rules_list", {})
    assert refreshed["scopes"][1]["active"] == 1
    assert refreshed["scopes"][1]["provisional"] == 0


async def test_task_paths_and_search_are_bounded_and_live_only(tmp_path: Path) -> None:
    config, _, rules = _workspace(tmp_path)
    for index in range(12):
        rules.add(f"Spotify playlist guidance {index}.", "x", scope="spotify")
    retired = rules.add("Retired Venmo secret body.", "x", scope="venmo")
    rules.retire(retired.id)
    tools = TaskToolset(config=config, rules=rules)

    for hostile in ("../mailbox", "/etc/passwd", "spotify/../../state", "Spotify"):
        result = await tools.call("harness_rules_read", {"scope": hostile})
        assert result["ok"] is False
    search = await tools.call(
        "harness_rules_search", {"query": "playlist", "limit": 3, "status": "retired"}
    )
    assert len(search["rules"]) == 3
    assert all(rule["status"] == "candidate" for rule in search["rules"])
    assert all(len(rule["snippet"]) <= 240 for rule in search["rules"])
    assert retired.id not in {rule["id"] for rule in search["rules"]}


async def test_descriptions_are_bounded_and_list_reads_fresh_store_state(
    tmp_path: Path,
) -> None:
    config, journal, rules = _workspace(tmp_path)
    rules.register_scope_metadata("spotify", "Spotify " + "library " * 100)
    spotify = rules.add("Paginate the saved library.", "complete results", scope="spotify")
    tools = TaskToolset(config=config, rules=rules)

    first = await tools.call("harness_rules_list", {})
    assert len(first["scopes"][0]["description"]) <= 161
    assert spotify.rule not in first["content"]

    # Simulate a later writer in the same persisted workspace. Task-facing list
    # renders from JSONL rather than caching a stale file/index snapshot.
    later = RulesStore(config, journal=journal)
    later.add("Verify the transfer result.", "avoid duplicates", scope="venmo")
    refreshed = await tools.call("harness_rules_list", {})
    assert [entry["scope"] for entry in refreshed["scopes"]] == ["spotify", "venmo"]


def test_legacy_workspace_regenerates_index_without_scope_migration(tmp_path: Path) -> None:
    config, _, rules = _workspace(tmp_path)
    config.rules_store_file.parent.mkdir(parents=True, exist_ok=True)
    config.rules_store_file.write_text(
        json.dumps(
            {
                "id": "r-old",
                "created_at": "2026-01-01T00:00:00+00:00",
                "rule": "Old custom guidance.",
                "rationale": "legacy",
                "status": "active",
                "scope": "custom-scope",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rules.sync_markdown()

    assert "[`custom-scope`](rules/custom-scope.md)" in config.rules_file.read_text(
        encoding="utf-8"
    )
    assert "Old custom guidance." in config.rules_scope_file("custom-scope").read_text(
        encoding="utf-8"
    )


async def test_similar_provisional_candidate_is_suppressed_at_write_boundary(
    tmp_path: Path,
) -> None:
    config, journal, rules = _workspace(tmp_path)
    mailbox = Mailbox(config)
    mailbox.provision()
    notice = DiagnosticNotice.from_json(
        {
            "id": "n-similar",
            "created_at": "2026-08-06T00:00:00+00:00",
            "session_id": "s",
            "turn_index": 1,
            "severity": "breach",
            "signatures": ["breach:tool_correctness"],
            "recommended_scope": "venmo",
            "scope_hints": [
                {
                    "key": "venmo",
                    "description": "Venmo payment and reminder workflows.",
                    "applicability": "topical",
                    "recommended": True,
                }
            ],
        }
    )
    mailbox.post(notice)
    existing = rules.add(
        "Before retrying a Venmo payment, check its transaction status.",
        "avoid duplicate payments",
        scope="venmo",
        tags=("venmo", "payment", "retry"),
        failure_signatures=notice.signatures,
    )
    repair = RepairToolset(
        config=config,
        cli=FakeCliClient(),
        mailbox=mailbox,
        journal=journal,
        rules=rules,
        notice_id=notice.id,
        recommended_scope="venmo",
        scope_hints=notice.scope_hints,
        allowed_trace_ids=(),
    )

    result = await repair.call(
        "harness_rule_add",
        {
            "rule": "Check the Venmo transaction status before retrying the payment.",
            "rationale": "prevent duplicate payments",
            "tags": ["venmo", "payment", "retry"],
        },
    )

    assert result["created"] is False
    assert result["recommended_resolution"] == "already_covered"
    assert result["existing_rule"]["id"] == existing.id
    assert len(rules.candidates()) == 1


async def test_similar_active_rule_is_reported_as_duplicate(tmp_path: Path) -> None:
    config, journal, rules = _workspace(tmp_path)
    mailbox = Mailbox(config)
    mailbox.provision()
    notice = DiagnosticNotice.from_json(
        {
            "id": "n-active",
            "created_at": "2026-08-06T00:00:00+00:00",
            "session_id": "s",
            "turn_index": 1,
            "severity": "breach",
            "signatures": ["breach:tool_correctness"],
            "recommended_scope": "venmo",
        }
    )
    mailbox.post(notice)
    existing = rules.add(
        "Before retrying a Venmo payment, check its transaction status.",
        "avoid duplicate payments",
        scope="venmo",
        tags=("venmo", "payment", "retry"),
        failure_signatures=notice.signatures,
    )
    rules.promote(existing.id, reason="fixture", validator="test")
    repair = RepairToolset(
        config=config,
        cli=FakeCliClient(),
        mailbox=mailbox,
        journal=journal,
        rules=rules,
        notice_id=notice.id,
        recommended_scope="venmo",
        allowed_trace_ids=(),
    )

    result = await repair.call(
        "harness_rule_add",
        {
            "rule": "Check the Venmo transaction status before retrying the payment.",
            "rationale": "prevent duplicate payments",
            "tags": ["venmo", "payment", "retry"],
        },
    )

    assert result["created"] is False
    assert result["recommended_resolution"] == "duplicate"
    assert result["existing_rule"]["id"] == existing.id


async def test_concurrent_repair_writes_cannot_create_exact_duplicates(
    tmp_path: Path,
) -> None:
    config, journal, rules = _workspace(tmp_path)
    mailbox = Mailbox(config)
    mailbox.provision()
    repairs: list[RepairToolset] = []
    for index in range(2):
        notice = DiagnosticNotice.from_json(
            {
                "id": f"n-{index}",
                "created_at": f"2026-08-06T00:00:0{index}+00:00",
                "session_id": f"s-{index}",
                "turn_index": 1,
                "severity": "breach",
                "signatures": ["breach:tool_correctness"],
                "recommended_scope": "venmo",
            }
        )
        mailbox.post(notice)
        repairs.append(
            RepairToolset(
                config=config,
                cli=FakeCliClient(),
                mailbox=mailbox,
                journal=journal,
                rules=rules,
                notice_id=notice.id,
                recommended_scope="venmo",
                allowed_trace_ids=(),
            )
        )

    results = await asyncio.gather(
        *(
            repair.call(
                "harness_rule_add",
                {
                    "rule": "Check Venmo payment status before retrying.",
                    "rationale": "prevent duplicate payments",
                },
            )
            for repair in repairs
        )
    )

    assert sorted(result["created"] for result in results) == [False, True]
    assert len(rules.candidates()) == 1
