"""Offline coverage for benchmark-only configuration and frozen eval rules."""

from __future__ import annotations

import json
import tomllib
from importlib.metadata import distribution
from pathlib import Path

import pytest
from pandaprobe_harness import HarnessConfig

from pandabench.agents.frozen_wiring import FrozenEvalWiring
from pandabench.config import HarnessKnobs, load_study
from pandabench.frozen_rules import FrozenRulesSnapshot
from pandabench.harness_glue import build_harness_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BENCH_ROOT = CONFIGS.parent


def _rules() -> list[dict[str, object]]:
    return [
        {
            "id": "r-candidate",
            "created_at": "2026-08-05T02:00:00+00:00",
            "rule": "Confirm the account before changing it.",
            "rationale": "A learning trace changed the wrong account.",
            "source_notice_id": "n-2",
            "metric": "tool_correctness",
            "status": "candidate",
            "tags": ["account", "change"],
            "scope": "accounts",
            "trial": {"observed_sessions": ["s-1"], "verdict": "pending"},
        },
        {
            "id": "r-active",
            "created_at": "2026-08-05T01:00:00+00:00",
            "rule": "Read the order before issuing a refund.",
            "rationale": "The order state determines refund eligibility.",
            "source_notice_id": "n-1",
            "metric": "task_completion",
            "status": "active",
            "tags": ["order", "refund"],
            "scope": "global",
            "trial": None,
        },
        {
            "id": "r-retired",
            "created_at": "2026-08-05T03:00:00+00:00",
            "rule": "Use the legacy refund endpoint.",
            "rationale": "Retired after regression.",
            "source_notice_id": "n-3",
            "metric": "task_completion",
            "status": "retired",
            "tags": ["legacy"],
            "scope": "global",
            "trial": {"verdict": "retired"},
        },
    ]


def test_benchmark_gate_window_defaults_and_installed_package_default(tmp_path: Path) -> None:
    study = load_study(CONFIGS / "study.yaml")
    assert study.harness.gate_window == 10
    assert HarnessKnobs().gate_window == 10

    cfg = build_harness_config(
        harness_root=tmp_path / "harness", phase="learning", study=study,
        benchmark="appworld", repair_model="mock/mock",
    )
    assert cfg.gate_window == 10
    assert cfg.repair_model == "mock/mock"
    assert cfg.repair_reasoning_effort == "none"
    assert cfg.rule_validation is True
    assert cfg.trace_repair_agent is True
    # The benchmark override must not leak into the exact-pinned released package.
    assert HarnessConfig().gate_window == 5


def test_benchmark_installs_candidate_from_local_built_wheel() -> None:
    project = tomllib.loads((BENCH_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    source = project["tool"]["uv"]["sources"]["pandaprobe-harness"]["path"]
    assert source == "../dist/pandaprobe_harness-0.8.0-py3-none-any.whl"

    direct_url = json.loads(distribution("pandaprobe-harness").read_text("direct_url.json") or "{}")
    assert direct_url["url"].endswith("/dist/pandaprobe_harness-0.8.0-py3-none-any.whl")


def test_study_can_select_a_dedicated_repair_model(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    path.write_text(
        "harness:\n  repair_model: anthropic/test-repair\n",
        encoding="utf-8",
    )
    study = load_study(path)
    cfg = build_harness_config(
        harness_root=tmp_path / "harness",
        phase="learning",
        study=study,
        benchmark="appworld",
        repair_model="openai/task-model",
    )
    assert cfg.repair_model == "anthropic/test-repair"


def test_study_gate_window_can_be_explicitly_overridden(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    path.write_text("harness:\n  gate_window: 17\n", encoding="utf-8")
    study = load_study(path)
    assert study.harness.gate_window == 17
    cfg = build_harness_config(
        harness_root=tmp_path / "harness", phase="learning", study=study,
        benchmark="appworld", repair_model="mock/mock",
    )
    assert cfg.gate_window == 17


def test_snapshot_is_deterministic_immutable_and_preserves_lifecycle(tmp_path: Path) -> None:
    rules = _rules()
    timestamp = "2026-08-05T04:00:00+00:00"
    first = FrozenRulesSnapshot.create(reversed(rules), created_at=timestamp)
    second = FrozenRulesSnapshot.create(rules, created_at=timestamp)

    assert first.to_dict() == second.to_dict()
    assert first.sha256 == second.sha256
    assert [rule["id"] for rule in first.rules] == [
        "r-active", "r-candidate", "r-retired"
    ]
    assert (first.active_count, first.candidate_count, first.retired_count) == (1, 1, 1)
    assert first.rules[1]["trial"] == {
        "observed_sessions": ["s-1"], "verdict": "pending"
    }

    path = tmp_path / "frozen-rules.json"
    first.save(path)
    assert FrozenRulesSnapshot.load(path).to_dict() == first.to_dict()

    # Neither a later workspace write nor mutation of a returned copy reaches
    # the canonical records held by the snapshot.
    rules[1]["rule"] = "MUTATED LIVE WORKSPACE"
    detached = first.rules[0]
    detached["rule"] = "MUTATED RETURN VALUE"
    assert "MUTATED" not in json.dumps(first.to_dict())


def test_snapshot_rejects_hash_mismatch_and_accepts_explicit_empty(tmp_path: Path) -> None:
    empty = FrozenRulesSnapshot.create((), created_at="2026-08-05T04:00:00+00:00")
    assert empty.rules == ()
    assert empty.active_count == empty.candidate_count == empty.retired_count == 0

    path = tmp_path / "frozen-rules.json"
    empty.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["active_count"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenRulesSnapshot.load(path)


async def test_frozen_wiring_is_read_only_and_keeps_live_rules_retrievable() -> None:
    snapshot = FrozenRulesSnapshot.create(
        _rules(), created_at="2026-08-05T04:00:00+00:00"
    )
    wiring = FrozenEvalWiring(snapshot)

    preamble = wiring.system_preamble().lower()
    assert "frozen learned rules are available" in preamble
    assert "does not automatically insert rule contents" in preamble
    # Product terminology: rules are not "optional guidance".
    assert "optional" not in preamble
    assert "guidance" not in preamble
    assert "rules/global.md" not in preamble
    assert "Read the order" not in preamble
    assert "mailbox" not in preamble
    assert "diagnostic" not in preamble
    assert "acknowledge" not in preamble
    assert "write rules" not in preamble
    assert wiring.pending_notice_ids(session_id="anything") == ()
    assert wiring.settles_turns is False

    tool_names = {
        schema["function"]["name"] for schema in wiring.harness_tools()
    }
    assert tool_names == {
        "harness_rules_read", "harness_rules_search",
        "harness_rules_list", "harness_rule_status",
    }
    assert "harness_rule_add" not in tool_names
    assert "harness_rule_retire" not in tool_names

    read = await wiring.dispatch("harness_rules_read", {"scope": "global"})
    assert "Read the order" in read["content"]
    assert "legacy refund endpoint" not in read["content"]
    candidate = await wiring.dispatch("harness_rules_read", {"scope": "accounts"})
    assert "provisional" in candidate["content"].lower()
    assert "Confirm the account" in candidate["content"]

    search = await wiring.dispatch("harness_rules_search", {"query": "refund order"})
    assert search["rules"][0]["id"] == "r-active"
    status = await wiring.dispatch("harness_rule_status", {"rule_id": "r-candidate"})
    assert status["lifecycle"]["status"] == "candidate"
    index = await wiring.dispatch("harness_rules_list", {})
    assert index["scopes"][0]["scope"] == "global"
    assert "Read the order" not in index["content"]
    rejected = await wiring.dispatch(
        "harness_rule_add", {"rule": "mutate", "rationale": "should fail"}
    )
    assert rejected["ok"] is False
    assert "unavailable" in rejected["error"]
    await wiring.settle_turn(1)  # compatibility no-op, no live Harness exists
