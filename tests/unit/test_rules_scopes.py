"""The ``rules/`` subtree: scope routing, the skill root, and migration."""

from __future__ import annotations

import json
from pathlib import Path

from pandaprobe_harness import HarnessConfig, Journal, RulesStore
from pandaprobe_harness.workspace.rules import (
    GLOBAL_SCOPE,
    SCOPED_SCOPE,
    Rule,
    normalize_scope,
)


def _store(tmp_path: Path, **kw: object) -> RulesStore:
    config = HarnessConfig(
        harness_root=tmp_path / "harness",
        rule_validation=False,  # adds are active immediately; scoping is the subject
        **kw,  # type: ignore[arg-type]
    )
    config.rules_dir.mkdir(parents=True, exist_ok=True)
    return RulesStore(config, journal=Journal(config))


# -- scope normalization -------------------------------------------------------


def test_empty_scope_defaults_to_scoped() -> None:
    assert normalize_scope(None) == SCOPED_SCOPE
    assert normalize_scope("") == SCOPED_SCOPE
    assert normalize_scope("   ") == SCOPED_SCOPE


def test_scope_is_slugified_not_trusted() -> None:
    assert normalize_scope("Payment Flows") == "payment-flows"
    assert normalize_scope("  TOOL_USE  ") == "tool_use"


def test_scope_cannot_escape_the_rules_directory() -> None:
    """Scope becomes a filename, so a traversal attempt must collapse to a single
    safe component rather than reaching outside ``rules/``."""

    for hostile in ("../../etc/passwd", "..", ".", "/etc/passwd", "a/../../b"):
        slug = normalize_scope(hostile)
        assert "/" not in slug
        assert slug not in {".", ".."}


def test_unslugifiable_scope_falls_back_to_the_catch_all() -> None:
    # No usable characters at all: the rule still has to land somewhere.
    assert normalize_scope("///") == SCOPED_SCOPE
    assert normalize_scope("!!!") == SCOPED_SCOPE


# -- routing -------------------------------------------------------------------


def test_rules_route_to_their_scope_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cfg = store._config  # noqa: SLF001 - reading the derived paths under test

    store.add("read state first", "global lesson", scope="global")
    store.add("check the amount", "step lesson", scope="scoped")
    store.add("never double-charge", "topic lesson", scope="payments")

    assert "read state first" in cfg.rules_scope_file("global").read_text(encoding="utf-8")
    assert "check the amount" in cfg.rules_scope_file("scoped").read_text(encoding="utf-8")
    assert "never double-charge" in cfg.rules_scope_file("payments").read_text(
        encoding="utf-8"
    )
    # No cross-contamination between files.
    assert "never double-charge" not in cfg.rules_scope_file("global").read_text(
        encoding="utf-8"
    )


def test_an_agent_created_scope_appears_in_the_references(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("settle before refunding", "why", scope="payments")

    root = store.render_root()

    assert "rules/payments.md" in root
    assert "1 active" in root


def test_scopes_are_ordered_global_then_all_others_alphabetically(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("z topic rule", "x", scope="zebra")
    store.add("a topic rule", "x", scope="alpha")
    store.add("default granular rule", "x", scope="scoped")
    store.add("global rule", "x", scope="global")

    assert store.scopes() == [GLOBAL_SCOPE, "alpha", SCOPED_SCOPE, "zebra"]


def test_a_scope_whose_rules_all_retired_is_emptied_not_left_stale(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    rule = store.add("temporary lesson", "x", scope="payments")
    path = store._config.rules_scope_file("payments")  # noqa: SLF001
    assert "temporary lesson" in path.read_text(encoding="utf-8")

    store.retire(rule.id)

    # The file survives (the agent may have referenced the path) but says so.
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "temporary lesson" not in content
    assert "_No learned rules yet._" in content
    assert "rules/payments.md" not in store.render_root()


def test_the_skill_root_never_carries_rule_text(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for scope in ("global", "scoped", "payments"):
        store.add(f"secret {scope} body", "x", scope=scope)

    root = store.render_root()

    for scope in ("global", "scoped", "payments"):
        assert f"rules/{scope}.md" in root
        assert f"secret {scope} body" not in root


# -- migration -----------------------------------------------------------------


def test_a_v1_record_without_a_scope_migrates_by_its_tags(tmp_path: Path) -> None:
    """v1 had no `scope` but did treat an *untagged* rule as global. Preserve that
    reading so an existing workspace keeps its meaning."""

    store = _store(tmp_path)
    path = store._config.rules_store_file  # noqa: SLF001
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = [
        {"id": "r-untagged", "created_at": "2026-01-01T00:00:00+00:00",
         "rule": "old global", "rationale": "x", "status": "active", "tags": []},
        {"id": "r-tagged", "created_at": "2026-01-02T00:00:00+00:00",
         "rule": "old scoped", "rationale": "x", "status": "active",
         "tags": ["breach:tool_correctness"]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in legacy) + "\n", encoding="utf-8")

    by_id = {rule.id: rule for rule in store.all()}

    assert by_id["r-untagged"].scope == GLOBAL_SCOPE
    assert by_id["r-tagged"].scope == SCOPED_SCOPE


def test_scope_round_trips_through_json() -> None:
    rule = Rule(
        id="r-1", created_at="2026-01-01T00:00:00+00:00", rule="r", rationale="x",
        scope="payments",
    )
    assert Rule.from_json(rule.to_json()).scope == "payments"


def test_syncing_materializes_the_whole_subtree(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add("global rule", "x", scope="global")
    store.add("topic rule", "x", scope="payments")

    store.sync_markdown()

    cfg = store._config  # noqa: SLF001
    written = sorted(path.name for path in cfg.rules_dir.glob("*.md"))
    assert written == ["global.md", "payments.md"]
    assert cfg.rules_file.is_file()
