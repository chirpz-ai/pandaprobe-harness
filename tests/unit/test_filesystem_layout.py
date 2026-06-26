from __future__ import annotations

from pandaprobe_harness import HarnessConfig, HarnessFilesystem


def test_provision_creates_tree_and_rules(config: HarnessConfig) -> None:
    fs = HarnessFilesystem(config)
    fs.provision()
    assert config.harness_root.is_dir()
    assert config.traces_dir.is_dir()
    assert config.rules_file.exists()
    assert "Learned Mitigations" in fs.read_rules()


def test_provision_idempotent_does_not_clobber(config: HarnessConfig) -> None:
    fs = HarnessFilesystem(config)
    fs.provision()
    fs.append_rule("a learned rule")
    before = fs.read_rules()
    fs.provision()  # second call must not overwrite
    assert fs.read_rules() == before
    assert "a learned rule" in fs.read_rules()


def test_append_rule_is_timestamped_and_attributed(config: HarnessConfig) -> None:
    fs = HarnessFilesystem(config)
    fs.provision()
    fs.append_rule("never double-charge", source="self-heal")
    rules = fs.read_rules()
    assert "never double-charge" in rules
    assert "(self-heal)" in rules


def test_write_latest_eval_atomic_roundtrip(config: HarnessConfig) -> None:
    fs = HarnessFilesystem(config)
    fs.provision()
    payload = {"session_id": "s-1", "any_breach": True, "scores": []}
    fs.write_latest_eval(payload)
    assert config.latest_eval_file.exists()
    # no leftover temp file
    assert not config.latest_eval_file.with_suffix(".json.tmp").exists()
    assert fs.read_latest_eval() == payload
