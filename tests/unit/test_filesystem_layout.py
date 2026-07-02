from __future__ import annotations

from pandaprobe_harness import HarnessConfig, HarnessFilesystem


def test_provision_creates_tree_and_rules(config: HarnessConfig) -> None:
    fs = HarnessFilesystem(config)
    fs.provision()
    assert config.harness_root.is_dir()
    assert config.traces_dir.is_dir()
    assert config.state_dir.is_dir()
    assert config.rules_file.exists()
    assert "Learned Mitigations" in fs.read_rules()


def test_provision_idempotent_does_not_clobber(config: HarnessConfig) -> None:
    fs = HarnessFilesystem(config)
    fs.provision()
    # Simulate accumulated learned content (the RulesStore renders this file).
    config.rules_file.write_text(
        fs.read_rules() + "\n- **r-1**: a learned rule\n", encoding="utf-8"
    )
    before = fs.read_rules()
    fs.provision()  # second call must not overwrite
    assert fs.read_rules() == before
    assert "a learned rule" in fs.read_rules()


def test_write_latest_eval_atomic_roundtrip(config: HarnessConfig) -> None:
    fs = HarnessFilesystem(config)
    fs.provision()
    payload = {"session_id": "s-1", "any_breach": True, "scores": []}
    fs.write_latest_eval(payload)
    assert config.latest_eval_file.exists()
    # no leftover temp files
    assert not list(config.traces_dir.glob("*.tmp"))
    assert fs.read_latest_eval() == payload


def test_trace_dump_roundtrip(config: HarnessConfig) -> None:
    fs = HarnessFilesystem(config)
    fs.provision()
    payload = {"session_id": "s-1", "any_breach": True}
    path = fs.write_trace_dump("n-123", payload)
    assert path == config.traces_dir / "n-123.json"
    assert fs.read_trace_dump("n-123") == payload
    assert fs.read_trace_dump("n-missing") is None
    assert not list(config.traces_dir.glob("*.tmp"))
