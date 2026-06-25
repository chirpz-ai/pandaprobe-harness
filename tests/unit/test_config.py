from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig


def test_derived_paths_from_root() -> None:
    cfg = HarnessConfig(harness_root=Path("/tmp/harness"))
    assert cfg.traces_dir == Path("/tmp/harness/traces")
    assert cfg.rules_file == Path("/tmp/harness/harness_rules.md")
    assert cfg.latest_eval_file == Path("/tmp/harness/traces/latest_eval.json")


def test_defaults() -> None:
    cfg = HarnessConfig()
    assert cfg.cli_binary == "pandaprobe"
    assert cfg.reliability_threshold == 0.5
    assert cfg.consistency_threshold == 0.5
    assert cfg.eval_reliability and cfg.eval_consistency and cfg.concurrent_eval


def test_from_env_reads_overrides(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_ROOT", "/srv/h")
    monkeypatch.setenv("HARNESS_RELIABILITY_THRESHOLD", "0.7")
    monkeypatch.setenv("HARNESS_EVAL_CONSISTENCY", "false")
    monkeypatch.setenv("HARNESS_POLL_MAX_ATTEMPTS", "3")
    cfg = HarnessConfig.from_env()
    assert cfg.harness_root == Path("/srv/h")
    assert cfg.reliability_threshold == 0.7
    assert cfg.eval_consistency is False
    assert cfg.poll_max_attempts == 3


def test_from_env_explicit_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_RELIABILITY_THRESHOLD", "0.7")
    cfg = HarnessConfig.from_env(reliability_threshold=0.2)
    assert cfg.reliability_threshold == 0.2


def test_frozen() -> None:
    import dataclasses

    import pytest

    cfg = HarnessConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.cli_binary = "other"  # type: ignore[misc]
