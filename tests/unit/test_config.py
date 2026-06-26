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


def test_active_metrics_default_and_selective() -> None:
    cfg = HarnessConfig()
    assert set(cfg.active_metrics()) == {"agent_reliability", "agent_consistency"}
    only_rel = HarnessConfig(eval_consistency=False)
    assert only_rel.active_metrics() == ("agent_reliability",)


def test_threshold_resolution() -> None:
    cfg = HarnessConfig(
        reliability_threshold=0.6,
        thresholds={"agent_consistency": 0.4},
    )
    assert cfg.threshold_for("agent_reliability") == 0.6  # scalar fallback
    assert cfg.threshold_for("agent_consistency") == 0.4  # per-metric map wins
    assert cfg.threshold_for("unknown_metric") == 0.5  # default


def test_derived_state_paths() -> None:
    cfg = HarnessConfig(harness_root=Path("/h"))
    assert cfg.state_dir == Path("/h/state")
    assert cfg.history_file == Path("/h/state/score_history.json")


def test_frozen() -> None:
    import dataclasses

    import pytest

    cfg = HarnessConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.cli_binary = "other"  # type: ignore[misc]
