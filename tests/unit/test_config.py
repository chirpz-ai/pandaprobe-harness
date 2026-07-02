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


def test_derived_workspace_paths() -> None:
    cfg = HarnessConfig(harness_root=Path("/h"))
    assert cfg.mailbox_dir == Path("/h/mailbox")
    assert cfg.mailbox_pending_dir == Path("/h/mailbox/pending")
    assert cfg.mailbox_processed_dir == Path("/h/mailbox/processed")
    assert cfg.mailbox_status_file == Path("/h/mailbox/status.json")
    assert cfg.journal_file == Path("/h/journal.jsonl")
    assert cfg.rules_store_file == Path("/h/rules.jsonl")


def test_control_defaults() -> None:
    cfg = HarnessConfig()
    assert cfg.eval_sample_every == 1
    assert cfg.session_min_eval_interval_s == 0.0
    assert cfg.max_concurrent_evals == 4
    assert cfg.max_evals_per_run == 0
    assert cfg.observe_only is False
    assert cfg.circuit_breaker_max_notices == 5
    assert cfg.circuit_breaker_window_s == 600.0
    assert cfg.max_active_rules == 50
    assert cfg.health_check is True
    assert cfg.hydrate_history_from_backend is False


def test_from_env_reads_control_knobs(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_EVAL_SAMPLE_EVERY", "5")
    monkeypatch.setenv("HARNESS_MAX_CONCURRENT_EVALS", "2")
    monkeypatch.setenv("HARNESS_OBSERVE_ONLY", "true")
    monkeypatch.setenv("HARNESS_CIRCUIT_BREAKER_MAX_NOTICES", "9")
    monkeypatch.setenv("HARNESS_HEALTH_CHECK", "false")
    monkeypatch.setenv("HARNESS_HYDRATE_HISTORY_FROM_BACKEND", "1")
    cfg = HarnessConfig.from_env()
    assert cfg.eval_sample_every == 5
    assert cfg.max_concurrent_evals == 2
    assert cfg.observe_only is True
    assert cfg.circuit_breaker_max_notices == 9
    assert cfg.health_check is False
    assert cfg.hydrate_history_from_backend is True


def test_frozen() -> None:
    import dataclasses

    import pytest

    cfg = HarnessConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.cli_binary = "other"  # type: ignore[misc]
