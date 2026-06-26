from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.evaluation.history import ScoreHistoryStore


def _store(tmp_path: Path, **kw: object) -> tuple[ScoreHistoryStore, HarnessConfig]:
    cfg = HarnessConfig(harness_root=tmp_path / "h", **kw)  # type: ignore[arg-type]
    return ScoreHistoryStore(cfg), cfg


def test_record_seeds_then_moves_ewma(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    s1 = store.record("s", "m", 0.8)
    assert s1.count == 1 and s1.fast == 0.8 and s1.slow == 0.8
    s2 = store.record("s", "m", 0.4)
    assert s2.count == 2
    assert s2.fast < s1.fast  # moved toward the lower value
    assert store.values("s", "m") == [0.8, 0.4]


def test_persistence_across_instances(tmp_path: Path) -> None:
    store, cfg = _store(tmp_path)
    store.record("s", "m", 0.7, run_id="r1")
    reopened = ScoreHistoryStore(cfg)
    assert reopened.values("s", "m") == [0.7]
    state = reopened.ewma("s", "m")
    assert state is not None and state.count == 1


def test_ewma_none_before_any_record(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    assert store.ewma("s", "m") is None
    assert store.values("s", "m") == []


def test_atomic_write_leaves_no_temp(tmp_path: Path) -> None:
    store, cfg = _store(tmp_path)
    store.record("s", "m", 0.5)
    assert cfg.history_file.exists()
    assert not cfg.history_file.with_suffix(".json.tmp").exists()


def test_separate_keys_per_metric(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    store.record("s", "agent_reliability", 0.5)
    store.record("s", "agent_consistency", 0.9)
    assert store.values("s", "agent_reliability") == [0.5]
    assert store.values("s", "agent_consistency") == [0.9]
