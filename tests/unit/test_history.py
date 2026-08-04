from __future__ import annotations

import json
from pathlib import Path

from pandaprobe_harness import GateState, HarnessConfig
from pandaprobe_harness.evaluation.history import ScoreHistoryStore


def _store(tmp_path: Path) -> tuple[ScoreHistoryStore, HarnessConfig]:
    cfg = HarnessConfig(harness_root=tmp_path / "h")
    return ScoreHistoryStore(cfg), cfg


def _record(
    store: ScoreHistoryStore, session: str, metric: str, value: float
) -> GateState:
    states = store.record_gated(
        session,
        [
            (
                metric,
                value,
                lambda state: GateState(
                    peak=value, turns_since_gain=state.turns_since_gain + 1
                ),
            )
        ],
    )
    return states[metric]


def test_record_gated_appends_series_and_folds_state(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    first = _record(store, "s", "task_completion", 0.8)
    second = _record(store, "s", "task_completion", 0.4)

    assert first == GateState(peak=0.8, turns_since_gain=1)
    assert second == GateState(peak=0.4, turns_since_gain=2)
    assert store.values("s", "task_completion") == [0.8, 0.4]


def test_persistence_across_instances(tmp_path: Path) -> None:
    store, cfg = _store(tmp_path)
    _record(store, "s", "task_completion", 0.7)

    reopened = ScoreHistoryStore(cfg)
    assert reopened.values("s", "task_completion") == [0.7]


def test_values_empty_before_any_record(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    assert store.values("s", "task_completion") == []


def test_atomic_write_leaves_no_temp(tmp_path: Path) -> None:
    store, cfg = _store(tmp_path)
    _record(store, "s", "task_completion", 0.5)
    assert cfg.history_file.exists()
    assert list(cfg.history_file.parent.glob("*.tmp")) == []


def test_separate_keys_per_metric(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    _record(store, "s", "task_completion", 0.5)
    _record(store, "s", "coherence", 0.9)
    assert store.values("s", "task_completion") == [0.5]
    assert store.values("s", "coherence") == [0.9]


def test_legacy_ewma_state_is_ignored_and_removed_on_next_write(tmp_path: Path) -> None:
    store, cfg = _store(tmp_path)
    cfg.history_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.history_file.write_text(
        json.dumps(
            {
                "s::task_completion": {
                    "series": [{"value": 0.6, "ts": "t", "run_id": "r1"}],
                    "ewma": {"fast": 0.6, "slow": 0.7, "count": 3},
                    "gate": {"peak": 0.7, "turns_since_gain": 1},
                }
            }
        ),
        encoding="utf-8",
    )

    assert store.values("s", "task_completion") == [0.6]
    _record(store, "s", "task_completion", 0.8)

    persisted = json.loads(cfg.history_file.read_text(encoding="utf-8"))
    assert "ewma" not in persisted["s::task_completion"]
    assert store.values("s", "task_completion") == [0.6, 0.8]
