"""Study configuration: arms, seeds, k, splits, thresholds, harness knobs.

Loaded from ``configs/study.yaml``; nothing study-relevant is hardcoded. The
same threshold is used for every arm/seed of a benchmark (set once by
Checkpoint 1). Per-benchmark task universes/subset rules live in
``configs/benchmarks/*.yaml`` and are merged in on load.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["BenchmarkConfig", "HarnessKnobs", "SmokeConfig", "StudyConfig", "load_study"]


@dataclass(frozen=True, slots=True)
class HarnessKnobs:
    rule_trial_min_sessions: int = 3
    rule_promote_margin: float = 0.05
    rule_regress_margin: float = 0.05
    replay_timeout_s: float = 180.0
    replay_max_turns: int = 15
    regression_sample: int = 0
    # How long refresh() waits for platform session-eval scores per trial:
    # poll_interval_s * poll_max_attempts (bounded; benchmark sessions have many
    # traces so scoring can be slow — keep this generous).
    poll_interval_s: float = 3.0
    poll_max_attempts: int = 60


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    name: str
    max_turns: int
    dataset: str  # e.g. appworld 'dev', tau2 'retail'
    learning_fraction: float
    learning_split: tuple[str, ...]  # explicit ids override the seeded partition
    eval_split: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    model: str
    tasks: int
    k: int
    arms: tuple[str, ...]
    benchmarks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StudyConfig:
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    k: int
    harness: HarnessKnobs
    breach_thresholds: dict[str, float]
    benchmarks: dict[str, BenchmarkConfig]
    smoke: SmokeConfig
    cost_cap_usd: float | None = None

    def breach_threshold(self, benchmark: str) -> float:
        return self.breach_thresholds.get(benchmark, self.breach_thresholds.get("default", 0.5))

    def benchmark(self, name: str) -> BenchmarkConfig:
        try:
            return self.benchmarks[name]
        except KeyError:
            raise KeyError(f"no benchmark config for {name!r} in study.yaml") from None


def _benchmark_from(name: str, raw: Mapping[str, Any]) -> BenchmarkConfig:
    known = {"max_turns", "dataset", "learning_fraction", "learning_split", "eval_split"}
    return BenchmarkConfig(
        name=name,
        max_turns=int(raw.get("max_turns", 30)),
        dataset=str(raw.get("dataset", "")),
        learning_fraction=float(raw.get("learning_fraction", 0.35)),
        learning_split=tuple(str(t) for t in raw.get("learning_split", []) or []),
        eval_split=tuple(str(t) for t in raw.get("eval_split", []) or []),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def load_study(path: str | Path, *, benchmarks_dir: str | Path | None = None) -> StudyConfig:
    """Load study.yaml and merge per-benchmark configs/benchmarks/*.yaml."""

    study_path = Path(path)
    data = yaml.safe_load(study_path.read_text(encoding="utf-8")) or {}
    bench_dir = Path(benchmarks_dir) if benchmarks_dir else study_path.parent / "benchmarks"

    harness_raw = data.get("harness") or {}
    harness = HarnessKnobs(
        rule_trial_min_sessions=int(harness_raw.get("rule_trial_min_sessions", 3)),
        rule_promote_margin=float(harness_raw.get("rule_promote_margin", 0.05)),
        rule_regress_margin=float(harness_raw.get("rule_regress_margin", 0.05)),
        replay_timeout_s=float(harness_raw.get("replay_timeout_s", 180.0)),
        replay_max_turns=int(harness_raw.get("replay_max_turns", 15)),
        regression_sample=int(harness_raw.get("regression_sample", 0)),
        poll_interval_s=float(harness_raw.get("poll_interval_s", 3.0)),
        poll_max_attempts=int(harness_raw.get("poll_max_attempts", 60)),
    )

    benchmarks: dict[str, BenchmarkConfig] = {}
    for name, raw in (data.get("benchmarks") or {}).items():
        merged: dict[str, Any] = dict(raw or {})
        bench_file = bench_dir / f"{name}.yaml"
        if bench_file.exists():
            file_data = yaml.safe_load(bench_file.read_text(encoding="utf-8")) or {}
            merged = {**file_data, **merged}  # study.yaml overrides the per-benchmark file
        benchmarks[str(name)] = _benchmark_from(str(name), merged)

    smoke_raw = data.get("smoke") or {}
    smoke = SmokeConfig(
        model=str(smoke_raw.get("model", "gemini-3.1-flash-lite")),
        tasks=int(smoke_raw.get("tasks", 2)),
        k=int(smoke_raw.get("k", 1)),
        arms=tuple(str(a) for a in smoke_raw.get("arms", ["baseline", "harness"])),
        benchmarks=tuple(str(b) for b in smoke_raw.get("benchmarks", list(benchmarks))),
    )

    thresholds = {str(k): float(v) for k, v in (data.get("breach_thresholds") or {}).items()}
    thresholds.setdefault("default", 0.5)

    return StudyConfig(
        arms=tuple(str(a) for a in data.get("arms", ["baseline", "harness"])),
        seeds=tuple(int(s) for s in data.get("seeds", [1, 2, 3])),
        k=int(data.get("k", 4)),
        harness=harness,
        breach_thresholds=thresholds,
        benchmarks=benchmarks,
        smoke=smoke,
        cost_cap_usd=(
            float(data["cost_cap_usd"]) if data.get("cost_cap_usd") is not None else None
        ),
    )
