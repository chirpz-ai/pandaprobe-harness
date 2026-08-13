"""Study configuration: arms, seeds, k, datasets, thresholds, harness knobs.

Loaded from ``configs/study.yaml``; nothing study-relevant is hardcoded. The
same threshold is used for every arm/seed of a benchmark (set once by
Checkpoint 1). Per-benchmark task universes live in
``configs/benchmarks/*.yaml`` and are merged in on load.

A benchmark's dataset is run whole, so nothing here subdivides a task set.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["BenchmarkConfig", "HarnessKnobs", "SmokeConfig", "StudyConfig", "load_study"]

#: Non-improving evaluated trace updates before a Tier-1 STALL. Short relative to a
#: benchmark episode by design: AppWorld averaged 11.6 turns, and any gain above
#: gate_gain resets the counter, so a long window makes the STALL branch unreachable.
#: A module constant rather than a class attribute because ``HarnessKnobs`` uses
#: ``slots=True``, where ``HarnessKnobs.gate_window`` is a descriptor, not the default.
DEFAULT_GATE_WINDOW = 10


@dataclass(frozen=True, slots=True)
class HarnessKnobs:
    rule_trial_min_sessions: int = 3
    rule_promote_margin: float = 0.05
    rule_regress_margin: float = 0.05
    replay_timeout_s: float = 180.0
    replay_max_turns: int = 15
    # AppWorld serializes every task lifecycle behind one world lock, so a
    # background replay can queue behind a live trial for minutes. This is the
    # grace to reach the starting line; replay_timeout_s then bounds the run
    # itself. Without the split, queueing consumes the run budget and the replay
    # is scored as inconclusive evidence about a rule that never executed.
    replay_env_wait_timeout_s: float = 900.0
    # Wall-clock bound on replay work in one validation round. Past it, remaining
    # candidates get the cheap forward-trial verdict instead of waiting for a
    # replay slot — candidates accrue faster than sequential replays retire them,
    # and a candidate with no verdict at the learning boundary is a wasted trial.
    validation_round_budget_s: float = 600.0
    regression_sample: int = 0
    # Background eval poll budget (poll_interval_s * poll_max_attempts). Benchmark
    # trace evals are LLM-judged and can take 6-12 min, so this
    # must exceed that or scores/notices never land.
    poll_interval_s: float = 5.0
    poll_max_attempts: int = 200
    # Settle barrier: once at the end of a run (before archiving) wait for
    # outstanding turn evals + candidate-rule validation to drain, so a candidate
    # that earned a verdict gets one. Bounded; breaks early.
    settle_timeout_s: float = 1080.0
    settle_poll_s: float = 10.0
    gate_window: int = DEFAULT_GATE_WINDOW
    enable_tier3: bool = False
    # The per-turn evaluation + managed-repair barrier's budget. Must exceed one turn's
    # trace evals to land (poll_interval_s * poll_max_attempts bounds that), or the
    # barrier gives up before the diagnosis arrives and healing goes back to being
    # after-the-fact.
    barrier_timeout_s: float = 1080.0
    outcome_threshold: float = 0.9
    # Managed repair is package-owned. ``None`` means the benchmark explicitly
    # reuses the resolved task-model identifier; a value selects a dedicated
    # LiteLLM model through the same PandaProbe wrapper path.
    repair_model: str | None = None
    repair_timeout_s: float = 60.0
    repair_max_turns: int = 6
    repair_max_tokens: int = 4096
    repair_temperature: float | None = None
    repair_reasoning_effort: str | None = None
    trace_repair_agent: bool = True


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    name: str
    max_turns: int
    dataset: str  # e.g. appworld 'dev', tau2 'retail'
    # Replay budget in this benchmark's own ``max_turns`` unit. ``None`` falls back
    # to ``HarnessKnobs.replay_max_turns``; set it whenever the unit is not an
    # agent turn, or a replay is cut off before it can reproduce the failure.
    replay_max_turns: int | None = None
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

    def replay_max_turns(self, benchmark: str) -> int:
        """Replay budget for ``benchmark``, in that benchmark's own turn unit.

        The global knob is a default, not a universal: ``max_turns`` means whatever
        the benchmark's runner passes it to, and tau2 forwards it as ``max_steps``
        (message hops). A single global value in agent-turn units silently truncated
        tau2 replays to roughly a quarter of a median episode, so every candidate
        rule was judged on a run that could not reach the behaviour under test.
        """

        override = self.benchmarks.get(benchmark)
        if override is not None and override.replay_max_turns is not None:
            return override.replay_max_turns
        return self.harness.replay_max_turns


def _benchmark_from(name: str, raw: Mapping[str, Any]) -> BenchmarkConfig:
    known = {"max_turns", "dataset", "replay_max_turns"}
    replay_raw = raw.get("replay_max_turns")
    return BenchmarkConfig(
        name=name,
        max_turns=int(raw.get("max_turns", 30)),
        dataset=str(raw.get("dataset", "")),
        replay_max_turns=int(replay_raw) if replay_raw is not None else None,
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
        replay_env_wait_timeout_s=float(
            harness_raw.get("replay_env_wait_timeout_s", 900.0)
        ),
        validation_round_budget_s=float(
            harness_raw.get("validation_round_budget_s", 600.0)
        ),
        regression_sample=int(harness_raw.get("regression_sample", 0)),
        poll_interval_s=float(harness_raw.get("poll_interval_s", 5.0)),
        poll_max_attempts=int(harness_raw.get("poll_max_attempts", 200)),
        settle_timeout_s=float(harness_raw.get("settle_timeout_s", 1080.0)),
        gate_window=int(harness_raw.get("gate_window", DEFAULT_GATE_WINDOW)),
        enable_tier3=bool(harness_raw.get("enable_tier3", False)),
        barrier_timeout_s=float(harness_raw.get("barrier_timeout_s", 1080.0)),
        outcome_threshold=float(harness_raw.get("outcome_threshold", 0.9)),
        settle_poll_s=float(harness_raw.get("settle_poll_s", 10.0)),
        repair_model=(
            str(harness_raw["repair_model"])
            if harness_raw.get("repair_model") is not None
            else None
        ),
        repair_timeout_s=float(harness_raw.get("repair_timeout_s", 60.0)),
        repair_max_turns=int(harness_raw.get("repair_max_turns", 6)),
        repair_max_tokens=int(harness_raw.get("repair_max_tokens", 4096)),
        repair_temperature=(
            float(harness_raw["repair_temperature"])
            if harness_raw.get("repair_temperature") is not None
            else None
        ),
        repair_reasoning_effort=(
            str(harness_raw["repair_reasoning_effort"])
            if harness_raw.get("repair_reasoning_effort") is not None
            else None
        ),
        trace_repair_agent=bool(harness_raw.get("trace_repair_agent", True)),
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
