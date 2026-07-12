"""Harness construction, session-id plumbing, and per-benchmark replay wiring.

Isolates every point where pandabench touches the harness package so the
runners stay benchmark-focused. The session id is the linchpin: the *same*
sanitized id is used for the SDK trace context, ``on_turn_end``, ``refresh``,
the record row, and the calibrate label join — one function mints it everywhere
so ids never drift (a drift would silently break resume + label joins).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pandaprobe_harness import Harness, HarnessConfig

from .config import StudyConfig

if TYPE_CHECKING:
    from pandaprobe_harness import EvalCase, ReplayFn

__all__ = [
    "ReplayRunner",
    "build_harness",
    "build_harness_config",
    "harness_root_for",
    "make_replay_fn",
    "make_session_id",
    "sanitize_component",
]

# A benchmark's replay entry point: given a task id + the harness-rendered rules
# context, re-run the task once (cheap, traced, no turn hooks) and return the
# NEW session id the run produced.
ReplayRunner = Callable[[str, str], Awaitable[str]]

_UNSAFE = re.compile(r"[^a-z0-9._-]+")


def sanitize_component(value: str) -> str:
    """Lowercase and reduce to a safe session-id component (``[a-z0-9._-]``)."""

    cleaned = _UNSAFE.sub("-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "x"


def make_session_id(
    *, benchmark: str, task_id: str, arm: str, model_key: str, seed: int, trial: int
) -> str:
    """The one session-id format used everywhere (loop, records, labels, replay)."""

    parts = [benchmark, task_id, arm, model_key, str(seed), f"t{trial}"]
    return "-".join(sanitize_component(p) for p in parts)


def harness_root_for(run_dir: Path) -> Path:
    return run_dir / "harness_root"


def build_harness_config(
    *,
    harness_root: Path,
    phase: str,
    study: StudyConfig,
    benchmark: str,
    noval: bool = False,
    health_check: bool = True,
) -> HarnessConfig:
    """Resolve a HarnessConfig for one run.

    Capture is on only in the learning phase; validation is off only for the
    B' (harness-noval) ablation; the breach threshold is identical across all
    arms/seeds of a benchmark (set by Checkpoint 1). Explicit overrides beat any
    ambient ``HARNESS_*`` env so runs are deterministic.
    """

    threshold = study.breach_threshold(benchmark)
    return HarnessConfig.from_env(
        harness_root=harness_root,
        capture_eval_cases=(phase == "learning"),
        rule_validation=(not noval),
        rule_trial_min_sessions=study.harness.rule_trial_min_sessions,
        rule_promote_margin=study.harness.rule_promote_margin,
        rule_regress_margin=study.harness.rule_regress_margin,
        reliability_threshold=threshold,
        consistency_threshold=threshold,
        replay_timeout_s=study.harness.replay_timeout_s,
        regression_sample=study.harness.regression_sample,
        rule_retrieval=True,
        health_check=health_check,
    )


def build_harness(*, cfg: HarnessConfig, replay: ReplayFn | None = None) -> Harness:
    """Assemble a harness against the real ``pandaprobe`` CLI (no ``cli=`` seam)."""

    return Harness.create(cfg, replay=replay)


def make_replay_fn(*, replay_runner: ReplayRunner) -> ReplayFn:
    """Build the harness ReplayFn from a benchmark's replay entry point.

    The harness calls ``replay(case, context)`` during candidate validation and
    regression: we pull the task id from ``case.replay_input`` (the end_state we
    stashed via ``on_turn_end``) and re-run that task under ``context`` (the
    rendered rules string with the candidate in force), returning the new
    session id for the harness to score.
    """

    async def replay(case: EvalCase, context: str) -> str:
        payload = case.replay_input or {}
        task_id = payload.get("task_id") if isinstance(payload, dict) else None
        if not task_id:
            raise RuntimeError(f"eval case {case.id} has no replayable task_id")
        return await replay_runner(str(task_id), context)

    return replay
