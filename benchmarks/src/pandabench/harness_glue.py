"""Harness construction, session-id plumbing, and per-benchmark replay wiring.

Isolates every point where pandabench touches the harness package so the
runners stay benchmark-focused. A fresh namespace is minted for every runner
invocation, then the *same* namespaced session id is used for the SDK trace
context, ``on_turn_end``, ``refresh``, and the record row. This keeps separate
benchmark invocations from ever sharing a remote PandaProbe session, including
when an interrupted ``--run-id`` is resumed.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pandaprobe_harness import Harness, HarnessConfig

from .config import StudyConfig

if TYPE_CHECKING:
    from pandaprobe_harness import EvalCase, ReplayContext, ReplayFn, VerifierFn

__all__ = [
    "OutcomeGrader",
    "ReplayRunner",
    "RepairSettings",
    "build_harness",
    "build_harness_config",
    "harness_root_for",
    "make_replay_fn",
    "make_session_id",
    "make_verifier_fn",
    "new_session_namespace",
    "resolve_repair_settings",
    "sanitize_component",
]

# A benchmark's replay entry point: given a task id plus a capability-only
# ReplayContext carrying read-only rule tools, re-run the task once (cheap,
# traced, no turn hooks) and return the NEW session id the run produced.
ReplayRunner = Callable[[str, "ReplayContext"], Awaitable[str]]

# A benchmark's own grader: task id + session id -> pass ratio in [0, 1], or
# None when it has no verdict (no grader, or the session has not been graded).
OutcomeGrader = Callable[[str, str], "float | None"]

_UNSAFE = re.compile(r"[^a-z0-9._-]+")
_SESSION_ID_MAX_LENGTH = 255


@dataclass(frozen=True, slots=True)
class RepairSettings:
    """Effective managed-repair settings recorded and passed to the package."""

    model: str
    timeout_s: float
    max_turns: int
    max_tokens: int
    temperature: float | None
    reasoning_effort: str | None

    def to_manifest(self) -> dict[str, Any]:
        """Use the established manifest field names."""

        return {
            "repair_model": self.model,
            "repair_timeout_s": self.timeout_s,
            "repair_max_turns": self.max_turns,
            "repair_max_tokens": self.max_tokens,
            "repair_temperature": self.temperature,
            "repair_reasoning_effort": self.reasoning_effort,
        }


def resolve_repair_settings(
    *,
    study: StudyConfig,
    repair_model: str,
    repair_overrides: Mapping[str, Any] | None = None,
) -> RepairSettings:
    """Resolve study defaults plus per-model managed-repair compatibility knobs.

    Per-model overrides apply only while the task model itself is the repair
    model. An explicit study-level ``repair_model`` selects a different model,
    so its study-level settings remain authoritative.
    """

    overrides = (
        dict(repair_overrides or {})
        if study.harness.repair_model is None
        else {}
    )
    allowed = {
        "timeout_s",
        "max_turns",
        "max_tokens",
        "temperature",
        "reasoning_effort",
    }
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"unknown repair override(s): {sorted(unknown)}")

    def chosen(name: str, default: Any) -> Any:
        return overrides[name] if name in overrides else default

    temperature = chosen("temperature", study.harness.repair_temperature)
    reasoning_effort = chosen(
        "reasoning_effort", study.harness.repair_reasoning_effort
    )
    return RepairSettings(
        model=study.harness.repair_model or repair_model,
        timeout_s=float(chosen("timeout_s", study.harness.repair_timeout_s)),
        max_turns=int(chosen("max_turns", study.harness.repair_max_turns)),
        max_tokens=int(chosen("max_tokens", study.harness.repair_max_tokens)),
        temperature=float(temperature) if temperature is not None else None,
        reasoning_effort=(
            str(reasoning_effort) if reasoning_effort is not None else None
        ),
    )


def sanitize_component(value: str) -> str:
    """Lowercase and reduce to a safe session-id component (``[a-z0-9._-]``)."""

    cleaned = _UNSAFE.sub("-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "x"


def new_session_namespace() -> str:
    """Return a globally unique namespace for one runner invocation.

    This is deliberately not derived from ``run_id``. Resuming an interrupted
    run must not append a retried task's traces to the partial remote session
    left by the previous process.
    """

    return uuid4().hex


def make_session_id(
    *,
    session_namespace: str,
    benchmark: str,
    task_id: str,
    arm: str,
    model_key: str,
    seed: int,
    trial: int,
    phase: str,
) -> str:
    """Mint a unique, readable PandaProbe session id (maximum 255 characters).

    Calls with the same namespace and semantic identity are stable, while a new
    runner invocation changes the namespace. ``phase`` is ``"live"`` for graded
    trials and ``"replay"`` for a validation replay, so the two cannot collide.
    """

    parts = [
        benchmark,
        task_id,
        arm,
        model_key,
        f"s{seed}",
        phase,
        f"t{trial}",
        f"r{session_namespace}",
    ]
    session_id = "-".join(sanitize_component(p) for p in parts)
    if len(session_id) <= _SESSION_ID_MAX_LENGTH:
        return session_id

    namespace_component = sanitize_component(f"r{session_namespace}")
    digest = sha256(session_id.encode()).hexdigest()
    suffix = f"-h{digest}-{namespace_component}"
    prefix = session_id[: _SESSION_ID_MAX_LENGTH - len(suffix)].rstrip("-")
    return f"{prefix}{suffix}"


def harness_root_for(run_dir: Path) -> Path:
    return run_dir / "harness_root"


def build_harness_config(
    *,
    harness_root: Path,
    capture: bool,
    study: StudyConfig,
    benchmark: str,
    repair_model: str,
    repair_overrides: Mapping[str, Any] | None = None,
    health_check: bool = True,
) -> HarnessConfig:
    """Resolve a HarnessConfig for one run.

    ``capture`` is explicit, not derived from a phase name: it gates
    ``capture_eval_cases``, the sole switch on whether a breaching session becomes a
    replayable failure case, and so on whether replay validation has anything to
    replay. Inferring it from a string is how a rename silently disables promotion.

    Managed repair always retains the candidate-validation lifecycle required by
    the package. The breach threshold is identical across all arms/seeds of a
    benchmark (set by Checkpoint 1). Explicit overrides beat ambient ``HARNESS_*``
    env so runs are deterministic.
    """

    threshold = study.breach_threshold(benchmark)
    repair = resolve_repair_settings(
        study=study,
        repair_model=repair_model,
        repair_overrides=repair_overrides,
    )
    benchmark_config = study.benchmarks.get(benchmark)
    benchmark_policy = (
        benchmark_config.extra.get("domain_policy") if benchmark_config is not None else None
    )
    return HarnessConfig.from_env(
        harness_root=harness_root,
        capture_eval_cases=capture,
        rule_validation=True,
        rule_trial_min_sessions=study.harness.rule_trial_min_sessions,
        rule_promote_margin=study.harness.rule_promote_margin,
        rule_regress_margin=study.harness.rule_regress_margin,
        replay_timeout_s=study.harness.replay_timeout_s,
        replay_env_wait_timeout_s=study.harness.replay_env_wait_timeout_s,
        validation_round_budget_s=study.harness.validation_round_budget_s,
        regression_sample=study.harness.regression_sample,
        # Bound how long a barrier waits for platform scores per turn
        # (poll_interval_s * poll_max_attempts); generous because trace-heavy
        # sessions score slowly.
        poll_interval_s=study.harness.poll_interval_s,
        poll_max_attempts=study.harness.poll_max_attempts,
        rule_retrieval=True,
        health_check=health_check,
        gate_window=study.harness.gate_window,
        gate_target=threshold,
        enable_tier3=study.harness.enable_tier3,
        barrier_timeout_s=study.harness.barrier_timeout_s,
        outcome_threshold=study.harness.outcome_threshold,
        repair_model=repair.model,
        repair_timeout_s=repair.timeout_s,
        repair_max_turns=repair.max_turns,
        repair_max_tokens=repair.max_tokens,
        repair_temperature=repair.temperature,
        repair_reasoning_effort=repair.reasoning_effort,
        trace_repair_agent=study.harness.trace_repair_agent,
        domain_policy=str(benchmark_policy) if benchmark_policy is not None else None,
    )


def build_harness(
    *,
    cfg: HarnessConfig,
    replay: ReplayFn | None = None,
    verifier: VerifierFn | None = None,
) -> Harness:
    """Assemble a harness against the real ``pandaprobe`` CLI (no ``cli=`` seam)."""

    return Harness.create(cfg, replay=replay, verifier=verifier)


def make_verifier_fn(*, outcome_for: OutcomeGrader) -> VerifierFn:
    """Adapt a benchmark's own grader into the harness outcome verifier.

    The task id travels in the ``end_state`` the runner hands to ``on_turn_end``,
    which is also what a captured eval case stores as its ``replay_input`` — so the
    same adapter grades live turns and replays. ``None`` back from the grader means
    "no verdict for this session yet", which the harness treats as a normal answer.
    """

    def verifier(session_id: str, end_state: Mapping[str, Any]) -> float | None:
        task_id = end_state.get("task_id")
        return outcome_for(str(task_id), session_id) if task_id else None

    return verifier


def make_replay_fn(*, replay_runner: ReplayRunner) -> ReplayFn:
    """Build the harness ReplayFn from a benchmark's replay entry point.

    The harness calls ``replay(case, context)`` during candidate validation and
    regression: we pull the task id from ``case.replay_input`` (the end_state we
    stashed via ``on_turn_end``) and re-run that task under ``context``. The
    context carries a stable capability preamble and read-only rule tools; it
    never carries rule bodies. The replay returns a new session id to score.
    """

    async def replay(case: EvalCase, context: ReplayContext) -> str:
        payload = case.replay_input or {}
        task_id = payload.get("task_id") if isinstance(payload, dict) else None
        if not task_id:
            raise RuntimeError(f"eval case {case.id} has no replayable task_id")
        return await replay_runner(str(task_id), context)

    return replay
