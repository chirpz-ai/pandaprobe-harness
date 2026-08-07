"""Generic run orchestration shared by every benchmark.

A :class:`BenchmarkRunner` drives one ``(benchmark x model x arm x seed)`` run:
the learning phase (arm B captures + validates rules; arm A runs the same split
for symmetry) then the frozen eval phase, ``k`` trials each, with resumability
and the arm-B ``refresh`` + ``drain_validation`` pacing baked in. Benchmark-
specific work is confined to a :class:`SingleTaskRunner` (``run_once`` or the
optional bulk ``run_phase`` hook); the harness session lifecycle lives here so it
is identical across benchmarks and so ``run_once`` can be reused verbatim by the
ReplayFn (with ``wiring=None``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pandaprobe_harness import RuleScopeHint

from ..agents.frozen_wiring import FrozenEvalWiring
from ..agents.harness_wiring import AgentWiring, HarnessWiring, ReplayRuleWiring
from ..config import StudyConfig
from ..frozen_rules import FROZEN_RULES_FILENAME, FrozenRulesSnapshot
from ..harness_glue import (
    build_harness,
    build_harness_config,
    harness_root_for,
    make_replay_fn,
    make_session_id,
    make_verifier_fn,
    new_session_namespace,
)
from ..providers.litellm_client import ChatClient, LiteLLMClient, MockClient, Usage
from ..providers.models import ModelRegistry, ResolvedModel
from ..providers.tracing import PandaTracer
from ..results import (
    RecordWriter,
    RunManifest,
    TrialRecord,
    archive_workspace,
    collect_harness_telemetry,
    env_fingerprint,
    frozen_harness_telemetry,
    git_sha,
    package_version,
    resume_key,
    uv_lock_hash,
)

logger = logging.getLogger("pandabench.runner")

# Credentials/config whose PRESENCE (not value) we fingerprint into the manifest.
_ENV_KEYS = (
    "VERTEXAI_PROJECT", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "CLAUDE_BACKEND",
    "PANDAPROBE_API_KEY", "PANDAPROBE_PROJECT_NAME",
)


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """The benchmark-native result of driving one task once."""

    passed: bool
    native_metrics: dict[str, Any]
    turns: int
    wall_time_s: float
    usage: Usage
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TaskSplits:
    learning: list[str]
    eval: list[str]


class SingleTaskRunner(Protocol):
    """A benchmark's task surface: enumerate tasks and drive one to completion."""

    name: str

    def configure_dataset(self, dataset: str) -> None:
        """Select the dataset/domain used by subsequent task runs.

        Most runners receive all dataset context through ``list_tasks`` and need
        no setup. Stateful integrations such as tau2 override this hook because
        the selected domain also determines the environment and evaluator.
        """

        return None

    def list_tasks(self, dataset: str) -> list[str]: ...

    def rule_scope_hints(self, task_id: str) -> tuple[RuleScopeHint, ...]:
        """Return safe deterministic scope metadata already known by the host."""

        del task_id
        return ()

    def task_summary(self, task_id: str) -> str:
        """What this task asks for, in the benchmark's own words.

        Managed repair reads it as untrusted evidence when diagnosing a failure
        and choosing a rule scope, so a benchmark with only opaque task ids does
        not force every rule into one undifferentiated file. Empty means "no
        statement available"; the harness never requires one.
        """

        del task_id
        return ""

    async def run_once(
        self,
        *,
        task_id: str,
        session_id: str,
        model: ResolvedModel,
        client: ChatClient,
        max_turns: int,
        wiring: AgentWiring | None,
    ) -> TaskOutcome: ...

    def outcome_for(self, task_id: str, session_id: str) -> float | None:
        """This benchmark's verdict for one task session, as a score in ``[0, 1]``.

        The gold signal for the harness's outcome verifier. Default ``None`` means
        "this benchmark has no grader", so a runner opts in by overriding — and the
        capability is named in the type rather than discovered by ``getattr``,
        where a misspelling would silently produce a verifier-less run.
        """

        return None

    async def run_phase(
        self,
        *,
        tasks: Sequence[str],
        k: int,
        arm: str,
        model: ResolvedModel,
        phase: str,
        dataset: str,
        harness_root: Path,
        writer: RecordWriter,
        run_id: str,
        session_namespace: str,
        seed: int,
        backend: str | None,
        max_turns: int,
        benchmark: str,
        frozen_rules_path: Path | None,
    ) -> bool:
        """Drive a whole phase when the benchmark owns task/attempt iteration.

        Structural implementations need not define this method. The orchestrator
        treats an absent hook exactly like this default ``False`` result and falls
        through to the normal per-task loop.
        """

        return False

    async def aclose(self) -> None: ...


def _run_id(benchmark: str, model: str, arm: str, seed: int) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{benchmark}_{model}_{arm}_{seed}_{ts}"


class BenchmarkRunner:
    """Orchestrates the learning + eval phases for one run tuple."""

    def __init__(
        self,
        *,
        single: SingleTaskRunner,
        study: StudyConfig,
        registry: ModelRegistry,
        run_root: Path,
        repo_root: Path,
        lock_path: Path,
        num_retries: int = 2,
        timeout_s: float = 120.0,
    ) -> None:
        self._single = single
        self._study = study
        self._registry = registry
        self._run_root = run_root
        self._repo_root = repo_root
        self._lock_path = lock_path
        self._num_retries = num_retries
        self._timeout_s = timeout_s
        self._replay_counter = 0

    # -- public entry ---------------------------------------------------------

    async def run(
        self,
        *,
        arm: str,
        model_key: str,
        backend: str | None,
        seed: int,
        k: int,
        limit: int | None = None,
        dry_run: bool = False,
        phases: Sequence[str] = ("learning", "eval"),
        run_id: str | None = None,
        max_turns_override: int | None = None,
        dataset_override: str | None = None,
    ) -> Path:
        benchmark = self._single.name
        bench_cfg = self._study.benchmark(benchmark)
        dataset = dataset_override or bench_cfg.dataset
        self._single.configure_dataset(dataset)
        model = self._resolve_model(model_key, backend, dry_run)
        run_id = run_id or _run_id(benchmark, model.key, arm, seed)
        session_namespace = new_session_namespace()
        run_dir = self._run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = RecordWriter(run_dir / "records.jsonl")
        max_turns = max_turns_override or bench_cfg.max_turns

        client = self._make_client(arm, dry_run)
        splits = self._splits(dataset, seed, limit, benchmark)
        harness_root = harness_root_for(run_dir)
        use_harness = arm == "harness" and not dry_run
        snapshot_path = run_dir / FROZEN_RULES_FILENAME
        snapshot: FrozenRulesSnapshot | None = None
        learning_outcome = _existing_learning_outcome(run_dir / "manifest.json")
        harness: Any = None

        logger.info(
            "run %s: arm=%s model=%s seed=%s learning=%d eval=%d",
            run_id, arm, model.key, seed, len(splits.learning), len(splits.eval),
        )

        if "learning" in phases:
            harness = await self._run_phase(
                phase="learning", tasks=splits.learning, k=k, arm=arm, model=model,
                client=client, writer=writer, run_id=run_id,
                session_namespace=session_namespace, seed=seed, backend=backend,
                max_turns=max_turns, benchmark=benchmark,
                dataset=dataset, harness_root=harness_root, use_harness=use_harness,
                frozen_snapshot=None, frozen_rules_path=None,
            )
            if harness is not None:
                # Barrier: drain learning-phase evals + candidate validation within
                # the existing bound, then freeze the exact resulting lifecycle state.
                await self._settle(harness, "learning")
                snapshot = self._freeze_learning_rules(snapshot_path, harness)
                learning_outcome = _checkpoint_two(snapshot)
                # Eval never owns a live harness, so archive the settled learning
                # workspace now rather than relying on a post-eval barrier.
                archive_workspace(harness_root, run_dir / "harness")

        if "eval" in phases:
            if use_harness and snapshot is None:
                snapshot = self._snapshot_for_eval(snapshot_path)
            # Deliberately drop the learning Harness. Frozen eval is backed only
            # by the persisted benchmark snapshot and cannot touch the workspace.
            harness = None
            await self._run_phase(
                phase="eval", tasks=splits.eval, k=k, arm=arm, model=model,
                client=client, writer=writer, run_id=run_id,
                session_namespace=session_namespace, seed=seed, backend=backend,
                max_turns=max_turns, benchmark=benchmark,
                dataset=dataset, harness_root=harness_root, use_harness=use_harness,
                frozen_snapshot=snapshot,
                frozen_rules_path=snapshot_path if snapshot is not None else None,
            )

        self._write_manifest(
            run_dir=run_dir, run_id=run_id, benchmark=benchmark, model=model,
            arm=arm, seed=seed, backend=backend, learning_outcome=learning_outcome,
            phases=phases, k=k, dry_run=dry_run, dataset=dataset, snapshot=snapshot,
        )
        await self._single.aclose()
        logger.info("run %s complete: %d records", run_id, writer.count)
        return run_dir

    # -- phases ---------------------------------------------------------------

    async def _run_phase(
        self, *, phase: str, tasks: Sequence[str], k: int, arm: str, model: ResolvedModel,
        client: ChatClient, writer: RecordWriter, run_id: str, seed: int,
        backend: str | None, max_turns: int, benchmark: str, dataset: str,
        harness_root: Path, use_harness: bool, session_namespace: str,
        frozen_snapshot: FrozenRulesSnapshot | None,
        frozen_rules_path: Path | None,
    ) -> Any:
        bulk_hook = getattr(self._single, "run_phase", None)
        if bulk_hook is not None:
            handled = await bulk_hook(
                tasks=tasks, k=k, arm=arm, model=model, phase=phase, dataset=dataset,
                harness_root=harness_root, writer=writer, run_id=run_id, seed=seed,
                backend=backend, max_turns=max_turns, benchmark=benchmark,
                session_namespace=session_namespace,
                frozen_rules_path=frozen_rules_path,
            )
            if handled:
                # Harbor's custom agent owns the live per-turn Harness. Construct
                # this process's phase-level view only after Harbor exits, with no
                # replay or verifier: Terminal-Bench supports neither capability.
                return (
                    self._build_harness(
                        harness_root, phase, benchmark, model, seed,
                        session_namespace, bulk=True,
                    )
                    if use_harness and phase == "learning"
                    else None
                )

        harness = (
            self._build_harness(
                harness_root, phase, benchmark, model, seed, session_namespace
            )
            if use_harness and phase == "learning"
            else None
        )
        for task_id in tasks:
            for trial in range(k):
                key = resume_key(benchmark, task_id, arm, model.key, backend, seed, trial, phase)
                if writer.done(key):
                    logger.info("skip (resumed): %s t%d %s", task_id, trial, phase)
                    continue
                record = await self._run_trial(
                    phase=phase, task_id=task_id, trial=trial, arm=arm, model=model,
                    client=client, harness=harness, run_id=run_id, seed=seed,
                    backend=backend, max_turns=max_turns, benchmark=benchmark,
                    session_namespace=session_namespace,
                    frozen_snapshot=frozen_snapshot,
                )
                writer.append(record)
                status = "PASS" if record.passed else ("ERR" if record.error else "fail")
                logger.info(
                    "%s %s t%d %s -> %s (%.1fs, $%.4f)",
                    benchmark, task_id, trial, phase, status,
                    record.wall_time_s, record.usage.get("cost_usd", 0.0),
                )
        return harness

    async def _run_trial(
        self, *, phase: str, task_id: str, trial: int, arm: str, model: ResolvedModel,
        client: ChatClient, harness: Any, run_id: str, seed: int, backend: str | None,
        max_turns: int, benchmark: str, session_namespace: str,
        frozen_snapshot: FrozenRulesSnapshot | None,
    ) -> TrialRecord:
        session_id = make_session_id(
            session_namespace=session_namespace, benchmark=benchmark, task_id=task_id,
            arm=arm, model_key=model.key, seed=seed, trial=trial, phase=phase,
        )
        wiring: AgentWiring | None = None
        if harness is not None:
            descriptor = {
                "benchmark": benchmark, "task_id": task_id, "arm": arm,
                "model_key": model.key, "backend": backend, "seed": seed, "trial": trial,
                "run_id": run_id, "phase": phase,
            }
            wiring = HarnessWiring(
                harness=harness, benchmark=benchmark, task_id=task_id,
                capture=(phase == "learning"), replay_descriptor=descriptor,
                # The loop settles every turn through this wiring, so the harness
                # sees the trajectory one trace at a time instead of once at the
                # end — the precondition for the trajectory gate having a series.
                session_id=session_id, flush=client.flush,
                rule_scope_hints=self._single.rule_scope_hints(task_id),
                task_summary=self._single.task_summary(task_id),
            )
        elif frozen_snapshot is not None and arm == "harness" and phase == "eval":
            wiring = FrozenEvalWiring(frozen_snapshot)

        outcome = await self._single.run_once(
            task_id=task_id, session_id=session_id, model=model, client=client,
            max_turns=max_turns, wiring=wiring,
        )

        report = None
        telemetry = None
        if harness is not None and wiring is not None:
            # The loop settles every turn that continues; this covers the final
            # turn, whose trace only exists once the loop has returned. Take the
            # report from this call — the barrier consumes the pending evaluation,
            # so settling again would yield an empty report (v1 recorded null
            # scores on every row for exactly that reason).
            settled = await wiring.settle_turn(max(outcome.turns, 1))
            report = settled.report if settled is not None else None
            repair = settled.repair if settled is not None else None
            telemetry = collect_harness_telemetry(
                harness, session_id, report, repair=repair
            ).to_dict()
        elif frozen_snapshot is not None and arm == "harness" and phase == "eval":
            telemetry = frozen_harness_telemetry(frozen_snapshot, session_id).to_dict()

        return TrialRecord(
            run_id=run_id, benchmark=benchmark, task_id=task_id, arm=arm,
            model=model.key, provider=model.provider, backend=model.backend,
            resolved_model=model.litellm_model, seed=seed, trial=trial, phase=phase,
            passed=outcome.passed, native_metrics=outcome.native_metrics,
            turns=outcome.turns, wall_time_s=outcome.wall_time_s,
            usage=outcome.usage.to_dict(), harness=telemetry, error=outcome.error,
        )

    # -- helpers --------------------------------------------------------------

    def _resolve_model(self, model_key: str, backend: str | None, dry_run: bool) -> ResolvedModel:
        if dry_run:
            return self._registry.resolve(self._registry.role("dry_run"))
        return self._registry.resolve(model_key, backend=backend)

    def _make_client(self, arm: str, dry_run: bool) -> ChatClient:
        if dry_run:
            return MockClient()
        tracer = PandaTracer.from_env() if arm == "harness" else PandaTracer.disabled()
        return LiteLLMClient(
            tracer=tracer, num_retries=self._num_retries, timeout_s=self._timeout_s
        )

    def _build_harness(
        self, harness_root: Path, phase: str, benchmark: str, model: ResolvedModel,
        seed: int, session_namespace: str, *, bulk: bool = False,
    ) -> Any:
        cfg = build_harness_config(
            harness_root=harness_root, phase=phase, study=self._study,
            benchmark=benchmark, repair_model=model.litellm_model,
            health_check=not bulk,
        )
        return build_harness(
            cfg=cfg,
            replay=(
                None
                if bulk
                else self._make_replay(benchmark, model, seed, session_namespace)
            ),
            verifier=(
                None
                if bulk
                else make_verifier_fn(outcome_for=self._single.outcome_for)
            ),
        )

    def _make_replay(
        self, benchmark: str, model: ResolvedModel, seed: int, session_namespace: str
    ) -> Any:
        """Build the Harness ReplayFn with candidate rules available on demand.

        Uses a TRACED client (so the replayed session is scoreable) and
        ReplayRuleWiring exposes read-only tools without settlement, so replay
        cannot recurse or re-capture. The
        replay session uses ``arm="replay"`` so it never collides with graded
        records and is excluded from metrics.
        """

        replay_client = LiteLLMClient(
            tracer=PandaTracer.from_env(), num_retries=self._num_retries, timeout_s=self._timeout_s
        )
        replay_max_turns = self._study.harness.replay_max_turns

        async def replay_runner(task_id: str, context: Any) -> str:
            self._replay_counter += 1
            session_id = make_session_id(
                session_namespace=session_namespace, benchmark=benchmark, task_id=task_id,
                arm="replay", model_key=model.key, seed=seed,
                trial=self._replay_counter, phase="replay",
            )
            try:
                await self._single.run_once(
                    task_id=task_id, session_id=session_id, model=model, client=replay_client,
                    max_turns=replay_max_turns, wiring=ReplayRuleWiring(context),
                )
            finally:
                replay_client.flush()
            return session_id

        return make_replay_fn(replay_runner=replay_runner)

    @staticmethod
    def _freeze_learning_rules(path: Path, harness: Any) -> FrozenRulesSnapshot:
        """Create once after learning settlement; never refresh an existing boundary."""

        if path.exists():
            logger.info("reusing existing frozen learning snapshot at %s", path)
            return FrozenRulesSnapshot.load(path)
        snapshot = FrozenRulesSnapshot.create(harness.rules.all())
        snapshot.save(path)
        logger.info(
            "froze learning rules: sha256=%s active=%d candidate=%d retired=%d",
            snapshot.sha256,
            snapshot.active_count,
            snapshot.candidate_count,
            snapshot.retired_count,
        )
        return snapshot

    @staticmethod
    def _snapshot_for_eval(path: Path) -> FrozenRulesSnapshot:
        """Load a resume snapshot, or persist an explicit empty eval-only one."""

        if path.exists():
            snapshot = FrozenRulesSnapshot.load(path)
            logger.info("loaded frozen eval ruleset sha256=%s", snapshot.sha256)
            return snapshot
        snapshot = FrozenRulesSnapshot.create(())
        snapshot.save(path)
        logger.info(
            "eval-only harness run has no prior snapshot; created explicit empty ruleset %s",
            snapshot.sha256,
        )
        return snapshot

    async def _settle(self, harness: Any, label: str) -> None:
        """Await outstanding background evals + validate candidate rules (bounded).

        Each session is scored in a detached background task that can take minutes
        (LLM-judged over many traces); ``refresh``/``on_turn_end`` don't block on it.
        This barrier loops until BOTH the evals and candidate validation are
        quiet, or ``settle_timeout_s`` elapses.

        Waiting on evals alone is not enough, and used to be the whole loop: it
        exits as soon as scores have landed, leaving validation to a trailing
        best-effort join bounded by ``drain_timeout_s`` (15s) — far less than one
        replay-bound round needs. The snapshot below then froze candidates that
        had already earned a verdict as permanently provisional.
        """

        deadline = time.monotonic() + self._study.harness.settle_timeout_s
        while time.monotonic() < deadline:
            try:
                await harness.refresh_all()
                await harness.drain_validation(
                    timeout=max(0.0, deadline - time.monotonic())
                )
            except Exception as exc:  # noqa: BLE001 - never crash the run on settle
                logger.warning("settle(%s): drain error: %s", label, exc)
                return
            pending = harness.hook.pending_sessions
            validating = harness.validation_pending
            if not pending and not validating:
                break
            logger.info(
                "settle(%s): %d turn eval(s), %d validation round(s) still pending...",
                label, len(pending), validating,
            )
            await asyncio.sleep(self._study.harness.settle_poll_s)
        # Evals are quiet, so every trial's evidence is in. Run validation out to a
        # standstill on the remaining budget: this is the only point in the run
        # where nothing holds the environment a replay needs.
        try:
            settled = await harness.settle_validation(
                timeout=max(0.0, deadline - time.monotonic())
            )
            if not settled:
                logger.warning(
                    "settle(%s): validation did not settle within settle_timeout_s; "
                    "undecided candidates will be frozen as provisional",
                    label,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("settle(%s): settle_validation error: %s", label, exc)

    def _splits(self, dataset: str, seed: int, limit: int | None, benchmark: str) -> TaskSplits:
        cfg = self._study.benchmark(benchmark)
        if cfg.learning_split or cfg.eval_split:
            learning = list(cfg.learning_split)
            eval_ = list(cfg.eval_split)
        else:
            all_ids = self._single.list_tasks(dataset)
            shuffled = list(all_ids)
            random.Random(seed).shuffle(shuffled)
            n_learn = round(len(shuffled) * cfg.learning_fraction)
            learning, eval_ = shuffled[:n_learn], shuffled[n_learn:]
        if limit is not None:
            learning, eval_ = learning[:limit], eval_[:limit]
        return TaskSplits(learning, eval_)

    def _write_manifest(
        self, *, run_dir: Path, run_id: str, benchmark: str, model: ResolvedModel,
        arm: str, seed: int, backend: str | None, learning_outcome: str | None,
        phases: Sequence[str], k: int, dry_run: bool,
        dataset: str, snapshot: FrozenRulesSnapshot | None,
    ) -> None:
        manifest = RunManifest(
            run_id=run_id, benchmark=benchmark, model=model.key, arm=arm, seed=seed,
            backend=model.backend, started_at=datetime.now(UTC).isoformat(),
            git_sha=git_sha(self._repo_root), uv_lock_hash=uv_lock_hash(self._lock_path),
            pandaprobe_harness_version=package_version("pandaprobe-harness"),
            litellm_version=package_version("litellm"),
            resolved_config={
                "resolved_model": model.litellm_model, "provider": model.provider,
                "dataset": dataset, "k": k, "phases": list(phases), "dry_run": dry_run,
                "breach_threshold": self._study.breach_threshold(benchmark),
                "rule_trial_min_sessions": self._study.harness.rule_trial_min_sessions,
                "gate_window": self._study.harness.gate_window,
                "repair_model": self._study.harness.repair_model or model.litellm_model,
                "repair_timeout_s": self._study.harness.repair_timeout_s,
                "repair_max_turns": self._study.harness.repair_max_turns,
                "repair_max_tokens": self._study.harness.repair_max_tokens,
                "repair_temperature": self._study.harness.repair_temperature,
                "repair_reasoning_effort": (
                    self._study.harness.repair_reasoning_effort
                ),
                "trace_repair_agent": self._study.harness.trace_repair_agent,
                "managed_repair": True,
                "eval_policy": "frozen_rules" if arm == "harness" else "baseline",
                "trace_eval_during_eval": False,
                **(
                    {
                        "ruleset_hash": snapshot.sha256,
                        "rules_active": snapshot.active_count,
                        "rules_candidate": snapshot.candidate_count,
                        "rules_retired": snapshot.retired_count,
                    }
                    if snapshot is not None
                    else {}
                ),
            },
            env_fingerprint=env_fingerprint(_ENV_KEYS),
            learning_outcome=learning_outcome,
        )
        manifest.write(run_dir / "manifest.json")


def _existing_learning_outcome(path: Path) -> str | None:
    """Preserve the learning checkpoint when an eval-only resume rewrites a manifest."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = raw.get("learning_outcome") if isinstance(raw, dict) else None
    return str(value) if value is not None else None


def _checkpoint_two(snapshot: FrozenRulesSnapshot) -> str:
    """Checkpoint 2: did the learning phase promote any rules? Stamp the outcome.

    "Learning produced nothing" and "learning produced N candidates that
    validation never decided" are very different results, and reporting both as
    ``no_rules`` hid the second one completely. Undecided candidates are named.
    """

    active = snapshot.active_count
    candidate = snapshot.candidate_count
    logger.info("checkpoint-2: rules_active=%d rules_candidate=%d", active, candidate)
    if active:
        return f"active={active}" if not candidate else f"active={active},pending={candidate}"
    return f"pending={candidate}" if candidate else "no_rules"


# Timer helper reused by run_once implementations.
class Stopwatch:
    def __enter__(self) -> Stopwatch:
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.monotonic() - self._start
