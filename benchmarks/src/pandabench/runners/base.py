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
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ..agents.harness_wiring import HarnessWiring
from ..config import StudyConfig
from ..harness_glue import (
    build_harness,
    build_harness_config,
    harness_root_for,
    make_replay_fn,
    make_session_id,
    make_verifier_fn,
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

    async def run_once(
        self,
        *,
        task_id: str,
        session_id: str,
        model: ResolvedModel,
        client: ChatClient,
        max_turns: int,
        wiring: HarnessWiring | None,
        preamble: str | None = None,
    ) -> TaskOutcome: ...

    def outcome_for(self, task_id: str) -> float | None:
        """This benchmark's own verdict for ``task_id``, as a score in ``[0, 1]``.

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
        seed: int,
        backend: str | None,
        max_turns: int,
        benchmark: str,
        noval: bool,
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
        noval: bool = False,
        max_turns_override: int | None = None,
        dataset_override: str | None = None,
    ) -> Path:
        benchmark = self._single.name
        bench_cfg = self._study.benchmark(benchmark)
        dataset = dataset_override or bench_cfg.dataset
        self._single.configure_dataset(dataset)
        model = self._resolve_model(model_key, backend, dry_run)
        run_id = run_id or _run_id(benchmark, model.key, arm, seed)
        run_dir = self._run_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = RecordWriter(run_dir / "records.jsonl")
        max_turns = max_turns_override or bench_cfg.max_turns

        client = self._make_client(arm, dry_run)
        splits = self._splits(dataset, seed, limit, benchmark)
        harness_root = harness_root_for(run_dir)
        use_harness = arm == "harness" and not dry_run
        learning_outcome: str | None = None
        harness: Any = None

        logger.info(
            "run %s: arm=%s model=%s seed=%s learning=%d eval=%d",
            run_id, arm, model.key, seed, len(splits.learning), len(splits.eval),
        )

        if "learning" in phases:
            harness = await self._run_phase(
                phase="learning", tasks=splits.learning, k=k, arm=arm, model=model,
                client=client, writer=writer, run_id=run_id,
                seed=seed, backend=backend, max_turns=max_turns, benchmark=benchmark,
                dataset=dataset, harness_root=harness_root, use_harness=use_harness,
                noval=noval,
            )
            if harness is not None:
                # Barrier: drain learning-phase evals + validate candidate rules so
                # the eval phase runs with a settled, promoted ruleset.
                await self._settle(harness, "learning")
                learning_outcome = _checkpoint_two(harness)

        if "eval" in phases:
            # Rebuild against the SAME root with capture off; learned rules persist.
            # Drop the learning-phase object before a bulk runner starts Harbor,
            # which constructs its own Harness instances against this root.
            harness = None
            harness = await self._run_phase(
                phase="eval", tasks=splits.eval, k=k, arm=arm, model=model,
                client=client, writer=writer, run_id=run_id,
                seed=seed, backend=backend, max_turns=max_turns, benchmark=benchmark,
                dataset=dataset, harness_root=harness_root, use_harness=use_harness,
                noval=noval,
            )

        if use_harness and harness is not None:
            # Final barrier: let outstanding evals land + validate before archiving
            # so the workspace (journal/rules/evalset) is complete.
            await self._settle(harness, "final")
            archive_workspace(harness_root, run_dir / "harness")

        self._write_manifest(
            run_dir=run_dir, run_id=run_id, benchmark=benchmark, model=model,
            arm=arm, seed=seed, backend=backend, learning_outcome=learning_outcome,
            phases=phases, k=k, dry_run=dry_run, dataset=dataset,
        )
        await self._single.aclose()
        logger.info("run %s complete: %d records", run_id, writer.count)
        return run_dir

    # -- phases ---------------------------------------------------------------

    async def _run_phase(
        self, *, phase: str, tasks: Sequence[str], k: int, arm: str, model: ResolvedModel,
        client: ChatClient, writer: RecordWriter, run_id: str, seed: int,
        backend: str | None, max_turns: int, benchmark: str, dataset: str,
        harness_root: Path, use_harness: bool, noval: bool,
    ) -> Any:
        bulk_hook = getattr(self._single, "run_phase", None)
        if bulk_hook is not None:
            handled = await bulk_hook(
                tasks=tasks, k=k, arm=arm, model=model, phase=phase, dataset=dataset,
                harness_root=harness_root, writer=writer, run_id=run_id, seed=seed,
                backend=backend, max_turns=max_turns, benchmark=benchmark, noval=noval,
            )
            if handled:
                # Harbor's custom agent owns the live per-turn Harness. Construct
                # this process's phase-level view only after Harbor exits, with no
                # replay or verifier: Terminal-Bench supports neither capability.
                return (
                    self._build_harness(
                        harness_root, phase, benchmark, noval, model, seed, bulk=True
                    )
                    if use_harness
                    else None
                )

        harness = (
            self._build_harness(harness_root, phase, benchmark, noval, model, seed)
            if use_harness
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
        max_turns: int, benchmark: str,
    ) -> TrialRecord:
        session_id = make_session_id(
            benchmark=benchmark, task_id=task_id, arm=arm,
            model_key=model.key, seed=seed, trial=trial,
        )
        wiring: HarnessWiring | None = None
        if harness is not None:
            descriptor = {
                "benchmark": benchmark, "task_id": task_id, "arm": arm,
                "model_key": model.key, "backend": backend, "seed": seed, "trial": trial,
            }
            wiring = HarnessWiring(
                harness=harness, benchmark=benchmark, task_id=task_id,
                capture=(phase == "learning"), replay_descriptor=descriptor,
                # The loop settles every turn through this wiring, so the harness
                # sees the trajectory one trace at a time instead of once at the
                # end — the precondition for the trajectory gate having a series.
                session_id=session_id, flush=client.flush,
            )

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
            telemetry = collect_harness_telemetry(harness, session_id, report).to_dict()

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
        self, harness_root: Path, phase: str, benchmark: str, noval: bool,
        model: ResolvedModel, seed: int, *, bulk: bool = False,
    ) -> Any:
        cfg = build_harness_config(
            harness_root=harness_root, phase=phase, study=self._study,
            benchmark=benchmark, noval=noval, health_check=not bulk,
        )
        return build_harness(
            cfg=cfg,
            replay=None if bulk else self._make_replay(benchmark, model, seed),
            verifier=(
                None
                if bulk
                else make_verifier_fn(outcome_for=self._single.outcome_for)
            ),
        )

    def _make_replay(self, benchmark: str, model: ResolvedModel, seed: int) -> Any:
        """Build the harness ReplayFn: re-run a captured task under candidate rules.

        Uses a TRACED client (so the replayed session is scoreable) and
        ``wiring=None`` (no ``on_turn_end`` -> no recursion / re-capture); the
        replay session uses ``arm="replay"`` so it never collides with graded
        records and is excluded from metrics.
        """

        replay_client = LiteLLMClient(
            tracer=PandaTracer.from_env(), num_retries=self._num_retries, timeout_s=self._timeout_s
        )
        replay_max_turns = self._study.harness.replay_max_turns

        async def replay_runner(task_id: str, preamble: str) -> str:
            self._replay_counter += 1
            session_id = make_session_id(
                benchmark=benchmark, task_id=task_id, arm="replay",
                model_key=model.key, seed=seed, trial=self._replay_counter,
            )
            await self._single.run_once(
                task_id=task_id, session_id=session_id, model=model, client=replay_client,
                max_turns=replay_max_turns, wiring=None, preamble=preamble,
            )
            return session_id

        return make_replay_fn(replay_runner=replay_runner)

    async def _settle(self, harness: Any, label: str) -> None:
        """Await outstanding background evals + validate candidate rules (bounded).

        Each session is scored in a detached background task that can take minutes
        (LLM-judged over many traces); ``refresh``/``on_turn_end`` don't block on it.
        This barrier loops ``refresh_all`` + ``drain_validation`` until those tasks
        drain or ``settle_timeout_s`` elapses, so notices land and candidate rules
        get promoted/retired before the next phase / before archiving.
        """

        deadline = time.monotonic() + self._study.harness.settle_timeout_s
        while time.monotonic() < deadline:
            try:
                await harness.refresh_all()
                await harness.drain_validation()
            except Exception as exc:  # noqa: BLE001 - never crash the run on settle
                logger.warning("settle(%s): drain error: %s", label, exc)
                return
            pending = harness.hook.pending_sessions
            if not pending:
                break
            logger.info("settle(%s): %d session eval(s) still pending...", label, len(pending))
            await asyncio.sleep(self._study.harness.settle_poll_s)
        try:
            await harness.drain_validation()
        except Exception as exc:  # noqa: BLE001
            logger.warning("settle(%s): final drain_validation error: %s", label, exc)

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
        dataset: str,
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
            },
            env_fingerprint=env_fingerprint(_ENV_KEYS),
            learning_outcome=learning_outcome,
        )
        manifest.write(run_dir / "manifest.json")


def _checkpoint_two(harness: Any) -> str:
    """Checkpoint 2: did the learning phase promote any rules? Stamp the outcome."""

    active = candidate = 0
    try:
        for rule in harness.rules.all():
            active += getattr(rule, "status", "") == "active"
            candidate += getattr(rule, "status", "") == "candidate"
    except Exception as exc:  # noqa: BLE001
        logger.warning("checkpoint-2 rule read failed: %s", exc)
    logger.info("checkpoint-2: rules_active=%d rules_candidate=%d", active, candidate)
    return "no_rules" if active == 0 else f"active={active}"


# Timer helper reused by run_once implementations.
class Stopwatch:
    def __enter__(self) -> Stopwatch:
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.monotonic() - self._start
