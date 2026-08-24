"""Generic run orchestration shared by every benchmark.

A :class:`BenchmarkRunner` drives one ``(benchmark x model x arm x seed)`` run as a
single continuous pass over the benchmark's whole dataset, ``k`` trials per task,
with resumability. In the ``harness`` arm the harness stays live for that entire
pass, so a rule learned at task N can help task N+1 — the in-session healing claim
under test, and the reason there is no learning/eval split or frozen ruleset here.

Task order is therefore load-bearing: it decides what has been learned by task N.
See :meth:`BenchmarkRunner._tasks`.

Benchmark-specific work is confined to a :class:`SingleTaskRunner` (``run_once`` or
the optional bulk ``run_phase`` hook); the harness session lifecycle lives here so
it is identical across benchmarks and so ``run_once`` can be reused verbatim by the
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

from pandaprobe_harness import RuleScopeHint

from ..agents.harness_wiring import AgentWiring, HarnessWiring, ReplayRuleWiring
from ..config import StudyConfig
from ..harness_glue import (
    build_harness,
    build_harness_config,
    harness_root_for,
    make_replay_fn,
    make_session_id,
    make_verifier_fn,
    new_session_namespace,
    resolve_repair_settings,
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

#: The one phase a graded trial can be in. Kept as a recorded field (and a
#: ``resume_key`` component) because dropping it would change every trial's dedup
#: identity and break resume against existing ``records.jsonl`` files.
LIVE_PHASE = "live"

# Credentials/config whose PRESENCE (not value) we fingerprint into the manifest.
_ENV_KEYS = (
    "VERTEXAI_PROJECT", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "AWS_PROFILE_NAME", "AWS_REGION", "CLAUDE_BACKEND",
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
    ) -> bool:
        """Drive the whole run when the benchmark owns task/attempt iteration.

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
    """Orchestrates one continuous, harness-live run for one run tuple."""

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
        tasks = self._tasks(dataset, seed, limit)
        harness_root = harness_root_for(run_dir)
        use_harness = arm == "harness" and not dry_run
        rules_outcome: str | None = None

        logger.info(
            "run %s: arm=%s model=%s seed=%s tasks=%d k=%d (harness live throughout)",
            run_id, arm, model.key, seed, len(tasks), k,
        )

        harness = await self._run_tasks(
            tasks=tasks, k=k, arm=arm, model=model, client=client, writer=writer,
            run_id=run_id, session_namespace=session_namespace, seed=seed,
            backend=backend, max_turns=max_turns, benchmark=benchmark,
            dataset=dataset, harness_root=harness_root, use_harness=use_harness,
        )
        if harness is not None:
            # The only point where nothing holds an environment a replay needs.
            await self._settle(harness, "run")
            rules_outcome = _rules_outcome(harness)
            archive_workspace(harness_root, run_dir / "harness")

        self._write_manifest(
            run_dir=run_dir, run_id=run_id, benchmark=benchmark, model=model,
            arm=arm, seed=seed, backend=backend, rules_outcome=rules_outcome,
            k=k, dry_run=dry_run, dataset=dataset, n_tasks=len(tasks),
        )
        await self._single.aclose()
        logger.info("run %s complete: %d records", run_id, writer.count)
        return run_dir

    # -- the single pass ------------------------------------------------------

    async def _run_tasks(
        self, *, tasks: Sequence[str], k: int, arm: str, model: ResolvedModel,
        client: ChatClient, writer: RecordWriter, run_id: str, seed: int,
        backend: str | None, max_turns: int, benchmark: str, dataset: str,
        harness_root: Path, use_harness: bool, session_namespace: str,
    ) -> Any:
        bulk_hook = getattr(self._single, "run_phase", None)
        if bulk_hook is not None:
            handled = await bulk_hook(
                tasks=tasks, k=k, arm=arm, model=model, phase=LIVE_PHASE,
                dataset=dataset, harness_root=harness_root, writer=writer,
                run_id=run_id, seed=seed, backend=backend, max_turns=max_turns,
                benchmark=benchmark, session_namespace=session_namespace,
            )
            if handled:
                # Harbor's custom agent owns the live Harness in its own process;
                # build this process's read-back view only after Harbor exits, with
                # no replay or verifier (Terminal-Bench supports neither).
                return (
                    self._build_harness(
                        harness_root, benchmark, model, seed,
                        session_namespace, bulk=True,
                    )
                    if use_harness
                    else None
                )

        harness = (
            self._build_harness(
                harness_root, benchmark, model, seed, session_namespace
            )
            if use_harness
            else None
        )
        for task_id in tasks:
            for trial in range(k):
                key = resume_key(
                    benchmark, task_id, arm, model.key, backend, seed, trial, LIVE_PHASE
                )
                if writer.done(key):
                    logger.info("skip (resumed): %s t%d", task_id, trial)
                    continue
                record = await self._run_trial(
                    task_id=task_id, trial=trial, arm=arm, model=model,
                    client=client, harness=harness, run_id=run_id, seed=seed,
                    backend=backend, max_turns=max_turns, benchmark=benchmark,
                    session_namespace=session_namespace,
                )
                writer.append(record)
                status = "PASS" if record.passed else ("ERR" if record.error else "fail")
                logger.info(
                    "%s %s t%d -> %s (%.1fs, $%.4f)",
                    benchmark, task_id, trial, status,
                    record.wall_time_s, record.usage.get("cost_usd", 0.0),
                )
        return harness

    async def _run_trial(
        self, *, task_id: str, trial: int, arm: str, model: ResolvedModel,
        client: ChatClient, harness: Any, run_id: str, seed: int, backend: str | None,
        max_turns: int, benchmark: str, session_namespace: str,
    ) -> TrialRecord:
        session_id = make_session_id(
            session_namespace=session_namespace, benchmark=benchmark, task_id=task_id,
            arm=arm, model_key=model.key, seed=seed, trial=trial, phase=LIVE_PHASE,
        )
        wiring: AgentWiring | None = None
        if harness is not None:
            descriptor = {
                "benchmark": benchmark, "task_id": task_id, "arm": arm,
                "model_key": model.key, "backend": backend, "seed": seed, "trial": trial,
                "run_id": run_id, "phase": LIVE_PHASE,
            }
            wiring = HarnessWiring(
                harness=harness, benchmark=benchmark, task_id=task_id,
                capture=True, replay_descriptor=descriptor,
                # The loop settles every turn through this wiring, so the harness
                # sees the trajectory one trace at a time instead of once at the
                # end — the precondition for the trajectory gate having a series.
                session_id=session_id, flush=client.flush,
                rule_scope_hints=self._single.rule_scope_hints(task_id),
                task_summary=self._single.task_summary(task_id),
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
            repair = settled.repair if settled is not None else None
            telemetry = collect_harness_telemetry(
                harness, session_id, report, repair=repair
            ).to_dict()

        return TrialRecord(
            run_id=run_id, benchmark=benchmark, task_id=task_id, arm=arm,
            model=model.key, provider=model.provider, backend=model.backend,
            resolved_model=model.litellm_model, seed=seed, trial=trial,
            phase=LIVE_PHASE,
            passed=outcome.passed, native_metrics=dict(outcome.native_metrics),
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
        self, harness_root: Path, benchmark: str, model: ResolvedModel,
        seed: int, session_namespace: str, *, bulk: bool = False,
    ) -> Any:
        cfg = build_harness_config(
            harness_root=harness_root, capture=True, study=self._study,
            benchmark=benchmark, repair_model=model.litellm_model,
            repair_overrides=model.repair_overrides,
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
        replay_max_turns = self._study.replay_max_turns(benchmark)

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

    async def _settle(self, harness: Any, label: str) -> None:
        """Await outstanding background evals + validate candidate rules (bounded).

        Each session is scored in a detached background task that can take minutes
        (LLM-judged over many traces); ``refresh``/``on_turn_end`` don't block on it.
        This barrier loops until BOTH the evals and candidate validation are
        quiet, or ``settle_timeout_s`` elapses.

        Waiting on evals alone is not enough, and used to be the whole loop: it
        exits as soon as scores have landed, leaving validation to a trailing
        best-effort join bounded by ``drain_timeout_s`` (15s) — far less than one
        replay-bound round needs. Candidates that had already earned a verdict
        were then recorded as permanently provisional.

        Called once, at the end of the run. Validation itself is not deferred to
        here: the harness spawns it (single-flight, detached) from every handled
        report, so promotion can happen in-session. This barrier only decides what
        is still outstanding when the tasks run out.
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
        # Evals are quiet, so every trial's evidence is in; run validation out to a
        # standstill on the remaining budget.
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

    def _tasks(self, dataset: str, seed: int, limit: int | None) -> list[str]:
        """The whole dataset in one deterministic, arm-independent order.

        Deterministic given ``(dataset, seed)`` so a rerun learns in the same
        sequence, and ``arm`` is deliberately not a parameter so the two arms
        cannot diverge — the report's paired per-task comparison needs both.
        ``limit`` truncates the run.
        """

        shuffled = list(self._single.list_tasks(dataset))
        random.Random(seed).shuffle(shuffled)
        return shuffled if limit is None else shuffled[:limit]

    def _write_manifest(
        self, *, run_dir: Path, run_id: str, benchmark: str, model: ResolvedModel,
        arm: str, seed: int, backend: str | None, rules_outcome: str | None,
        k: int, dry_run: bool, dataset: str, n_tasks: int,
    ) -> None:
        repair = resolve_repair_settings(
            study=self._study,
            repair_model=model.litellm_model,
            repair_overrides=model.repair_overrides,
        )
        manifest = RunManifest(
            run_id=run_id, benchmark=benchmark, model=model.key, arm=arm, seed=seed,
            backend=model.backend, started_at=datetime.now(UTC).isoformat(),
            git_sha=git_sha(self._repo_root), uv_lock_hash=uv_lock_hash(self._lock_path),
            pandaprobe_harness_version=package_version("pandaprobe-harness"),
            litellm_version=package_version("litellm"),
            resolved_config={
                "resolved_model": model.litellm_model, "provider": model.provider,
                "dataset": dataset, "k": k, "dry_run": dry_run,
                "n_tasks": n_tasks,
                "task_default_max_tokens": model.default_max_tokens or 4096,
                **(
                    {
                        "user_simulator_policy": "same_as_agent",
                        "user_simulator_model": model.key,
                        "user_simulator_resolved_model": model.litellm_model,
                        "user_simulator_backend": model.backend,
                    }
                    if benchmark == "tau2"
                    else {}
                ),
                "breach_threshold": self._study.breach_threshold(benchmark),
                "rule_trial_min_sessions": self._study.harness.rule_trial_min_sessions,
                "gate_window": self._study.harness.gate_window,
                **repair.to_manifest(),
                "trace_repair_agent": self._study.harness.trace_repair_agent,
                "managed_repair": True,
                "harness_policy": (
                    "live_throughout" if arm == "harness" else "baseline"
                ),
            },
            env_fingerprint=env_fingerprint(_ENV_KEYS),
            rules_outcome=rules_outcome,
        )
        manifest.write(run_dir / "manifest.json")


def _rules_outcome(harness: Any) -> str:
    """The end-of-run rule lifecycle state, for the manifest.

    Names undecided candidates: "produced nothing" and "produced N candidates
    validation never decided" are different results.
    """

    active = candidate = 0
    try:
        for rule in harness.rules.all():
            status = getattr(rule, "status", "")
            active += status == "active"
            candidate += status == "candidate"
    except Exception as exc:  # noqa: BLE001 - a manifest stamp must not fail a run
        logger.warning("could not read end-of-run rule state: %s", exc)
        return "unknown"
    logger.info("end of run: rules_active=%d rules_candidate=%d", active, candidate)
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
