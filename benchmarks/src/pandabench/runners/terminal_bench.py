"""Terminal-Bench 2.x runner, driven through Harbor's stable CLI surface.

Harbor owns task download, container lifecycle, task attempts, and verification.
PandaBench invokes one serial Harbor job per phase and ingests its ``result.json``
artifacts into the shared :class:`~pandabench.results.TrialRecord` schema.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..agents.harness_wiring import HarnessWiring
from ..providers.litellm_client import ChatClient
from ..providers.models import ResolvedModel
from ..results import RecordWriter, TrialRecord, resume_key
from .base import SingleTaskRunner, TaskOutcome
from .mock import MockTaskRunner

logger = logging.getLogger("pandabench.terminal_bench")

__all__ = ["TerminalBenchRunner", "build_terminal_runner"]

_REGISTRY_URL = (
    "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"
)
_AGENT_IMPORT = "pandabench.adapters.harbor_agent:PandaBenchAgent"


def _harbor_executable() -> str:
    """Return the Harbor console script installed beside this interpreter.

    A separately installed ``uv tool`` may also be on ``PATH``, but that isolated
    environment cannot import this project's custom agent.  Resolving the sibling
    entry point makes the project dependency load-bearing even when the caller did
    not activate ``.venv`` before invoking ``pandabench-run``.
    """

    executable = Path(sys.executable).with_name("harbor")
    if not executable.is_file():
        raise RuntimeError(
            f"Harbor is not installed beside the active interpreter: {executable}"
        )
    return str(executable)


@lru_cache(maxsize=1)
def _registry() -> list[dict[str, Any]]:
    """Fetch Harbor's public legacy registry once per process."""

    try:
        with urllib.request.urlopen(_REGISTRY_URL, timeout=60) as response:  # noqa: S310
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 - convert network/schema failures to guidance
        raise RuntimeError(f"could not fetch Harbor registry: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError("Harbor registry response is not a dataset list")
    return [row for row in payload if isinstance(row, dict)]


class TerminalBenchRunner(SingleTaskRunner):
    """Runs complete Terminal-Bench phases through ``harbor run``."""

    name = "terminal_bench"

    def list_tasks(self, dataset: str) -> list[str]:
        name, version = _split_dataset_ref(dataset)
        for spec in _registry():
            if spec.get("name") != name or str(spec.get("version")) != version:
                continue
            tasks = spec.get("tasks")
            if not isinstance(tasks, list):
                raise RuntimeError(f"Harbor dataset {dataset!r} has no task list")
            names = [
                str(task["name"])
                for task in tasks
                if isinstance(task, dict) and task.get("name")
            ]
            if not names:
                raise RuntimeError(f"Harbor dataset {dataset!r} contains no named tasks")
            return names
        raise RuntimeError(f"Harbor dataset {dataset!r} was not found in {_REGISTRY_URL}")

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
        effective_backend = model.backend
        pending_tasks = [
            task_id
            for task_id in tasks
            if any(
                not writer.done(
                    resume_key(
                        benchmark, task_id, arm, model.key, effective_backend,
                        seed, trial, phase,
                    )
                )
                for trial in range(k)
            )
        ]
        if not pending_tasks:
            logger.info("Harbor phase %s is already complete; nothing to resume", phase)
            return True

        run_dir = harness_root.parent
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        job_dir = raw_dir / phase
        log_path = run_dir / f"harbor-{phase}.log"
        argv = _harbor_argv(
            dataset=dataset,
            tasks=pending_tasks,
            k=k,
            arm=arm,
            model=model,
            phase=phase,
            raw_dir=raw_dir,
            seed=seed,
            backend=backend,
            harness_root=harness_root,
            max_turns=max_turns,
            noval=noval,
        )

        logger.info("starting Harbor %s phase for %d task(s)", phase, len(pending_tasks))
        launch_error: str | None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            with log_path.open("a", encoding="utf-8") as log_handle:
                if process.stdout is not None:
                    async for raw_line in process.stdout:
                        line = raw_line.decode(errors="replace").rstrip()
                        log_handle.write(line + "\n")
                        log_handle.flush()
                        logger.info("harbor[%s] %s", phase, line)
            returncode = await process.wait()
        except Exception as exc:  # noqa: BLE001 - record every attempt, never abort study
            logger.warning("could not run Harbor %s phase: %s", phase, exc)
            returncode = -1
            launch_error = f"{type(exc).__name__}: {exc}"
        else:
            launch_error = None

        seen = _ingest_results(
            job_dir=job_dir,
            tasks=pending_tasks,
            k=k,
            arm=arm,
            model=model,
            phase=phase,
            writer=writer,
            run_id=run_id,
            seed=seed,
            benchmark=benchmark,
        )
        missing_error = launch_error or (
            f"Harbor exited with status {returncode} without producing this result"
            if returncode != 0
            else "Harbor produced no result.json for this task attempt"
        )
        for task_id in pending_tasks:
            for trial in range(k):
                key = resume_key(
                    benchmark, task_id, arm, model.key, effective_backend,
                    seed, trial, phase,
                )
                if writer.done(key) or (task_id, trial) in seen:
                    continue
                writer.append(
                    _error_record(
                        message=missing_error,
                        run_id=run_id,
                        benchmark=benchmark,
                        task_id=task_id,
                        arm=arm,
                        model=model,
                        seed=seed,
                        trial=trial,
                        phase=phase,
                        returncode=returncode,
                    )
                )

        if returncode != 0:
            logger.warning("Harbor %s phase exited with status %d", phase, returncode)
        return True

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
    ) -> TaskOutcome:
        del task_id, session_id, model, client, max_turns, wiring, preamble
        raise RuntimeError("Terminal-Bench is bulk-driven through run_phase(), not run_once()")

    def outcome_for(self, task_id: str) -> float | None:
        del task_id
        # Harbor's verifier runs after agent.run(), so no in-trial gold signal exists.
        return None

    async def aclose(self) -> None:
        return None


def _split_dataset_ref(dataset: str) -> tuple[str, str]:
    name, marker, version = dataset.rpartition("@")
    if not marker or not name or not version:
        raise ValueError(
            f"Harbor dataset must include a version, e.g. terminal-bench@2.0; got {dataset!r}"
        )
    return name, version


def _harbor_argv(
    *,
    dataset: str,
    tasks: Sequence[str],
    k: int,
    arm: str,
    model: ResolvedModel,
    phase: str,
    raw_dir: Path,
    seed: int,
    backend: str | None,
    harness_root: Path,
    max_turns: int,
    noval: bool,
) -> list[str]:
    argv = [
        _harbor_executable(), "run",
        "-d", dataset,
        "-a", _AGENT_IMPORT,
        "-m", model.key,
        "-k", str(k),
        "-n", "1",
        "-y",
        "-o", str(raw_dir),
        "--job-name", phase,
        "--ak", f"arm={arm}",
        "--ak", f"seed={seed}",
        "--ak", f"model_key={model.key}",
        "--ak", f"capture={str(phase == 'learning').lower()}",
        "--ak", f"harness_root={harness_root.resolve()}",
        "--ak", f"max_turns={max_turns}",
        "--ak", f"noval={str(noval).lower()}",
    ]
    resolved_backend = backend or model.backend
    if resolved_backend is not None:
        argv.extend(("--ak", f"backend={resolved_backend}"))
    for task_id in tasks:
        argv.extend(("-i", task_id))
    return argv


def _ingest_results(
    *,
    job_dir: Path,
    tasks: Sequence[str],
    k: int,
    arm: str,
    model: ResolvedModel,
    phase: str,
    writer: RecordWriter,
    run_id: str,
    seed: int,
    benchmark: str,
) -> set[tuple[str, int]]:
    """Ingest Harbor trial artifacts and return the task/ordinal slots observed."""

    wanted = set(tasks)
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(job_dir.glob("*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not read Harbor result %s: %s", path, exc)
            continue
        task_id = str(payload.get("task_name") or "")
        if task_id in wanted:
            grouped[task_id].append((path, payload))

    seen: set[tuple[str, int]] = set()
    for task_id, siblings in grouped.items():
        siblings.sort(key=lambda item: (_sort_timestamp(item[1].get("started_at")), item[0].name))
        for trial, (path, payload) in enumerate(siblings):
            if trial >= k:
                break
            seen.add((task_id, trial))
            key = resume_key(
                benchmark, task_id, arm, model.key, model.backend, seed, trial, phase
            )
            if writer.done(key):
                continue
            record = _record_from_result(
                payload=payload,
                source=path,
                run_id=run_id,
                benchmark=benchmark,
                task_id=task_id,
                arm=arm,
                model=model,
                seed=seed,
                trial=trial,
                phase=phase,
            )
            writer.append(record)
    return seen


def _record_from_result(
    *,
    payload: dict[str, Any],
    source: Path,
    run_id: str,
    benchmark: str,
    task_id: str,
    arm: str,
    model: ResolvedModel,
    seed: int,
    trial: int,
    phase: str,
) -> TrialRecord:
    verifier = payload.get("verifier_result") or {}
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    reward = _primary_reward(rewards)
    agent_result = payload.get("agent_result") or {}
    if not isinstance(agent_result, dict):
        agent_result = {}
    metadata = agent_result.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    harness = metadata.get("harness")
    if arm != "harness" or not isinstance(harness, dict):
        harness = None

    error = _exception_message(payload.get("exception_info"))
    if reward is None and error is None:
        error = "Harbor result has no verifier reward"
    native = {
        "reward": reward,
        "rewards": rewards if isinstance(rewards, dict) else {},
        "harbor_trial_name": payload.get("trial_name") or source.parent.name,
        "stopped_reason": metadata.get("stopped_reason"),
    }
    return TrialRecord(
        run_id=run_id,
        benchmark=benchmark,
        task_id=task_id,
        arm=arm,
        model=model.key,
        provider=model.provider,
        backend=model.backend,
        resolved_model=model.litellm_model,
        seed=seed,
        trial=trial,
        phase=phase,
        passed=bool(reward is not None and reward >= 1.0 - 1e-6),
        native_metrics=native,
        turns=int(metadata.get("turns", 0) or 0),
        wall_time_s=_wall_time(payload.get("started_at"), payload.get("finished_at")),
        usage={
            "input_tokens": int(agent_result.get("n_input_tokens", 0) or 0),
            "output_tokens": int(agent_result.get("n_output_tokens", 0) or 0),
            "cost_usd": float(agent_result.get("cost_usd", 0.0) or 0.0),
        },
        harness=harness,
        error=error,
    )


def _primary_reward(rewards: Any) -> float | None:
    if not isinstance(rewards, dict) or not rewards:
        return None
    value = rewards.get("reward")
    if value is None:
        value = next(iter(rewards.values()))
    return float(value) if isinstance(value, (int, float)) else None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _sort_timestamp(value: Any) -> datetime:
    return _parse_timestamp(value) or datetime.max.replace(tzinfo=UTC)


def _wall_time(started_at: Any, finished_at: Any) -> float:
    start = _parse_timestamp(started_at)
    finish = _parse_timestamp(finished_at)
    if start is None or finish is None:
        return 0.0
    return max(0.0, (finish - start).total_seconds())


def _exception_message(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    kind = str(value.get("exception_type") or "HarborError")
    message = str(value.get("exception_message") or "unknown Harbor exception")
    return f"{kind}: {message}"


def _error_record(
    *,
    message: str,
    run_id: str,
    benchmark: str,
    task_id: str,
    arm: str,
    model: ResolvedModel,
    seed: int,
    trial: int,
    phase: str,
    returncode: int,
) -> TrialRecord:
    return TrialRecord(
        run_id=run_id,
        benchmark=benchmark,
        task_id=task_id,
        arm=arm,
        model=model.key,
        provider=model.provider,
        backend=model.backend,
        resolved_model=model.litellm_model,
        seed=seed,
        trial=trial,
        phase=phase,
        passed=False,
        native_metrics={"harbor_exit_code": returncode},
        turns=0,
        wall_time_s=0.0,
        usage={"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        harness=None,
        error=message,
    )


def build_terminal_runner(*, dry_run: bool) -> SingleTaskRunner:
    if dry_run:
        return MockTaskRunner("terminal_bench")
    return TerminalBenchRunner()
