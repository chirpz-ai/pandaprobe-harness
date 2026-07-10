"""Command-line entry points: ``pandabench-run`` / ``-report`` / ``-calibrate``.

The Makefile targets are thin sugar over these commands; all logic lives here
and in the modules they call, so every run is reproducible from a plain CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import StudyConfig, load_study
from .providers.models import ModelRegistry, load_registry
from .runners.base import BenchmarkRunner, SingleTaskRunner

logger = logging.getLogger("pandabench")

# benchmarks/  (this file is src/pandabench/cli.py)
BENCH_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCH_ROOT.parent
CONFIGS = BENCH_ROOT / "configs"
RUN_ROOT = BENCH_ROOT / "results" / "runs"
LOCK_PATH = BENCH_ROOT / "uv.lock"

_BENCHMARKS = ("appworld", "terminal_bench", "tau2")


def _load_dotenv() -> None:
    env_file = BENCH_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load %s: %s", env_file, exc)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load(study_path: Path | None = None) -> tuple[StudyConfig, ModelRegistry]:
    study = load_study(study_path or CONFIGS / "study.yaml")
    registry = load_registry(CONFIGS / "models.yaml")
    return study, registry


def build_runner(benchmark: str, *, dry_run: bool) -> SingleTaskRunner:
    """Construct the SingleTaskRunner for a benchmark (its harness may need setup)."""

    if benchmark == "appworld":
        from .runners.appworld import build_appworld_runner

        return build_appworld_runner(dry_run=dry_run)
    if benchmark == "terminal_bench":
        from .runners.terminal_bench import build_terminal_runner

        return build_terminal_runner(dry_run=dry_run)
    if benchmark == "tau2":
        from .runners.tau2 import build_tau2_runner

        return build_tau2_runner(dry_run=dry_run)
    raise ValueError(f"unknown benchmark {benchmark!r} (known: {_BENCHMARKS})")


def _make_runner(
    benchmark: str, study: StudyConfig, registry: ModelRegistry, *, dry_run: bool
) -> BenchmarkRunner:
    return BenchmarkRunner(
        single=build_runner(benchmark, dry_run=dry_run),
        study=study, registry=registry, run_root=RUN_ROOT,
        repo_root=REPO_ROOT, lock_path=LOCK_PATH,
    )


# -- pandabench-run -----------------------------------------------------------


def run_main(argv: list[str] | None = None) -> int:
    _configure_logging()
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="pandabench-run", description="Run a benchmark arm.")
    parser.add_argument("--benchmark", choices=_BENCHMARKS)
    parser.add_argument("--arm", default="baseline")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("-k", "--k", type=int, default=None, dest="k")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--phases", default="learning,eval", help="comma list of phases")
    parser.add_argument("--dry-run", action="store_true", help="mock model, no external calls")
    parser.add_argument("--noval", action="store_true", help="B' ablation: rule_validation off")
    parser.add_argument("--run-id", default=None, help="reuse an existing run_id to resume")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--preflight", action="store_true", help="validate env + creds, then exit")
    parser.add_argument("--smoke", action="store_true", help="run the smoke pipeline and exit")
    parser.add_argument("--matrix", default=None, help="run the full study matrix from study.yaml")
    args = parser.parse_args(argv)

    if args.preflight:
        return preflight()
    if args.smoke:
        return _smoke()
    if args.matrix:
        return _matrix(Path(args.matrix))
    if not args.benchmark:
        parser.error("--benchmark is required (or use --smoke / --preflight / --matrix)")

    study, registry = _load()
    runner = _make_runner(args.benchmark, study, registry, dry_run=args.dry_run)
    phases = tuple(p.strip() for p in args.phases.split(",") if p.strip())
    asyncio.run(
        runner.run(
            arm=args.arm, model_key=args.model, backend=args.backend, seed=args.seed,
            k=args.k or study.k, limit=args.limit, dry_run=args.dry_run, phases=phases,
            run_id=args.run_id, noval=args.noval, max_turns_override=args.max_turns,
        )
    )
    return 0


# -- smoke --------------------------------------------------------------------


def _smoke() -> int:
    """Fast pipeline check: both arms x a tiny task set, all configured benchmarks.

    Runs in ``--dry-run`` (mock model, no external harnesses) so it is fully
    deterministic and dependency-free — the reliable acceptance gate for
    run -> records -> report. Real per-benchmark smokes are separate `make`
    targets that need each benchmark's harness provisioned (`make setup`).
    """

    study, registry = _load()
    benchmarks = [b for b in study.smoke.benchmarks if b in _BENCHMARKS]
    logger.info("smoke: benchmarks=%s arms=%s (dry-run)", benchmarks, study.smoke.arms)
    for benchmark in benchmarks:
        for arm in study.smoke.arms:
            runner = _make_runner(benchmark, study, registry, dry_run=True)
            asyncio.run(
                runner.run(
                    arm=arm, model_key=study.smoke.model, backend=None, seed=1,
                    k=study.smoke.k, limit=study.smoke.tasks, dry_run=True,
                    phases=("learning", "eval"), max_turns_override=6,
                )
            )
    logger.info("smoke complete — see results/runs/ and `make report`")
    return 0


def _matrix(study_path: Path) -> int:
    logger.error("`make matrix` is intentionally left for the operator to launch; "
                 "it spends real API budget. Run per-arm `make <benchmark>` commands instead.")
    return 1


# -- preflight ----------------------------------------------------------------


def preflight() -> int:
    """Validate tools + credentials + a 1-token ping; fail loud before expensive runs."""

    _load_dotenv()
    checks: list[tuple[str, bool, str]] = []

    checks.append(("pandaprobe CLI", shutil.which("pandaprobe") is not None, "on PATH"))
    checks.append(("harbor tool", shutil.which("harbor") is not None, "on PATH (Terminal-Bench)"))
    checks.append(("docker", _docker_ok(), "daemon reachable (Terminal-Bench)"))
    for var in ("VERTEXAI_PROJECT", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PANDAPROBE_API_KEY"):
        checks.append((var, bool(os.environ.get(var)), "set"))

    ping_model = os.environ.get("PANDABENCH_PING_MODEL", "gemini-2.5-flash")
    ok, detail = _ping(ping_model)
    checks.append((f"LLM ping ({ping_model})", ok, detail))

    print("\nPreflight:")
    for name, passed, detail in checks:
        print(f"  [{'OK ' if passed else 'XX '}] {name:28s} {detail}")

    # Hard requirement: pandaprobe CLI + at least one usable provider.
    provider_vars = ("VERTEXAI_PROJECT", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    provider_ok = any(os.environ.get(v) for v in provider_vars)
    hard_ok = shutil.which("pandaprobe") is not None and provider_ok
    print("\npreflight:", "PASS" if hard_ok else "FAIL (need pandaprobe CLI + >=1 provider)")
    return 0 if hard_ok else 1


def _docker_ok() -> bool:
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=False
        ).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _ping(model_key: str) -> tuple[bool, str]:
    try:
        _, registry = _load()
        model = registry.resolve(model_key)
    except Exception as exc:  # noqa: BLE001
        return False, f"resolve failed: {exc}"
    try:
        import litellm

        litellm.completion(
            model=model.litellm_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1, num_retries=0, timeout=30,
        )
        return True, "1-token completion ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"call failed: {type(exc).__name__}"


# -- pandabench-report / -calibrate (implemented in the report/checkpoint phase) --


def report_main(argv: list[str] | None = None) -> int:
    _configure_logging()
    from .report import aggregate

    parser = argparse.ArgumentParser(prog="pandabench-report")
    parser.add_argument("--runs", default=str(RUN_ROOT))
    parser.add_argument("--out", default=str(BENCH_ROOT / "results" / "summary"))
    args = parser.parse_args(argv)
    aggregate(Path(args.runs), Path(args.out))
    return 0


def calibrate_main(argv: list[str] | None = None) -> int:
    _configure_logging()
    _load_dotenv()
    from .checkpoints import run_calibration

    parser = argparse.ArgumentParser(prog="pandabench-calibrate")
    parser.add_argument("--benchmark", required=True, choices=_BENCHMARKS)
    parser.add_argument("--runs", default=str(RUN_ROOT))
    args = parser.parse_args(argv)
    return run_calibration(args.benchmark, Path(args.runs))


if __name__ == "__main__":
    raise SystemExit(run_main())
