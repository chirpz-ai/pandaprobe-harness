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
_BEDROCK_BEARER_ENV = "AWS_BEARER_TOKEN_BEDROCK"
_BEDROCK_BEARER_ERROR = (
    "AWS_BEARER_TOKEN_BEDROCK is unsupported by PandaBench because it overrides "
    "the auto-refreshing AWS_PROFILE_NAME credentials and can expire mid-run; "
    "unset it before launching"
)


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
    parser.add_argument("-d", "--dataset", default=None, help="override configured dataset")
    parser.add_argument("--arm", default="baseline")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("-k", "--k", type=int, default=None, dest="k")
    parser.add_argument(
        "--limit", type=int, default=None, help="run only the first N tasks of the dataset"
    )
    parser.add_argument("--dry-run", action="store_true", help="mock model, no external calls")
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
    if not args.dry_run and os.environ.get(_BEDROCK_BEARER_ENV):
        parser.error(_BEDROCK_BEARER_ERROR)

    study, registry = _load()
    runner = _make_runner(args.benchmark, study, registry, dry_run=args.dry_run)
    asyncio.run(
        runner.run(
            arm=args.arm, model_key=args.model, backend=args.backend, seed=args.seed,
            k=args.k or study.k, limit=args.limit, dry_run=args.dry_run,
            run_id=args.run_id, max_turns_override=args.max_turns,
            dataset_override=args.dataset,
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
                    max_turns_override=6,
                )
            )
    logger.info("smoke complete — see results/runs/ and `make report`")
    return 0


def _matrix(study_path: Path) -> int:
    logger.error(
        "`make matrix` is not yet wired to execute the full cross-product (it would "
        "spend real API budget across every model x seed x arm x benchmark). Run the "
        "full study with the per-arm loop documented in RUNNING.md §6, or run "
        "individual `make <benchmark> ARM=... MODEL=... SEED=...` commands."
    )
    return 1


# -- preflight ----------------------------------------------------------------


def preflight() -> int:
    """Validate tools + credentials + a 1-token ping; fail loud before expensive runs."""

    _load_dotenv()
    checks: list[tuple[str, bool, str]] = []
    bearer_unset = not bool(os.environ.get(_BEDROCK_BEARER_ENV))

    checks.append(("pandaprobe CLI", shutil.which("pandaprobe") is not None, "on PATH"))
    harbor_cli = Path(sys.executable).with_name("harbor")
    checks.append(
        (
            "Harbor CLI",
            harbor_cli.is_file(),
            "installed beside the active interpreter (Terminal-Bench)",
        )
    )
    checks.append(("docker", _docker_ok(), "daemon reachable (Terminal-Bench)"))
    for var in (
        "VERTEXAI_PROJECT",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "PANDAPROBE_API_KEY",
    ):
        checks.append((var, bool(os.environ.get(var)), "set"))
    checks.append(
        (
            "Bedrock bearer token",
            bearer_unset,
            "unset (unsupported)" if bearer_unset else _BEDROCK_BEARER_ERROR,
        )
    )
    bedrock_ok, bedrock_detail = _bedrock_auth()
    checks.append(("Bedrock auth", bedrock_ok, bedrock_detail))

    ping_model = os.environ.get("PANDABENCH_PING_MODEL", "gemini-3.1-flash-lite")
    ok, detail = _ping(ping_model)
    checks.append((f"LLM ping ({ping_model})", ok, detail))

    print("\nPreflight:")
    for name, passed, detail in checks:
        print(f"  [{'OK ' if passed else 'XX '}] {name:28s} {detail}")

    # Hard requirement: pandaprobe CLI + at least one usable provider.
    provider_ok = bool(
        os.environ.get("VERTEXAI_PROJECT")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or bedrock_ok
    )
    hard_ok = shutil.which("pandaprobe") is not None and provider_ok and bearer_unset
    print("\npreflight:", "PASS" if hard_ok else "FAIL (need pandaprobe CLI + >=1 provider)")
    return 0 if hard_ok else 1


def _bedrock_auth() -> tuple[bool, str]:
    """Resolve Bedrock credentials now, so a dead session fails here not hours in.

    Auth goes through ``AWS_PROFILE_NAME`` — LiteLLM's own name for it, NOT the
    standard ``AWS_PROFILE``, which it does not read. LiteLLM deliberately does not
    cache the profile path, so every call re-resolves through boto3 and picks up a
    refreshed SSO token from disk mid-run.

    PandaBench does not support ``AWS_BEARER_TOKEN_BEDROCK``. LiteLLM gives it
    precedence over the profile, and one expired partway through a multi-hour run,
    producing 409 useless trials that still looked complete. Both normal launches
    and preflight reject it.
    """

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_REGION_NAME")
    profile = os.environ.get("AWS_PROFILE_NAME")
    if os.environ.get(_BEDROCK_BEARER_ENV):
        return False, _BEDROCK_BEARER_ERROR
    if not profile:
        return False, "AWS_PROFILE_NAME is unset (needed for Bedrock models)"
    return _sso_profile_ok(profile, region)


def _sso_profile_ok(profile: str, region: str | None) -> tuple[bool, str]:
    """Force a credential resolve for ``profile`` so a dead SSO session fails now."""

    try:
        import boto3  # type: ignore[import-untyped]
    except ImportError:
        return False, f"profile {profile!r} set but boto3 is not installed"
    try:
        credentials = boto3.Session(profile_name=profile).get_credentials()
        if credentials is None:
            return False, f"profile {profile!r} resolves no credentials"
        # Forces a refresh through the SSO token cache; raises if the session died.
        credentials.get_frozen_credentials()
    except Exception as exc:  # noqa: BLE001 - any failure here means "log in again"
        return False, (
            f"profile {profile!r} FAILED ({type(exc).__name__}) -- "
            f"run: aws sso login --profile {profile}"
        )
    # Private attribute, so tolerate its absence rather than depending on it.
    expiry = getattr(credentials, "_expiry_time", None)
    window = f", creds expire {expiry:%H:%M UTC}" if expiry else ""
    note = "" if region else " -- WARNING: AWS_REGION is unset"
    region_label = region or "unset"
    return bool(region), (
        f"profile {profile!r} live, auto-refreshing{window}, region {region_label}{note}"
    )


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

        # 16, not 1: a reasoning model spends output tokens before it can emit a
        # stop, so a 1-token budget 400s ("max_tokens or model output limit was
        # reached") on models that are in fact perfectly reachable.
        litellm.completion(
            model=model.litellm_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16, num_retries=0, timeout=30,
        )
        return True, "completion ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"call failed: {type(exc).__name__}"


# -- pandabench-report / -calibrate (implemented in the report/checkpoint phase) --


def report_main(argv: list[str] | None = None) -> int:
    _configure_logging()
    from .report import DEFAULT_RELAX, aggregate

    parser = argparse.ArgumentParser(prog="pandabench-report")
    parser.add_argument("--runs", default=str(RUN_ROOT))
    parser.add_argument("--out", default=str(BENCH_ROOT / "results" / "summary"))
    parser.add_argument(
        "--relax", type=float, default=DEFAULT_RELAX,
        help=(
            "Report-time pass tolerance as a fraction of a perfect score, applied to "
            "BOTH arms. Re-reads existing records; no re-run needed."
        ),
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.relax < 1.0:
        parser.error("--relax must be in [0, 1)")
    aggregate(Path(args.runs), Path(args.out), relax=args.relax)
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
