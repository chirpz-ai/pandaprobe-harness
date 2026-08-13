"""Aggregate runs/ into paper-ready summary artifacts.

``make report`` -> ``summary/{all_records.csv, headline.csv, relax_sweep.csv,
harness_telemetry.csv, report.md}`` plus an optional learning-curve plot. Tables
cover the whole run; the harness arm is live throughout, so there is no eval phase
to filter down to.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeGuard

import pandas as pd

from .metrics import paired_delta, pass_any_k, pass_at_1, pass_hat_k

logger = logging.getLogger("pandabench.report")

__all__ = ["DEFAULT_RELAX", "RELAX_SWEEP", "aggregate", "load_records"]
DEFAULT_RELAX = 0.10
RELAX_SWEEP = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.34, 0.5)


def load_records(runs_dir: Path) -> pd.DataFrame:
    """Flatten every runs/*/records.jsonl into one DataFrame."""

    rows: list[dict[str, Any]] = []
    for records_file in sorted(runs_dir.glob("*/records.jsonl")):
        run_dir = records_file.parent
        dataset = _manifest_dataset(run_dir / "manifest.json")
        harbor = _harbor_test_counts(run_dir)
        for line in records_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = _flatten(json.loads(line), harbor)
                row["dataset"] = str(row.get("dataset") or dataset)
                rows.append(row)
            except json.JSONDecodeError:
                logger.warning("bad record line in %s", records_file)
    return pd.DataFrame(rows)


_PYTEST_BANNER = re.compile(r"^=+ (.*?) =+$", re.M)
_PYTEST_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors)\b")
_VERIFIER_TAIL_BYTES = 64_000


def _harbor_test_counts(run_dir: Path) -> dict[str, tuple[int, int]]:
    """Per-test pass counts for a Terminal-Bench run, keyed by Harbor trial name.
    """

    counts: dict[str, tuple[int, int]] = {}
    raw = run_dir / "raw"
    if not raw.is_dir():
        return counts
    for log in raw.glob("*/*/verifier/test-stdout.txt"):
        parsed = _parse_pytest_counts(log)
        if parsed is not None:
            counts[log.parent.parent.name] = parsed
    if counts:
        logger.info("recovered per-test counts for %d trials in %s", len(counts), run_dir.name)
    return counts


def _parse_pytest_counts(path: Path) -> tuple[int, int] | None:
    """``(passed, passed + failed)`` from a verifier log, or None if unreadable.

    Takes the LAST banner carrying both a count and a duration: a verifier may
    reinstall packages and print further banners after the test session. Skipped and
    xfailed are excluded from the denominator — they are not signal about the agent.
    """

    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - _VERIFIER_TAIL_BYTES))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    body = None
    for match in _PYTEST_BANNER.finditer(text):
        candidate = match.group(1)
        if " in " in candidate and _PYTEST_COUNT.search(candidate):
            body = candidate
    if body is None:
        return None
    tally = {kind: int(n) for n, kind in _PYTEST_COUNT.findall(body)}
    passed = tally.get("passed", 0)
    failed = tally.get("failed", 0) + tally.get("error", 0) + tally.get("errors", 0)
    total = passed + failed
    return (passed, total) if total > 0 else None


def _manifest_dataset(path: Path) -> str:
    """The run's dataset, for records that predate the column."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    resolved = manifest.get("resolved_config") or {}
    if not isinstance(resolved, dict):
        return "unknown"
    return str(resolved.get("dataset") or "unknown")


def _flatten(
    rec: dict[str, Any], harbor: dict[str, tuple[int, int]] | None = None
) -> dict[str, Any]:
    usage = rec.get("usage") or {}
    harness = rec.get("harness") or {}
    flat = {k: v for k, v in rec.items() if k not in ("usage", "harness", "native_metrics")}
    flat["input_tokens"] = usage.get("input_tokens", 0)
    flat["output_tokens"] = usage.get("output_tokens", 0)
    flat["cost_usd"] = usage.get("cost_usd", 0.0)
    flat["has_harness"] = bool(harness)
    for key in ("mode", "ruleset_hash", "breached", "gate_breached",
                "rules_active", "rules_candidate", "rules_retired", "notices",
                "repair_episodes"):
        flat[f"h_{key}"] = harness.get(key)
    flat["h_resolution_counts"] = json.dumps(harness.get("resolution_counts") or {})
    flat["h_rules_by_scope"] = json.dumps(harness.get("rules_by_scope") or {})
    validation = harness.get("validation") or {}
    for key in (
        "rounds", "promoted", "retired", "replays",
        "candidate_not_exercised", "env_wait_timeouts", "budget_exhausted_rounds",
    ):
        flat[f"h_validation_{key}"] = validation.get(key)
    flat["h_validation_pending_reasons"] = json.dumps(
        validation.get("pending_reasons") or {}
    )
    # Flatten each resolved trace metric into its own column.
    for name, value in (harness.get("scores") or {}).items():
        flat[f"h_score_{name}"] = value
    nm = rec.get("native_metrics") or {}
    tests = _recovered_tests(nm, harbor)
    flat["score"] = _score(nm, bool(rec.get("passed")), tests)
    flat["score_is_graded"] = _score_is_graded(nm, tests)
    flat["native_metrics"] = json.dumps(nm)
    return flat


def _recovered_tests(
    native: dict[str, Any], harbor: dict[str, tuple[int, int]] | None
) -> tuple[int, int] | None:
    """This trial's ``(passed, total)`` from the archived verifier log, if any."""

    if not harbor:
        return None
    name = native.get("harbor_trial_name")
    return harbor.get(str(name)) if name else None


def _score(
    native: dict[str, Any], passed: bool, tests: tuple[int, int] | None = None
) -> float:
    """The trial's continuous 0-1 quality, on one scale for every benchmark.

    Resolved most-graded-first, because each benchmark thresholds away detail it
    already computed and the finest surviving signal is the useful one:

    1. ``num_tests``/``num_passes`` — AppWorld's per-test counts (37 observed levels).
    2. ``tests`` — Terminal-Bench per-test counts recovered from the archived
       verifier log, since Harbor's own ``reward`` is 1.0-or-nothing (9 levels).
    3. ``pass_ratio``.
    4. ``reward_breakdown`` — tau2's per-component rewards. Its ``reward`` is exactly
       ``min(DB, COMMUNICATE)`` on all 200 measured records, so the mean of the
       components is strictly finer, though only 3 levels: 0.0, 0.5, 1.0.
    5. ``reward``, then the boolean verdict, so nothing is ever missing.

    A perfect score is 1.0 in every case, which is what lets one relaxation fraction
    mean the same thing across benchmarks. The counts outrank ``pass_ratio`` so this
    agrees with ``_score_is_graded``, which reads them.
    """

    total, passes = native.get("num_tests"), native.get("num_passes")
    if _is_number(total) and _is_number(passes) and float(total) > 0:
        return float(passes) / float(total)
    if tests is not None and tests[1] > 0:
        return tests[0] / tests[1]
    ratio = native.get("pass_ratio")
    if _is_number(ratio):
        return float(ratio)
    components = _reward_components(native)
    if components:
        return sum(components) / len(components)
    reward = native.get("reward")
    if _is_number(reward):
        return float(reward)
    return float(passed)


def _reward_components(native: dict[str, Any]) -> list[float]:
    """tau2's per-component rewards, or empty when there is no useful breakdown.

    A single component carries no more information than ``reward`` itself, so it is
    rejected: the point is partial credit across components.
    """

    breakdown = native.get("reward_breakdown")
    if not isinstance(breakdown, dict):
        return []
    values = [float(v) for v in breakdown.values() if _is_number(v)]
    return values if len(values) > 1 else []


def _is_number(value: Any) -> TypeGuard[float]:
    """Numeric and not a bool — ``isinstance(True, int)`` is True in Python."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _score_is_graded(
    native: dict[str, Any], tests: tuple[int, int] | None = None
) -> bool:
    """Whether this trial's score carries partial credit that relaxing can act on."""

    total = native.get("num_tests")
    if _is_number(total) and float(total) > 1:
        return True
    if tests is not None and tests[1] > 1:
        return True
    return len(_reward_components(native)) > 1


def aggregate(
    runs_dir: Path,
    out_dir: Path,
    *,
    relax: float = DEFAULT_RELAX,
    sweep: tuple[float, ...] = RELAX_SWEEP,
) -> None:
    """Write the summary artifacts. ``relax`` re-thresholds the harness arm only."""

    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_records(runs_dir)
    if df.empty:
        logger.warning("no records under %s; writing empty summary", runs_dir)
        (out_dir / "report.md").write_text("# PandaBench results\n\n_No records yet._\n")
        for name in (
            "all_records.csv", "headline.csv", "relax_sweep.csv", "harness_telemetry.csv",
        ):
            (out_dir / name).write_text("")
        return

    df["passed_relaxed"] = _relaxed(df, relax)
    df.to_csv(out_dir / "all_records.csv", index=False)

    headline = _headline(df, relax)
    headline.to_csv(out_dir / "headline.csv", index=False)

    sweep_table = _relax_sweep(df, sweep)
    sweep_table.to_csv(out_dir / "relax_sweep.csv", index=False)

    telemetry = _telemetry(df)
    telemetry.to_csv(out_dir / "harness_telemetry.csv", index=False)

    deltas = _paired(df)
    _plot_learning_curve(df, out_dir)
    _write_report_md(out_dir, headline, telemetry, deltas, df, relax, sweep_table)
    logger.info("wrote summary artifacts to %s (relax=%.2f)", out_dir, relax)


def _relaxed(df: pd.DataFrame, relax: float) -> pd.Series:
    """Relaxed pass at tolerance ``relax``."""

    lenient = df["score"].astype(float) >= (1.0 - float(relax) - 1e-9)
    is_harness = df["arm"].astype(str) == "harness"
    strict = df["passed"].astype(bool)
    return (lenient & is_harness) | (strict & ~is_harness)


def _first_trial_passes(group: pd.DataFrame, column: str = "passed") -> list[bool]:
    """One pass/fail per (seed, task) using trial 0."""

    firsts = group[group["trial"] == 0]
    return [bool(p) for p in firsts[column].tolist()]


def _all_trial_passes(group: pd.DataFrame, column: str = "passed") -> list[list[bool]]:
    """Per (seed, task): the list of pass/fail across trials."""

    out: list[list[bool]] = []
    for _, sub in group.groupby(["seed", "task_id"]):
        out.append([bool(p) for p in sub.sort_values("trial")[column].tolist()])
    return out


def _headline(df: pd.DataFrame, relax: float) -> pd.DataFrame:
    """Per (benchmark, dataset, model, arm) metrics, strictest first.

    ``pass_at_1``/``pass_hat_k`` come from ``passed``, the benchmark's own verdict,
    """

    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(rows)
    keys = ["benchmark", "dataset", "model", "arm"]
    for (benchmark, dataset, model, arm), group in df.groupby(keys):
        rows.append(
            {
                "benchmark": benchmark, "dataset": dataset, "model": model, "arm": arm,
                "n_tasks": group[["seed", "task_id"]].drop_duplicates().shape[0],
                "pass_at_1": round(pass_at_1(_first_trial_passes(group)), 4),
                "pass_hat_k": round(pass_hat_k(_all_trial_passes(group)), 4),
                "pass_any_k": round(pass_any_k(_all_trial_passes(group)), 4),
                "relax": relax,
                "graded_score": bool(group["score_is_graded"].any()),
                "pass_at_1_relaxed": round(
                    pass_at_1(_first_trial_passes(group, "passed_relaxed")), 4
                ),
                "pass_hat_k_relaxed": round(
                    pass_hat_k(_all_trial_passes(group, "passed_relaxed")), 4
                ),
                "mean_score": round(float(group["score"].mean()), 4),
                "mean_cost_usd": round(float(group["cost_usd"].mean()), 6),
                "mean_input_tokens": round(float(group["input_tokens"].mean()), 1),
                "n_error": int((group["error"].notna() & (group["error"] != "")).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _relax_sweep(df: pd.DataFrame, sweep: tuple[float, ...]) -> pd.DataFrame:
    """Harness positioning across a range of tolerances."""

    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(rows)
    for (benchmark, dataset, model), group in df.groupby(["benchmark", "dataset", "model"]):
        graded = bool(group["score_is_graded"].any())
        for relax in sweep:
            flags = _relaxed(group, relax)
            first = group[group["trial"] == 0]
            by_arm = {
                arm: flags.loc[sub.index].mean()
                for arm, sub in first.groupby("arm")
            }
            base = by_arm.get("baseline")
            harn = by_arm.get("harness")
            rows.append(
                {
                    "benchmark": benchmark, "dataset": dataset, "model": model,
                    "relax": relax, "graded_score": graded,
                    "baseline_pass_at_1": None if base is None else round(float(base), 4),
                    "harness_pass_at_1": None if harn is None else round(float(harn), 4),
                    "delta": (
                        None if base is None or harn is None
                        else round(float(harn) - float(base), 4)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Harness-vs-baseline comparison per (benchmark, dataset, model) and metric."""

    results: list[dict[str, Any]] = []
    if df.empty:
        return results
    for (benchmark, dataset, model), group in df.groupby(
        ["benchmark", "dataset", "model"]
    ):
        first = group[group["trial"] == 0]
        for metric, column in (("strict", "passed"), ("relaxed", "passed_relaxed")):
            by_arm: dict[Any, dict[tuple[Any, Any], bool]] = defaultdict(dict)
            for _, row in first.iterrows():
                by_arm[row["arm"]][(row["seed"], row["task_id"])] = bool(row[column])
            results.extend(
                _delta_row(benchmark, dataset, model, metric, by_arm)
            )
        any_k: dict[Any, dict[tuple[Any, Any], bool]] = defaultdict(dict)
        for (arm, seed, task_id), sub in group.groupby(["arm", "seed", "task_id"]):
            any_k[arm][(seed, task_id)] = bool(sub["passed"].any())
        results.extend(_delta_row(benchmark, dataset, model, "any_k", any_k))
    return results


def _delta_row(
    benchmark: Any, dataset: Any, model: Any, metric: str,
    by_arm: dict[Any, dict[tuple[Any, Any], bool]],
) -> list[dict[str, Any]]:
    """One paired row, or nothing when the two arms share no task."""

    base, harn = by_arm.get("baseline", {}), by_arm.get("harness", {})
    keys = sorted(set(base) & set(harn))
    if not keys:
        return []
    delta = paired_delta([(base[k], harn[k]) for k in keys])
    return [
        {
            "benchmark": benchmark, "dataset": dataset, "model": model,
            "metric": metric, **delta.to_dict(),
        }
    ]


def _telemetry(df: pd.DataFrame) -> pd.DataFrame:
    harn = df[df["arm"] == "harness"]
    rows: list[dict[str, Any]] = []
    if harn.empty:
        return pd.DataFrame(rows)
    keys = ["benchmark", "dataset", "model", "phase"]
    for (benchmark, dataset, model, phase), group in harn.groupby(keys):
        rows.append(
            {
                "benchmark": benchmark, "dataset": dataset, "model": model, "phase": phase,
                "trials": len(group),
                "mode": _mode(group["h_mode"]),
                "rules_active_max": _safe_max(group["h_rules_active"]),
                "rules_candidate_max": _safe_max(group["h_rules_candidate"]),
                "rules_retired_max": _safe_max(group["h_rules_retired"]),
                "notices_total": _safe_sum(group["h_notices"]),
                "repair_episodes_total": _safe_sum(group["h_repair_episodes"]),
                "breach_rate": _safe_mean(group["h_breached"]),
                "validation_rounds_max": _safe_max(group["h_validation_rounds"]),
                "promotions_max": _safe_max(group["h_validation_promoted"]),
                "retirements_max": _safe_max(group["h_validation_retired"]),
                "replays_max": _safe_max(group["h_validation_replays"]),
                "unexercised_replays_max": _safe_max(
                    group["h_validation_candidate_not_exercised"]
                ),
                "env_wait_timeouts_max": _safe_max(group["h_validation_env_wait_timeouts"]),
            }
        )
    return pd.DataFrame(rows)


def _mode(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna() if str(value)})
    return ",".join(values) if values else "legacy_live"


def _safe_max(s: pd.Series) -> float:
    vals = s.dropna()
    return float(vals.max()) if not vals.empty else 0.0


def _safe_sum(s: pd.Series) -> float:
    return float(s.dropna().sum())


def _safe_mean(s: pd.Series) -> float:
    vals = s.dropna()
    return round(float(vals.mean()), 4) if not vals.empty else 0.0


def _plot_learning_curve(df: pd.DataFrame, out_dir: Path) -> None:
    """Cumulative harness-arm pass rate over the run, in task order.

    A genuine in-session learning curve: the harness is live for every task plotted.
    """

    learn = df[df["arm"] == "harness"]
    if learn.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for (benchmark, dataset), group in learn.groupby(["benchmark", "dataset"]):
            ordered = group.sort_values(["seed", "task_id", "trial"]).reset_index(drop=True)
            cumulative = ordered["passed"].astype(float).expanding().mean()
            ax.plot(
                range(len(cumulative)), cumulative, marker="o",
                label=f"{benchmark}/{dataset}",
            )
        ax.set_xlabel("task-trial index (run order)")
        ax.set_ylabel("cumulative pass rate (arm B)")
        ax.set_title("In-session pass rate (harness arm, live throughout)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "learning_curve.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        logger.warning("learning-curve plot skipped: %s", exc)


def _write_report_md(
    out_dir: Path, headline: pd.DataFrame, telemetry: pd.DataFrame,
    deltas: list[dict[str, Any]], df: pd.DataFrame, relax: float,
    sweep_table: pd.DataFrame,
) -> None:
    lines = ["# PandaBench results", ""]
    lines += [
        "## Headline (whole run)",
        "",
        "`pass_at_1` / `pass_hat_k` are the **benchmark's own** all-or-nothing "
        "verdict and the only figures comparable to published results. "
        "`pass_any_k`, `pass_at_1_relaxed` / `pass_hat_k_relaxed` "
        f"(at `relax={relax}`) and `mean_score` are **ours** — see the "
        "relaxed-metric note below.",
        "",
        _md_table(headline),
        "",
    ]

    lines += ["## Harness vs baseline (paired: strict, relaxed, any-of-k)", ""]
    if deltas:
        dframe = pd.DataFrame(deltas)[
            ["benchmark", "dataset", "model", "metric", "n_pairs", "rate_a", "rate_b",
             "delta", "ci_low", "ci_high", "p_value", "underpowered"]
        ]
        lines += [
            _md_table(dframe),
            "",
            "`rate_a` is baseline, `rate_b` harness. **`strict` and `any_k` score both "
            "arms by the same rule** and are the comparable rows. **`relaxed` relaxes "
            "the harness arm only** — the baseline stays at the benchmark's own verdict "
            "— so its delta includes the definition gap by design. Diagnostic, not a "
            "result.",
            "",
        ]
    else:
        lines += ["_No baseline/harness pairs yet._", ""]

    lines += [
        "## Relaxation sweep",
        "",
        "Paired `pass@1` as the tolerance loosens.",
        _md_table(sweep_table),
        "",
    ]

    lines += ["## Harness telemetry", "", _md_table(telemetry), ""]
    lines += ["## Cost / overhead", "", _md_table(_overhead(df)), ""]

    lines += [
        "## Methodology notes",
        "",
        "- **Harness live throughout.** The harness arm runs the complete "
        "evaluation and repair loop — notices, package-owned managed repair, and "
        "candidate validation — across the benchmark's whole dataset in one "
        "continuous pass. There is no learning/eval split and no frozen ruleset, "
        "because the claim under test is in-session healing: a rule learned at "
        "task N helping task N+1 of the same run.",
        "- **Task order is load-bearing and shared.** Order is a pure function of "
        "`(dataset, seed)` and is identical in both arms, which is what makes the "
        "paired per-task comparison valid. Vary `seed` to counterbalance which "
        "tasks the harness sees early.",
        "- **Power caveat.** McNemar detects only large deltas (~10+ points) at "
        "these task counts; small effects are underpowered even pooling seeds. "
        "Results are directional — read the bootstrap CIs, not just point deltas.",
        "- **Nondeterminism.** The study does not send `temperature` to Claude, so "
        "trial-to-trial variance comes from natural model nondeterminism; no "
        "sampler seed is forced.",
        "- **Preamble confound.** The arm-B harness preamble plus four read-only "
        "rule tools cost context/tokens on every trial (see cost/overhead), which "
        "can depress arm B on long tasks independent of rule quality.",
        "- **Checkpoints.** Checkpoint 1 (metric<->failure calibration) and "
        "Checkpoint 2 (rule promotion; `rules_outcome` in each manifest) gate "
        "the full matrix; see IMPLEMENTATION_NOTES.md.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _overhead(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(rows)
    for (benchmark, dataset, model), group in df.groupby(
        ["benchmark", "dataset", "model"]
    ):
        by_arm = group.groupby("arm")["input_tokens"].mean()
        base = float(by_arm.get("baseline", float("nan")))
        harn = float(by_arm.get("harness", float("nan")))
        rows.append(
            {
                "benchmark": benchmark, "dataset": dataset, "model": model,
                "baseline_input_tokens": round(base, 1) if base == base else None,
                "harness_input_tokens": round(harn, 1) if harn == harn else None,
                "overhead_tokens": round(harn - base, 1) if harn == harn and base == base else None,
                "mean_cost_baseline": round(float(
                    group[group["arm"] == "baseline"]["cost_usd"].mean()), 6),
                "mean_cost_harness": round(float(
                    group[group["arm"] == "harness"]["cost_usd"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(none)_"
    try:
        return str(df.to_markdown(index=False))
    except Exception:  # noqa: BLE001 - tabulate may be absent
        return "```\n" + str(df.to_string(index=False)) + "\n```"
