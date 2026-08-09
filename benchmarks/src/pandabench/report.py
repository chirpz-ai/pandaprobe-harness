"""Aggregate runs/ into paper-ready summary artifacts.

``make report`` -> ``summary/{all_records.csv, headline.csv,
harness_telemetry.csv, report.md}`` plus an optional learning-curve plot. Tables
cover the whole run; the harness arm is live throughout, so there is no eval phase
to filter down to.

The headline table is a benchmark x dataset x model x arm view of three metrics
side by side — strict ``pass@1``/``pass^k`` (the benchmark's own verdict), a
relaxed pass rate (ours, harness-arm only), and the mean pass ratio — with the
harness-vs-baseline paired delta, bootstrap CIs, and McNemar p. The report prose
states the power caveat, the temperature/nondeterminism note, the
preamble+toolset token-overhead confound, and that only the strict metric is
comparable to published numbers (see ``IMPLEMENTATION_NOTES.md``).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import paired_delta, pass_at_1, pass_hat_k

logger = logging.getLogger("pandabench.report")

__all__ = ["aggregate", "load_records"]


def load_records(runs_dir: Path) -> pd.DataFrame:
    """Flatten every runs/*/records.jsonl into one DataFrame."""

    rows: list[dict[str, Any]] = []
    for records_file in sorted(runs_dir.glob("*/records.jsonl")):
        dataset, tolerance = _manifest_facts(records_file.parent / "manifest.json")
        for line in records_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = _flatten(json.loads(line))
                row["dataset"] = str(row.get("dataset") or dataset)
                row["pass_tolerance"] = tolerance
                rows.append(row)
            except json.JSONDecodeError:
                logger.warning("bad record line in %s", records_file)
    return pd.DataFrame(rows)


def _manifest_facts(path: Path) -> tuple[str, int]:
    """The run's dataset and the pass tolerance its records were scored with."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown", 0
    resolved = manifest.get("resolved_config") or {}
    if not isinstance(resolved, dict):
        return "unknown", 0
    raw = resolved.get("pass_tolerance")
    return str(resolved.get("dataset") or "unknown"), int(raw) if isinstance(raw, int) else 0


def _flatten(rec: dict[str, Any]) -> dict[str, Any]:
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
    relaxed = nm.get("passed_relaxed")
    flat["passed_relaxed"] = bool(rec.get("passed")) if relaxed is None else bool(relaxed)
    ratio = nm.get("pass_ratio")
    flat["pass_ratio"] = (
        float(ratio) if isinstance(ratio, (int, float)) else float(bool(rec.get("passed")))
    )
    flat["native_metrics"] = json.dumps(nm)
    return flat


def aggregate(runs_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_records(runs_dir)
    if df.empty:
        logger.warning("no records under %s; writing empty summary", runs_dir)
        (out_dir / "report.md").write_text("# PandaBench results\n\n_No records yet._\n")
        for name in ("all_records.csv", "headline.csv", "harness_telemetry.csv"):
            (out_dir / name).write_text("")
        return

    df.to_csv(out_dir / "all_records.csv", index=False)

    headline = _headline(df)
    headline.to_csv(out_dir / "headline.csv", index=False)

    telemetry = _telemetry(df)
    telemetry.to_csv(out_dir / "harness_telemetry.csv", index=False)

    deltas = _paired(df)
    _plot_learning_curve(df, out_dir)
    _write_report_md(out_dir, headline, telemetry, deltas, df)
    logger.info("wrote summary artifacts to %s", out_dir)


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


def _headline(df: pd.DataFrame) -> pd.DataFrame:
    """Strict, relaxed, and mean-ratio metrics per (benchmark, dataset, model, arm).

    ``pass_at_1``/``pass_hat_k`` come from ``passed``, the benchmark's own verdict,
    and are the only figures comparable to published results. ``pass_tolerance``
    names the definition the ``*_relaxed`` columns used — 0 in the baseline arm,
    where they equal the strict ones.
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
                "pass_tolerance": int(group["pass_tolerance"].max()),
                "pass_at_1_relaxed": round(
                    pass_at_1(_first_trial_passes(group, "passed_relaxed")), 4
                ),
                "pass_hat_k_relaxed": round(
                    pass_hat_k(_all_trial_passes(group, "passed_relaxed")), 4
                ),
                "mean_pass_ratio": round(float(group["pass_ratio"].mean()), 4),
                "mean_cost_usd": round(float(group["cost_usd"].mean()), 6),
                "mean_input_tokens": round(float(group["input_tokens"].mean()), 1),
                "n_error": int((group["error"].notna() & (group["error"] != "")).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _paired(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Harness-vs-baseline comparison per (benchmark, dataset, model) and metric.

    Computed twice: on the benchmark's own ``passed``, and on ``passed_relaxed``.
    The relaxed row is the intended harness-vs-baseline read — the harness arm at
    its configured ``pass_tolerance`` against a baseline at the benchmark's own
    criteria. The two arms therefore use different definitions by design, so the
    relaxed delta includes that gap; ``pass_tolerance`` in the headline names the
    definition each arm used.
    """

    results: list[dict[str, Any]] = []
    if df.empty:
        return results
    for (benchmark, dataset, model), group in df.groupby(
        ["benchmark", "dataset", "model"]
    ):
        first = group[group["trial"] == 0]
        for metric, column in (("strict", "passed"), ("relaxed", "passed_relaxed")):
            by_arm: dict[str, dict[tuple[Any, Any], bool]] = defaultdict(dict)
            for _, row in first.iterrows():
                by_arm[row["arm"]][(row["seed"], row["task_id"])] = bool(row[column])
            base, harn = by_arm.get("baseline", {}), by_arm.get("harness", {})
            keys = sorted(set(base) & set(harn))
            if not keys:
                continue
            delta = paired_delta([(base[k], harn[k]) for k in keys])
            results.append(
                {
                    "benchmark": benchmark, "dataset": dataset, "model": model,
                    "metric": metric, **delta.to_dict(),
                }
            )
    return results


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
                # Promotions and retirements are counted from validation verdicts,
                # not inferred from status high-water marks: a status count cannot
                # say whether validation ever reached a candidate.
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
    deltas: list[dict[str, Any]], df: pd.DataFrame,
) -> None:
    lines = ["# PandaBench results", ""]
    lines += [
        "## Headline (whole run)",
        "",
        "`pass_at_1` / `pass_hat_k` are the **benchmark's own** all-or-nothing "
        "verdict and the only figures comparable to published results. "
        "`pass_at_1_relaxed` / `pass_hat_k_relaxed` / `mean_pass_ratio` are "
        "**ours** — see the relaxed-metric note below.",
        "",
        _md_table(headline),
        "",
    ]

    lines += ["## Harness vs baseline (paired pass@1, strict and relaxed)", ""]
    if deltas:
        dframe = pd.DataFrame(deltas)[
            ["benchmark", "dataset", "model", "metric", "n_pairs", "rate_a", "rate_b",
             "delta", "ci_low", "ci_high", "p_value", "underpowered"]
        ]
        lines += [
            _md_table(dframe),
            "",
            "`metric=relaxed` pairs the harness arm at its configured "
            "`pass_tolerance` against a baseline scored by the benchmark's own "
            "criteria — see the relaxed-metric note below.",
            "",
        ]
    else:
        lines += ["_No baseline/harness pairs yet._", ""]

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
        "- **Relaxed metric is ours, and harness-arm only.** `passed` is the "
        "benchmark's own verdict (AppWorld: all tests pass; tau2: "
        "`is_successful(reward)`; Terminal-Bench: `reward >= 1.0`) and is applied "
        "identically in both arms. `passed_relaxed` allows up to `pass_tolerance` "
        "missed tests **in the harness arm only** — the baseline is always scored "
        "at tolerance 0, where it equals `passed`. It exists because the strict "
        "verdict is floored: in a measured 456-trial AppWorld run, 72% of trials "
        "failed by exactly one test, so pass@1 and pass^k could not move. The "
        "`relaxed` paired row is therefore an intentionally asymmetric comparison "
        "(harness at tolerance N vs baseline at 0), useful for seeing how the "
        "harness does under a given tolerance but NOT comparable to published "
        "numbers; read `pass_tolerance` in the headline for the definition each arm "
        "used. Benchmarks with no partial-credit signal report it equal to `passed` "
        "rather than inventing one.",
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
