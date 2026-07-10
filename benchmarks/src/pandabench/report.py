"""Aggregate runs/ into paper-ready summary artifacts.

``make report`` -> ``summary/{all_records.csv, headline.csv,
harness_telemetry.csv, report.md}`` plus an optional learning-curve plot. The
headline table is a benchmark x model x arm view of pass@1 / pass^k with the
harness-vs-baseline delta, bootstrap CIs, and McNemar p; the report prose states
the power caveat, the temperature/nondeterminism note, and the preamble+toolset
token-overhead confound (see docs/benchmark-study-brief.md §5, §9).
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
        for line in records_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_flatten(json.loads(line)))
            except json.JSONDecodeError:
                logger.warning("bad record line in %s", records_file)
    return pd.DataFrame(rows)


def _flatten(rec: dict[str, Any]) -> dict[str, Any]:
    usage = rec.get("usage") or {}
    harness = rec.get("harness") or {}
    flat = {k: v for k, v in rec.items() if k not in ("usage", "harness", "native_metrics")}
    flat["input_tokens"] = usage.get("input_tokens", 0)
    flat["output_tokens"] = usage.get("output_tokens", 0)
    flat["cost_usd"] = usage.get("cost_usd", 0.0)
    flat["has_harness"] = bool(harness)
    for key in ("reliability", "consistency", "breached", "rules_active",
                "rules_candidate", "rules_retired", "notices"):
        flat[f"h_{key}"] = harness.get(key)
    nm = rec.get("native_metrics") or {}
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

    eval_df = df[df["phase"] == "eval"]
    headline = _headline(eval_df)
    headline.to_csv(out_dir / "headline.csv", index=False)

    telemetry = _telemetry(df)
    telemetry.to_csv(out_dir / "harness_telemetry.csv", index=False)

    deltas = _paired(eval_df)
    _plot_learning_curve(df, out_dir)
    _write_report_md(out_dir, headline, telemetry, deltas, df)
    logger.info("wrote summary artifacts to %s", out_dir)


def _first_trial_passes(group: pd.DataFrame) -> list[bool]:
    """One pass/fail per (seed, task) using trial 0."""

    firsts = group[group["trial"] == 0]
    return [bool(p) for p in firsts["passed"].tolist()]


def _all_trial_passes(group: pd.DataFrame) -> list[list[bool]]:
    """Per (seed, task): the list of pass/fail across trials."""

    out: list[list[bool]] = []
    for _, sub in group.groupby(["seed", "task_id"]):
        out.append([bool(p) for p in sub.sort_values("trial")["passed"].tolist()])
    return out


def _headline(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if eval_df.empty:
        return pd.DataFrame(rows)
    for (benchmark, model, arm), group in eval_df.groupby(["benchmark", "model", "arm"]):
        rows.append(
            {
                "benchmark": benchmark, "model": model, "arm": arm,
                "n_tasks": group[["seed", "task_id"]].drop_duplicates().shape[0],
                "pass_at_1": round(pass_at_1(_first_trial_passes(group)), 4),
                "pass_hat_k": round(pass_hat_k(_all_trial_passes(group)), 4),
                "mean_cost_usd": round(float(group["cost_usd"].mean()), 6),
                "mean_input_tokens": round(float(group["input_tokens"].mean()), 1),
                "n_error": int((group["error"].notna() & (group["error"] != "")).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["benchmark", "model", "arm"]).reset_index(drop=True)


def _paired(eval_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Harness-vs-baseline paired pass@1 comparison per (benchmark, model)."""

    results: list[dict[str, Any]] = []
    if eval_df.empty:
        return results
    for (benchmark, model), group in eval_df.groupby(["benchmark", "model"]):
        first = group[group["trial"] == 0]
        by_arm: dict[str, dict[tuple[Any, Any], bool]] = defaultdict(dict)
        for _, row in first.iterrows():
            by_arm[row["arm"]][(row["seed"], row["task_id"])] = bool(row["passed"])
        base, harn = by_arm.get("baseline", {}), by_arm.get("harness", {})
        keys = sorted(set(base) & set(harn))
        if not keys:
            continue
        pairs = [(base[k], harn[k]) for k in keys]
        delta = paired_delta(pairs)
        results.append({"benchmark": benchmark, "model": model, **delta.to_dict()})
    return results


def _telemetry(df: pd.DataFrame) -> pd.DataFrame:
    harn = df[df["arm"] == "harness"]
    rows: list[dict[str, Any]] = []
    if harn.empty:
        return pd.DataFrame(rows)
    for (benchmark, model, phase), group in harn.groupby(["benchmark", "model", "phase"]):
        rows.append(
            {
                "benchmark": benchmark, "model": model, "phase": phase,
                "trials": len(group),
                "rules_active_max": _safe_max(group["h_rules_active"]),
                "rules_candidate_max": _safe_max(group["h_rules_candidate"]),
                "rules_retired_max": _safe_max(group["h_rules_retired"]),
                "notices_total": _safe_sum(group["h_notices"]),
                "breach_rate": _safe_mean(group["h_breached"]),
            }
        )
    return pd.DataFrame(rows)


def _safe_max(s: pd.Series) -> float:
    vals = s.dropna()
    return float(vals.max()) if not vals.empty else 0.0


def _safe_sum(s: pd.Series) -> float:
    return float(s.dropna().sum())


def _safe_mean(s: pd.Series) -> float:
    vals = s.dropna()
    return round(float(vals.mean()), 4) if not vals.empty else 0.0


def _plot_learning_curve(df: pd.DataFrame, out_dir: Path) -> None:
    learn = df[(df["arm"] == "harness") & (df["phase"] == "learning")]
    if learn.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for benchmark, group in learn.groupby("benchmark"):
            ordered = group.sort_values(["seed", "task_id", "trial"]).reset_index(drop=True)
            cumulative = ordered["passed"].astype(float).expanding().mean()
            ax.plot(range(len(cumulative)), cumulative, marker="o", label=str(benchmark))
        ax.set_xlabel("learning task-trial index")
        ax.set_ylabel("cumulative pass rate (arm B)")
        ax.set_title("Learning-phase pass rate (harness arm)")
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
    lines += ["## Headline (eval phase)", "", _md_table(headline), ""]

    lines += ["## Harness vs baseline (paired pass@1)", ""]
    if deltas:
        dframe = pd.DataFrame(deltas)[
            ["benchmark", "model", "n_pairs", "rate_a", "rate_b", "delta",
             "ci_low", "ci_high", "p_value", "underpowered"]
        ]
        lines += [_md_table(dframe), ""]
    else:
        lines += ["_No baseline/harness pairs yet._", ""]

    lines += ["## Harness telemetry", "", _md_table(telemetry), ""]
    lines += ["## Cost / overhead", "", _md_table(_overhead(df)), ""]

    lines += [
        "## Methodology notes",
        "",
        "- **Power caveat.** At ~30-40 eval tasks, McNemar detects only large "
        "deltas (~10+ points); small effects are underpowered even pooling seeds. "
        "Results are directional — read the bootstrap CIs, not just point deltas.",
        "- **Nondeterminism.** Current Claude models reject `temperature`, so "
        "trial-to-trial variance comes from natural model nondeterminism; no "
        "sampler seed is forced.",
        "- **Preamble confound.** The arm-B harness preamble + 14 tools cost "
        "context/tokens every turn (see cost/overhead), which can depress arm B "
        "on long tasks independent of rule quality.",
        "- **Checkpoints.** Checkpoint 1 (metric<->failure calibration) and "
        "Checkpoint 2 (rule promotion; `learning_outcome` in each manifest) gate "
        "the full matrix; see IMPLEMENTATION_NOTES.md.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _overhead(df: pd.DataFrame) -> pd.DataFrame:
    eval_df = df[df["phase"] == "eval"]
    rows: list[dict[str, Any]] = []
    if eval_df.empty:
        return pd.DataFrame(rows)
    for (benchmark, model), group in eval_df.groupby(["benchmark", "model"]):
        by_arm = group.groupby("arm")["input_tokens"].mean()
        base = float(by_arm.get("baseline", float("nan")))
        harn = float(by_arm.get("harness", float("nan")))
        rows.append(
            {
                "benchmark": benchmark, "model": model,
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
        return df.to_markdown(index=False)
    except Exception:  # noqa: BLE001 - tabulate may be absent
        return "```\n" + df.to_string(index=False) + "\n```"
