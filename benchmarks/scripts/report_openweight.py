"""Build the openweight report from the audited, complete benchmark runs.

Run from ``benchmarks/`` with ``make report-openweight``.
The explicit run IDs keep subsequent pilots or reruns from silently changing this
snapshot. Missing arm cells are retained as blank rows in the report matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pandabench import report

BENCH_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = BENCH_ROOT / "results" / "runs"
OUT_DIR = BENCH_ROOT / "results" / "summary" / "openweight"
REPORT_RELAX = 0.20

MODELS = (
    "gpt-oss-20b",
    "gpt-oss-120b",
    "kimi-k2.5",
    "nemotron-3-super-120b",
    "qwen3-32b",
)
DATASETS = (
    ("tau2", "airline", 50),
    ("tau2", "retail", 114),
    ("tau2", "telecom", 114),
    ("terminal_bench", "terminal-bench-sample@2.0", 10),
)
ARMS = ("baseline", "harness")
METRICS = ("strict", "relaxed", "any_k")
Cell = tuple[str, str, str, str]
Pair = tuple[str, str, str]

# These two differences are accepted for this report. Both arms still share the
# same task IDs, seed, K, and task model, so the per-task deltas can be computed.
ACCEPTED_PAIR_DIFFERENCES: dict[Pair, set[str]] = {
    ("tau2", "airline", "gpt-oss-20b"): {"simulator"},
    ("tau2", "telecom", "gpt-oss-20b"): {"order"},
}

# One run per recorded cell. Seven harness cells have no run and are intentionally
# omitted here; the report matrix supplies their blank rows.
RUN_IDS: dict[Cell, str] = {
    ("tau2", "airline", "gpt-oss-20b", "baseline"): "tau2_gpt-oss-20b_baseline_1_20260824-064958",
    ("tau2", "airline", "gpt-oss-20b", "harness"): "tau2_gpt-oss-20b_harness_1_20260825-054944",
    ("tau2", "retail", "gpt-oss-20b", "baseline"): "tau2_gpt-oss-20b_baseline_1_20260826-034447",
    ("tau2", "retail", "gpt-oss-20b", "harness"): "tau2_gpt-oss-20b_harness_1_20260826-050356",
    ("tau2", "telecom", "gpt-oss-20b", "baseline"): "tau2_gpt-oss-20b_baseline_1_20260826-034513",
    ("tau2", "telecom", "gpt-oss-20b", "harness"): "tau2_gpt-oss-20b_harness_1_20260916-070754",
    (
        "terminal_bench",
        "terminal-bench-sample@2.0",
        "gpt-oss-20b",
        "baseline",
    ): "terminal_bench_gpt-oss-20b_baseline_1_20260824-063433",
    (
        "terminal_bench",
        "terminal-bench-sample@2.0",
        "gpt-oss-20b",
        "harness",
    ): "terminal_bench_gpt-oss-20b_harness_1_20260825-052803",
    ("tau2", "airline", "gpt-oss-120b", "baseline"): "tau2_gpt-oss-120b_baseline_1_20260826-034624",
    ("tau2", "retail", "gpt-oss-120b", "baseline"): "tau2_gpt-oss-120b_baseline_1_20260826-034523",
    ("tau2", "telecom", "gpt-oss-120b", "baseline"): "tau2_gpt-oss-120b_baseline_1_20260826-034559",
    (
        "terminal_bench",
        "terminal-bench-sample@2.0",
        "gpt-oss-120b",
        "baseline",
    ): "terminal_bench_gpt-oss-120b_baseline_1_20260824-063228",
    ("tau2", "airline", "kimi-k2.5", "baseline"): "tau2_kimi-k2.5_baseline_1_20260904-050943",
    ("tau2", "airline", "kimi-k2.5", "harness"): "tau2_kimi-k2.5_harness_1_20260922-054843",
    ("tau2", "retail", "kimi-k2.5", "baseline"): "tau2_kimi-k2.5_baseline_1_20260916-201347",
    ("tau2", "retail", "kimi-k2.5", "harness"): "tau2_kimi-k2.5_harness_1_20260922-054934",
    ("tau2", "telecom", "kimi-k2.5", "baseline"): "tau2_kimi-k2.5_baseline_1_20260916-201359",
    (
        "terminal_bench",
        "terminal-bench-sample@2.0",
        "kimi-k2.5",
        "baseline",
    ): "terminal_bench_kimi-k2.5_baseline_1_20260904-051044",
    (
        "terminal_bench",
        "terminal-bench-sample@2.0",
        "kimi-k2.5",
        "harness",
    ): "terminal_bench_kimi-k2.5_harness_1_20260905-060359",
    (
        "tau2",
        "airline",
        "nemotron-3-super-120b",
        "baseline",
    ): "tau2_nemotron-3-super-120b_baseline_1_20260827-062153",
    (
        "tau2",
        "airline",
        "nemotron-3-super-120b",
        "harness",
    ): "tau2_nemotron-3-super-120b_harness_1_20260901-024047",
    (
        "tau2",
        "retail",
        "nemotron-3-super-120b",
        "baseline",
    ): "tau2_nemotron-3-super-120b_baseline_1_20260828-052100",
    (
        "tau2",
        "telecom",
        "nemotron-3-super-120b",
        "baseline",
    ): "tau2_nemotron-3-super-120b_baseline_1_20260828-052109",
    (
        "terminal_bench",
        "terminal-bench-sample@2.0",
        "nemotron-3-super-120b",
        "baseline",
    ): "terminal_bench_nemotron-3-super-120b_baseline_1_20260827-062055",
    (
        "terminal_bench",
        "terminal-bench-sample@2.0",
        "nemotron-3-super-120b",
        "harness",
    ): "terminal_bench_nemotron-3-super-120b_harness_1_20260901-023927",
    ("tau2", "airline", "qwen3-32b", "baseline"): "tau2_qwen3-32b_baseline_1_20260827-062133",
    ("tau2", "airline", "qwen3-32b", "harness"): "tau2_qwen3-32b_harness_1_20260904-050527",
    ("tau2", "retail", "qwen3-32b", "baseline"): "tau2_qwen3-32b_baseline_1_20260828-051919",
    ("tau2", "retail", "qwen3-32b", "harness"): "tau2_qwen3-32b_harness_1_20260905-060236",
    ("tau2", "telecom", "qwen3-32b", "baseline"): "tau2_qwen3-32b_baseline_1_20260828-052013",
    ("tau2", "telecom", "qwen3-32b", "harness"): "tau2_qwen3-32b_harness_1_20260906-162858",
    (
        "terminal_bench",
        "terminal-bench-sample@2.0",
        "qwen3-32b",
        "baseline",
    ): "terminal_bench_qwen3-32b_baseline_1_20260827-061945",
    (
        "terminal_bench",
        "terminal-bench-sample@2.0",
        "qwen3-32b",
        "harness",
    ): "terminal_bench_qwen3-32b_harness_1_20260903-060438",
}

# These simulator settings were verified in configs/models.yaml at each run's
# manifest git_sha. Older manifests do not all stamp the simulated-user model.
SIMULATOR_BY_SHA = {
    "071b8e7257226d2d18a2fe3d9a0490074541b839": "same_as_agent",
    "9148ef67547ebf8feafe7c68b79ba7f59ce7f424": "gemini-3.1-flash-lite",
    "00bbae7a3af978d02206ba4ddbf853e6582e2640": "nova-lite",
}


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    manifest: dict[str, Any]
    order: tuple[str, ...]
    simulator: str
    records: int
    errors: int


def _cells() -> list[Cell]:
    return [
        (benchmark, dataset, model, arm)
        for benchmark, dataset, _ in DATASETS
        for model in MODELS
        for arm in ARMS
    ]


def _read_run(cell: Cell, run_id: str, expected_tasks: int) -> RunMeta:
    directory = RUNS_DIR / run_id
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    benchmark, dataset, model, arm = cell
    config = manifest["resolved_config"]
    assert (manifest["benchmark"], config["dataset"], manifest["model"], manifest["arm"]) == cell
    assert manifest["run_id"] == run_id
    assert manifest["seed"] == 1 and config["k"] == 4
    assert config["n_tasks"] == expected_tasks and not config["dry_run"]
    assert len(rows) == expected_tasks * 4, f"incomplete run: {run_id}"
    slots = {(str(row["task_id"]), row["trial"]) for row in rows}
    assert len(slots) == len(rows), f"duplicate task/trial slots: {run_id}"
    assert {trial for _, trial in slots} == {0, 1, 2, 3}
    assert all(
        (row["run_id"], row["benchmark"], row["model"], row["arm"], row["seed"], row["phase"])
        == (run_id, benchmark, model, arm, 1, "live")
        for row in rows
    )
    order = tuple(str(row["task_id"]) for row in rows if row["trial"] == 0)
    assert len(order) == expected_tasks and len(set(order)) == expected_tasks
    assert all((task, trial) in slots for task in order for trial in range(4))
    simulator = ""
    if benchmark == "tau2":
        simulator = str(config.get("user_simulator_model") or SIMULATOR_BY_SHA[manifest["git_sha"]])
        if simulator == "same_as_agent":
            simulator = model
    return RunMeta(
        run_id,
        manifest,
        order,
        simulator,
        len(rows),
        sum(bool(row.get("error")) for row in rows),
    )


def _pair_reason(pair: Pair, runs: dict[Cell, RunMeta]) -> str:
    benchmark, dataset, model = pair
    baseline = runs.get((benchmark, dataset, model, "baseline"))
    harness = runs.get((benchmark, dataset, model, "harness"))
    if baseline is None or harness is None:
        return "missing arm"
    a = baseline.manifest["resolved_config"]
    b = harness.manifest["resolved_config"]
    for key in ("resolved_model", "k", "n_tasks", "task_default_max_tokens"):
        if a.get(key) != b.get(key):
            return f"different {key}"
    if set(baseline.order) != set(harness.order):
        return "different task set"
    accepted = ACCEPTED_PAIR_DIFFERENCES.get(pair, set())
    if baseline.simulator != harness.simulator and "simulator" not in accepted:
        return "different simulated user"
    if baseline.order != harness.order and "order" not in accepted:
        return "different task order"
    return "paired"


def _headline_matrix(headline: pd.DataFrame) -> pd.DataFrame:
    by_cell = {
        (row["benchmark"], row["dataset"], row["model"], row["arm"]): row.to_dict()
        for _, row in headline.iterrows()
    }
    assert set(by_cell) == set(RUN_IDS)
    columns = list(headline.columns)
    rows = []
    for cell in _cells():
        values = by_cell.get(cell)
        if values is None:
            values = dict.fromkeys(columns, "")
            values.update(zip(("benchmark", "dataset", "model", "arm"), cell, strict=True))
        elif cell[3] == "baseline":
            values["relax"] = ""
        rows.append(values)
    return pd.DataFrame(rows, columns=columns)


def _telemetry_matrix(telemetry: pd.DataFrame) -> pd.DataFrame:
    by_pair = {
        (row["benchmark"], row["dataset"], row["model"]): row.to_dict()
        for _, row in telemetry.iterrows()
    }
    expected = {cell[:3] for cell in RUN_IDS if cell[3] == "harness"}
    assert set(by_pair) == expected
    columns = list(telemetry.columns)
    rows = []
    for benchmark, dataset, _ in DATASETS:
        for model in MODELS:
            values = by_pair.get((benchmark, dataset, model))
            if values is None:
                values = dict.fromkeys(columns, "")
                values.update(
                    {
                        "benchmark": benchmark,
                        "dataset": dataset,
                        "model": model,
                        "phase": "live",
                    }
                )
            rows.append(values)
    return pd.DataFrame(rows, columns=columns)


def _paired_matrix(df: pd.DataFrame, pair_reasons: dict[Pair, str]) -> list[dict[str, Any]]:
    valid = {pair for pair, reason in pair_reasons.items() if reason == "paired"}
    mask = [
        (benchmark, dataset, model) in valid
        for benchmark, dataset, model in zip(
            df["benchmark"], df["dataset"], df["model"], strict=True
        )
    ]
    measured = {
        (row["benchmark"], row["dataset"], row["model"], row["metric"]): row
        for row in report._paired(df.loc[mask])
    }
    rows = []
    for benchmark, dataset, _ in DATASETS:
        for model in MODELS:
            for metric in METRICS:
                key = (benchmark, dataset, model, metric)
                row = measured.get(key)
                if row is None:
                    row = {
                        "benchmark": benchmark,
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "n_pairs": "",
                        "rate_a": "",
                        "rate_b": "",
                        "delta": "",
                        "ci_low": "",
                        "ci_high": "",
                        "p_value": "",
                        "underpowered": "",
                    }
                rows.append(row)
    assert len(measured) == len(valid) * len(METRICS)
    return rows


def _methodology_notes(relax: float) -> list[str]:
    return [
        "- **Task universe.** τ²-bench uses the full airline (50), retail (114), "
        "and telecom (114) domains. Terminal-Bench uses the 10-task "
        "`terminal-bench-sample@2.0`, not its 89-task full benchmark.",
        "- **Study size.** Every recorded cell has seed 1 and four trials per task. "
        "`pass_at_1` uses trial 0; `pass_hat_k` requires all four to pass. "
        "One seed and the small Terminal-Bench sample limit precision.",
        "- **Harness live throughout.** The harness arm can learn and validate rules "
        "during one continuous pass. Results depend on the recorded task order; "
        "the plot follows JSONL order within each run.",
        "- **Pairing.** Paired rows require both arms, the same task model and "
        "task IDs, seed, and trial count. The `gpt-oss-20b` airline simulated-user "
        "difference and telecom task-order difference are included by study "
        "choice; blank paired cells lack a harness run.",
        "- **Errors.** `n_error` counts recorded trial errors; these trials remain "
        "in the reported denominators with their recorded native verdicts. High "
        "error counts can confound a model or harness comparison.",
        "- **Metrics.** Strict `pass_at_1` and `pass_hat_k` use the benchmark's "
        f"verdict. `relax={relax:.2f}` counts a harness trial with score at least "
        f"{1 - relax:.2f} "
        "as a relaxed pass; baseline remains at its native strict verdict. "
        "`mean_score` uses available partial-credit signals where present.",
        "- **Cost.** Mean cost and input tokens are task-agent usage per trial; "
        "τ² simulated-user cost is separately present in native metrics.",
        "- **Power.** The paired bootstrap intervals and McNemar tests are "
        "directional, especially at 10 Terminal-Bench tasks. No multiple-seed "
        "or multiple-model correction is applied.",
    ]


def main() -> None:
    expected = {(benchmark, dataset): tasks for benchmark, dataset, tasks in DATASETS}
    assert set(RUN_IDS) <= set(_cells())
    assert len(set(RUN_IDS.values())) == len(RUN_IDS)
    runs = {cell: _read_run(cell, run_id, expected[cell[:2]]) for cell, run_id in RUN_IDS.items()}
    pair_reasons = {
        (benchmark, dataset, model): _pair_reason((benchmark, dataset, model), runs)
        for benchmark, dataset, _ in DATASETS
        for model in MODELS
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = []
    for cell in _cells():
        benchmark, dataset, model, arm = cell
        meta = runs.get(cell)
        inventory.append(
            {
                "benchmark": benchmark,
                "dataset": dataset,
                "model": model,
                "arm": arm,
                "run_id": "" if meta is None else meta.run_id,
                "records": "" if meta is None else meta.records,
                "expected_records": expected[(benchmark, dataset)] * 4,
                "n_error": "" if meta is None else meta.errors,
                "git_sha": "" if meta is None else meta.manifest["git_sha"],
                "simulated_user": "" if meta is None else meta.simulator,
                "pair_status": pair_reasons[(benchmark, dataset, model)],
            }
        )
    pd.DataFrame(inventory).to_csv(OUT_DIR / "run_inventory.csv", index=False)

    df = report.load_records(RUNS_DIR, run_ids=set(RUN_IDS.values()))
    assert len(df) == sum(meta.records for meta in runs.values())
    relax = REPORT_RELAX
    df["passed_relaxed"] = report._relaxed(df, relax)
    baseline = df["arm"] == "baseline"
    assert (df.loc[baseline, "passed_relaxed"] == df.loc[baseline, "passed"]).all()
    df.to_csv(OUT_DIR / "all_records.csv", index=False)

    headline = _headline_matrix(report._headline(df, relax))
    headline.to_csv(OUT_DIR / "headline.csv", index=False)
    sweep_values = tuple(sorted(set((*report.RELAX_SWEEP, relax))))
    sweep = report._relax_sweep(df, sweep_values)
    for pair, reason in pair_reasons.items():
        if reason not in ("paired", "missing arm"):
            mask = (
                (sweep["benchmark"] == pair[0])
                & (sweep["dataset"] == pair[1])
                & (sweep["model"] == pair[2])
            )
            sweep.loc[mask, "delta"] = float("nan")
    sweep.to_csv(OUT_DIR / "relax_sweep.csv", index=False)

    telemetry = _telemetry_matrix(report._telemetry(df))
    telemetry.to_csv(OUT_DIR / "harness_telemetry.csv", index=False)
    deltas = _paired_matrix(df, pair_reasons)
    overhead = report._overhead(df)
    for pair, reason in pair_reasons.items():
        if reason != "paired":
            mask = (
                (overhead["benchmark"] == pair[0])
                & (overhead["dataset"] == pair[1])
                & (overhead["model"] == pair[2])
            )
            overhead.loc[mask, "overhead_tokens"] = float("nan")

    report._plot_learning_curve(df, OUT_DIR)
    n_paired = sum(reason == "paired" for reason in pair_reasons.values())
    n_errors = sum(meta.errors for meta in runs.values())
    strict_rows = [row for row in deltas if row["metric"] == "strict" and row["n_pairs"] != ""]
    assert len(strict_rows) == n_paired
    excluding_zero = [row for row in strict_rows if row["ci_low"] > 0 or row["ci_high"] < 0]
    terminal_runs = [meta for cell, meta in runs.items() if cell[0] == "terminal_bench"]
    terminal_errors = sum(meta.errors for meta in terminal_runs)
    terminal_records = sum(meta.records for meta in terminal_runs)
    terminal_error_text = df.loc[
        (df["benchmark"] == "terminal_bench") & df["error"].notna(), "error"
    ].astype(str)
    terminal_timeouts = int(terminal_error_text.str.startswith("AgentTimeoutError").sum())
    terminal_loops = int(terminal_error_text.str.startswith("AgentLoopError").sum())
    assert terminal_timeouts + terminal_loops == terminal_errors
    interval_summary = (
        f"None of the {n_paired} strict `pass_at_1` deltas has a 95% paired "
        "bootstrap interval excluding zero. "
        if not excluding_zero
        else f"{len(excluding_zero)} of the {n_paired} strict `pass_at_1` deltas "
        "have a 95% paired bootstrap interval excluding zero. "
    )
    overview = [
        "## Results summary",
        "",
        interval_summary + "The largest positive "
        "point delta is `gpt-oss-20b` on τ² retail (+7.9 percentage points), "
        "where baseline recorded 35 trial errors and harness 7; that difference "
        "complicates attribution. On the 10-task Terminal-Bench sample, harness "
        "strict `pass_at_1` is equal or lower in the four available pairs, and "
        f"{terminal_errors} of {terminal_records} terminal trials have errors "
        f"({terminal_timeouts} timeouts and {terminal_loops} agent loop errors). "
        "These are single-seed results.",
        "",
        f"The relaxed tables use `relax={relax:.2f}` for harness trials: a score "
        f"of at least {1 - relax:.2f} counts as a relaxed pass. Baseline rows "
        "retain the benchmark's strict verdict.",
        "",
    ]
    extra = [
        "## Run coverage and data quality",
        "",
        f"{len(runs)} of 40 arm cells have recorded runs; the seven missing harness "
        "cells are blank in the headline table. "
        f"{n_paired} of 20 model/dataset pairs have both arms included in the "
        "paired analysis. "
        "Exact run IDs, record counts, errors, revisions, and pairing status are "
        "in [run_inventory.csv](run_inventory.csv).",
        "",
        "- Missing harness runs: `gpt-oss-120b` on all three τ² domains and "
        "Terminal-Bench; `kimi-k2.5` on τ² telecom; "
        "`nemotron-3-super-120b` on τ² retail and telecom.",
        "- `gpt-oss-20b` τ² airline used itself as the baseline simulated user "
        "and Gemini 3.1 Flash Lite as the harness simulated user. "
        "The comparison is included by study choice.",
        "- `gpt-oss-20b` τ² telecom ran the same 114 tasks but in a different "
        "order across arms. The comparison is included by study choice.",
        f"- {n_errors} of {len(df)} recorded trials have an error. "
        "The tables retain their recorded verdicts; inspect `n_error` and "
        "[run_inventory.csv](run_inventory.csv) before interpreting deltas.",
        "",
    ]
    report._write_report_md(
        OUT_DIR,
        headline,
        telemetry,
        deltas,
        df,
        relax,
        sweep,
        title="PandaBench openweight results",
        headline_note=(
            "`pass_at_1` / `pass_hat_k` use each benchmark's own all-or-nothing "
            "verdict. External comparisons require matching task sets and, for "
            "τ²-bench, the same simulated-user model. `pass_any_k`, "
            f"`pass_at_1_relaxed` / `pass_hat_k_relaxed` (at `relax={relax:.2f}`) and "
            "`mean_score` are PandaBench diagnostics."
        ),
        overview=overview,
        overhead=overhead,
        extra_sections=extra,
        methodology_notes=_methodology_notes(relax),
    )
    print(f"Wrote {OUT_DIR} ({len(runs)} runs, {n_paired} paired results, {n_errors} errors)")


if __name__ == "__main__":
    main()
