# Running PandaBench

Run everything from `benchmarks/`. Each benchmark runs two
arms — `baseline` (no harness) and `harness` — over the same tasks/models.

## 0. Prerequisites

- **uv** and **Python 3.13**.
- `pandaprobe` **CLI** on PATH: `curl -fsSL https://cli.pandaprobe.com/install.sh | sh`
- **Credentials** (put in `benchmarks/.env`; see `.env.example`):
  - Vertex AI: `gcloud auth application-default login` + `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`
  - OpenAI: `OPENAI_API_KEY`
  - Claude via Bedrock (default): `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION`
  - Claude via Anthropic (optional fallback): `ANTHROPIC_API_KEY`
  - `PANDAPROBE_API_KEY` (required for the `harness` arm)
- **Docker** running — Terminal-Bench only.



## 1. One-time setup

```bash
cp .env.example .env          # fill in credentials
make setup                    # uv sync (including Harbor), isolated AppWorld env, preflight
uv run pandabench-run --preflight   # re-check tools + creds + a 1-token ping
```

`make setup` prints two env vars for AppWorld — add them to `.env`:

```bash
export PANDABENCH_APPWORLD_PYTHON=$HOME/.pandabench/awenv/bin/python
export APPWORLD_ROOT=$HOME/.pandabench/appworld
```

The benchmark installs `pandaprobe-harness==0.9.0` from PyPI — the released
artifact, not `../src` and not a local build. Confirm with
`uv pip show pandaprobe-harness` (version 0.9.0, location under `.venv`). A study
run therefore measures exactly what a user installs; editing `../src` has no
effect on it. To move to a newer harness release, bump the pin in `pyproject.toml`
and run `uv lock && uv sync --all-extras --group dev`, then re-record affected
runs — `manifest.json` stamps `pandaprobe_harness_version` per run.

The `harness` arm is **live for the benchmark's whole dataset** — one continuous
pass, no learning/eval split, no frozen ruleset. Task order is a pure function of
`(dataset, seed)` and identical in both arms, and it is load-bearing: it decides
what has been learned by the time task N runs.

Harness runs use package-owned managed repair. By default PandaBench reuses
the resolved task model and explicitly sets `repair_reasoning_effort: "none"`,
which current OpenAI reasoning models require when using function tools through
the PandaProbe-wrapped LiteLLM chat-completions API. Choose another current
LiteLLM model deliberately when needed. The task agent sees only a stable
capability note and four read-only rule tools. Rule bodies and the expanded scope
index are never force-injected; list/read/search/status happen only if the task
model chooses them. Listing returns the canonical `rules.md` and generated
scope references.

Managed repair chooses each rule's scope from the failure evidence, inside the
repair call it already makes: `global` for broadly reusable rules, a contextual name
(application, workflow, domain) when the rule belongs to that context, `scoped` when
no meaningful name can be determined. Benchmarks pass the task statement and safe
metadata as context; they do not name the file.

Validation promotes or retires candidates — repair cannot. `validation_round_budget_s`
bounds replay work per round and the remaining candidates get the cheap forward-trial
verdict, and `replay_env_wait_timeout_s` keeps time spent queueing for AppWorld's
single world out of the replay execution budget. Validation runs *during* the pass,
so a rule can be promoted mid-run; a single settlement at the **end of the run**
then waits (within `settle_timeout_s`) for outstanding evals and in-flight
validation before the workspace is archived, and logs a warning if it could not.



## 2. Smoke test (pipeline check, no external harnesses)

```bash
make smoke        # 2 tasks x 1 trial x both arms, all benchmarks, dry-run (mock model)
make report       # regenerate results/summary/
```



## 3. Run a benchmark

Model keys: `gemini-3.1-flash-lite`, `gemini-3.5-flash`, `gemini-3.1-pro`,
`gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`, `claude-opus-5`,
`claude-sonnet-5`, `claude-haiku-4-5`.

**Knobs:**


| Knob       | Meaning                                                                              | Default                                           |
| ---------- | ------------------------------------------------------------------------------------ | ------------------------------------------------- |
| `ARM`      | `baseline` (no harness) or `harness`                                                 | `baseline`                                        |
| `MODEL`    | a model key from the list above                                                      | `gemini-3.1-flash-lite`                           |
| `SEED`     | shuffles task order (same order in both arms); run several (1, 2, 3) as replicates   | `1`                                               |
| `K`        | trials per task — `pass@1` = first trial passed, `pass^k` = all K passed             | `4`                                               |
| `DATASET`  | override the configured task universe (for example, Terminal-Bench's 10-task sample) | benchmark config                                  |
| `LIMIT`    | run only the **first N tasks** of the dataset; **omit to run all of it**              | unset (all)                                       |
| `MAXTURNS` | per-task **agent-turn cap** (how long the agent works on one task)                   | `study.yaml` `max_turns` (100 for all benchmarks) |
| `BACKEND`  | **Claude only**: `bedrock` or `anthropic`                                            | `bedrock`                                         |


- Omitting `DATASET` selects that benchmark's single configured default from
`configs/study.yaml`; it does **not** run every available dataset.
- `LIMIT` **controls the number of tasks.** It truncates the run to the first N
tasks of the seeded order: `LIMIT=5` runs 5 tasks, and `K=4` makes that 20 trials
per arm. On the raw CLI, use the equivalent `--limit 5`.
- `LIMIT` **≠ task length.** To make each task run longer, raise `MAXTURNS`, e.g.
`MAXTURNS=60`, or bump `max_turns` in `configs/study.yaml` for that benchmark.
- For a paired A/B comparison, keep `MODEL`, `DATASET`, `SEED`, `K`, `LIMIT`, and
`MAXTURNS` identical; change only `ARM`. Both arms then run the same tasks in the
same order, which the paired statistics require.
- **OpenAI / Gemini route automatically** by their `models.yaml` prefix
(`openai/…` → OpenAI API via `OPENAI_API_KEY`; `vertex_ai/…` → Vertex). Claude
defaults to Bedrock; use `BACKEND=anthropic` only when intentionally falling back
to the direct Anthropic API. Passing `BACKEND` to an OpenAI/Gemini model errors.
- Bedrock's short-term API key is region-bound and expires after at most 12 hours.
Generate it for the same region as `AWS_REGION`, and refresh it before long runs.
- Bedrock routes Claude through AWS's `global.*` system inference profiles. The
catalog's underlying `anthropic.*` foundation-model IDs identify the model but
reject on-demand invocation; the profile IDs are the callable on-demand targets.
- Verify Bedrock specifically before spending on a benchmark:
  `PANDABENCH_PING_MODEL=claude-sonnet-5 uv run pandabench-run --preflight`.
- The third configured model is Claude **Haiku 4.5**, not 4.6; both its official
Anthropic ID and the supplied Bedrock catalog ID identify it as 4.5.

The examples below are paired medium-pilot runs: the first 5 tasks of the dataset,
once per arm. Drop `LIMIT` to run the dataset whole, which is what a real study does.

### AppWorld

Needs the isolated env from `make setup` (`PANDABENCH_APPWORLD_PYTHON`, `APPWORLD_ROOT`).

```bash
make appworld ARM=baseline MODEL=gpt-5.6-terra DATASET=dev SEED=1 K=1 LIMIT=5
make appworld ARM=harness MODEL=gpt-5.6-terra DATASET=dev SEED=1 K=1 LIMIT=5
```


| `DATASET` value  | Tasks | Use                     |
| ---------------- | ----- | ----------------------- |
| `train`          | 90    | AppWorld training split |
| `dev`            | 57    | Configured default      |
| `test_normal`    | 168   | Normal test split       |
| `test_challenge` | 417   | Challenge test split    |

The selected split is run **whole** — PandaBench never re-partitions one, so results
for a split stay comparable to published AppWorld numbers for it. `dev` is the
default; `test_normal` gives ~3× the tasks, hence more in-session learning runway and
more statistical power, at ~3× the spend.




### Terminal-Bench (Harbor)

Needs **Docker running**. Harbor is installed into this project so its CLI can import
the custom `pandabench` agent. Real runs are driven by the Terminal-Bench runner.

```bash
make terminal ARM=baseline MODEL=gpt-5.6-terra DATASET=terminal-bench@2.0 SEED=1 K=1 LIMIT=5
make terminal ARM=harness MODEL=gpt-5.6-terra DATASET=terminal-bench@2.0 SEED=1 K=1 LIMIT=5
```


| `DATASET` value             | Tasks | Use                                     |
| --------------------------- | ----- | --------------------------------------- |
| `terminal-bench@2.0`        | 89    | Full benchmark                          |
| `terminal-bench-sample@2.0` | 10    | Configured default; smoke/pilot dataset |


Omitting `DATASET` uses the 10-task sample. Select `DATASET=terminal-bench@2.0` for
the full benchmark. `LIMIT=5` runs the first 5 of the sample's 10 tasks; drop it to
run all 10.

### τ²-bench

Install the optional dependency with `uv sync --extra tau2` and set `TAU2_DATA_DIR`
(the data is not shipped). tau2 runs in the same Python 3.13 environment as PandaBench.

```bash
make tau2 ARM=baseline MODEL=gpt-5.6-terra DATASET=airline SEED=1 K=1 LIMIT=5
make tau2 ARM=harness MODEL=gpt-5.6-terra DATASET=airline SEED=1 K=1 LIMIT=5
```


| `DATASET` value | Tasks | Use                                         |
| --------------- | ----- | ------------------------------------------- |
| `airline`       | 50    | Configured default; official airline domain |
| `retail`        | 114   | Official tau2 retail domain                 |
| `telecom`       | 114   | Official tau2 telecom domain                |


The runner switches the tau2 task set, environment, policy, tools, and deterministic
evaluator together. Run the baseline/harness pair once for each table row to benchmark
all three official domains. `telecom-workflow` is an upstream policy-format ablation,
not a fourth leaderboard domain, and is intentionally excluded.

All three domains, paired at the same settings:

```bash
for dataset in airline retail telecom; do make tau2 ARM=baseline MODEL=gpt-5.6-terra DATASET=$dataset SEED=1 K=1 LIMIT=5; make tau2 ARM=harness MODEL=gpt-5.6-terra DATASET=$dataset SEED=1 K=1 LIMIT=5; done
```



## 4. Report — aggregate results into paper-ready tables

`make report` is pure post-processing (no API calls). It reads **every**
`results/runs/*/records.jsonl` already on disk and (re)writes `results/summary/`.
Run your benchmark commands first, then report — re-run it any time to refresh.

```bash
make report
```

Produces in `results/summary/`:

- `all_records.csv` — every task-trial row, flattened.
- `headline.csv` — benchmark × dataset × model × arm over the **whole run**, with three
metrics side by side: strict `pass@1`/`pass^k` (the **benchmark's own** verdict, the
only publishable one), `pass_at_1_relaxed`/`pass_hat_k_relaxed` and `mean_pass_ratio`
(**ours**), the `pass_tolerance` each arm was scored at, plus mean cost and tokens.
- `harness_telemetry.csv` — rules active/candidate/retired, notices, breach rate (arm B).
- `report.md` — headline table + harness-vs-baseline paired delta on **both** the strict
and the relaxed metric (bootstrap CI + McNemar p) + cost/overhead + methodology notes.
- `learning_curve.png` — arm-B cumulative pass rate across the run in task order. With
the harness live throughout, this is a genuine in-session learning curve.

Why a relaxed metric exists: in a measured 456-trial AppWorld run, 72% of trials failed
by exactly one test, so the strict all-or-nothing verdict was floored and could not
move. It applies to the **harness arm only** — the baseline is always scored by the
benchmark's own criteria (tolerance 0). The `relaxed` paired row is therefore an
intentionally asymmetric comparison (harness at tolerance N vs baseline at 0), which is
the intended read: how the harness does under a given tolerance against an unrelaxed
baseline. It is **not** comparable to published numbers, and the headline's
`pass_tolerance` column names the definition each arm used. Benchmarks with no
partial-credit signal (τ², Terminal-Bench) report it equal to the strict verdict.

With no records yet it writes an empty summary — that's expected before any run.

## 5. Calibrate — Checkpoint 1 (metric ↔ failure correlation)

`make calibrate BENCH=<name>` verifies that the PandaProbe metrics actually correlate
with *this benchmark's* task failures before you trust the harness arm. It finds the
**latest** `harness`**-arm run** for the benchmark, turns its records
into labels (`failed = not passed`), and runs the `pandaprobe-harness-calibrate` CLI
against that run's archived workspace, appending precision/recall/F1 to
`IMPLEMENTATION_NOTES.md`.

```bash
make calibrate BENCH=appworld
```

- **Prereqs:** a completed **real** `ARM=harness` run (with `PANDAPROBE_API_KEY`), so an
archived workspace + platform scores exist. Dry-run and baseline-only runs produce
nothing to calibrate.
- **When:** right after the first harness-arm run of a benchmark, and *before*
launching the full study. If the metrics don't separate pass/fail (low F1), the harness
arm would be inert — adjust the breach threshold in `study.yaml` (per the CLI's sweep)
and re-run, or record the null result and stop. See `IMPLEMENTATION_NOTES.md`.



## 6. Full study (all models × seeds × arms)

`make matrix` is **not yet wired to execute** — it prints guidance and exits, so it
never silently spends budget. Run the full study with an explicit loop over the per-arm
commands (edit the lists to your models/seeds/benchmarks):

```bash
for bench in appworld terminal tau2; do
  for model in claude-sonnet-5 gpt-5.6-terra gemini-3.1-pro; do
    for seed in 1 2 3; do
      for arm in baseline harness; do
        make $bench ARM=$arm MODEL=$model SEED=$seed K=4
      done
    done
  done
done
make report
```

Each run is **resumable** (rerun with the same `RUN_ID` to skip finished task-trials),
so a long study can be interrupted and continued. Budget deliberately: this is
`benchmarks × models × seeds × arms × K × tasks` LLM sessions — start with one
`(benchmark, model)` cell and `LIMIT` a few tasks to estimate cost before scaling up.

## Notes

- **Resumable:** rerun with the same `RUN_ID` to skip task-trials already recorded.
- **Dry-run anything:** append `--dry-run` to any `uv run pandabench-run …` (mock model,
no API calls) to validate wiring.
- **Everything is a plain CLI command** — the Makefile is sugar over
`uv run pandabench-run …` / `pandabench-report` / `pandabench-calibrate`.
