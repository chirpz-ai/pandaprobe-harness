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
cd ..
uv build --wheel                 # build the unreleased managed-repair candidate
cd benchmarks
cp .env.example .env          # fill in credentials
make setup                    # uv sync (including Harbor), isolated AppWorld env, preflight
uv run pandabench-run --preflight   # re-check tools + creds + a 1-token ping
```

`make setup` prints two env vars for AppWorld — add them to `.env`:

```bash
export PANDABENCH_APPWORLD_PYTHON=$HOME/.pandabench/awenv/bin/python
export APPWORLD_ROOT=$HOME/.pandabench/appworld
```

The benchmark's exact `pandaprobe-harness==0.8.0` requirement is temporarily
resolved from that wheel by `[tool.uv.sources]`, not from `../src` and not from
PyPI. Confirm with `uv pip show pandaprobe-harness`; the location metadata should
name `../dist/pandaprobe_harness-0.8.0-py3-none-any.whl`. Rebuild and sync after
any root-package change. Once the new architecture is released, delete the source
override and update the exact dependency pin.

Harness learning uses package-owned managed repair. By default PandaBench reuses
the resolved task model and explicitly sets `repair_reasoning_effort: "none"`,
which current OpenAI reasoning models require when using function tools through
the PandaProbe-wrapped LiteLLM chat-completions API. Choose another current
LiteLLM model deliberately when needed. The task agent sees learned guidance and
four read-only rule tools only.



## 2. Smoke test (pipeline check, no external harnesses)

```bash
make smoke        # 2 tasks/phase x 1 trial x both arms, all benchmarks, dry-run (mock model)
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
| `SEED`     | shuffles task order; run several (1, 2, 3) as replicates for statistics              | `1`                                               |
| `K`        | trials per task — `pass@1` = first trial passed, `pass^k` = all K passed             | `4`                                               |
| `DATASET`  | override the configured task universe (for example, Terminal-Bench's 10-task sample) | benchmark config                                  |
| `LIMIT`    | max **tasks per phase**; **omit to run the whole split**                             | unset (all)                                       |
| `MAXTURNS` | per-task **agent-turn cap** (how long the agent works on one task)                   | `study.yaml` `max_turns` (100 for all benchmarks) |
| `BACKEND`  | **Claude only**: `bedrock` or `anthropic`                                            | `bedrock`                                         |


- Omitting `DATASET` selects that benchmark's single configured default from
`configs/study.yaml`; it does **not** run every available dataset.
- `LIMIT` **controls the number of tasks.** It is applied independently after the
seeded learning/eval split: `LIMIT=5` with both phases runs up to 5 learning + 5 eval
tasks (10 total), and `K=4` makes that up to 40 trials per arm. On the raw CLI, use
the equivalent `--limit 5`.
- `LIMIT` **≠ task length.** To make each task run longer, raise `MAXTURNS`, e.g.
`MAXTURNS=60`, or bump `max_turns` in `configs/study.yaml` for that benchmark.
- For a paired A/B comparison, keep `MODEL`, `DATASET`, `SEED`, `K`, `LIMIT`, and
`MAXTURNS` identical; change only `ARM`.
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

The examples below are paired medium-pilot runs: 5 learning + 5 evaluation tasks,
once per arm.

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
the full benchmark. With the sample's 50/50 split, `LIMIT=5` runs all 5 learning and
all 5 evaluation tasks.

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
- `headline.csv` — benchmark × dataset × model × arm: `pass@1`, `pass^k`, mean cost,
tokens.
- `harness_telemetry.csv` — rules active/candidate/retired, notices, breach rate (arm B).
- `report.md` — headline table + harness-vs-baseline paired delta (bootstrap CI + McNemar
p) + cost/overhead + methodology notes.
- `learning_curve.png` — arm-B pass rate across the learning phase.

With no records yet it writes an empty summary — that's expected before any run.

## 5. Calibrate — Checkpoint 1 (metric ↔ failure correlation)

`make calibrate BENCH=<name>` verifies that the PandaProbe metrics actually correlate
with *this benchmark's* task failures before you trust the harness arm. It finds the
**latest** `harness`**-arm run** for the benchmark, turns its **learning-phase** records
into labels (`failed = not passed`), and runs the `pandaprobe-harness-calibrate` CLI
against that run's archived workspace, appending precision/recall/F1 to
`IMPLEMENTATION_NOTES.md`.

```bash
make calibrate BENCH=appworld
```

- **Prereqs:** a completed **real** `ARM=harness` run (with `PANDAPROBE_API_KEY`), so an
archived workspace + platform scores exist. Dry-run and baseline-only runs produce
nothing to calibrate.
- **When:** right after the first harness learning run of a benchmark, and *before*
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
