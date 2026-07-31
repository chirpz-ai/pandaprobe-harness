# Running PandaBench

Run everything from `benchmarks/`. Each benchmark runs two
arms — `baseline` (no harness) and `harness` — over the same tasks/models.

## 0. Prerequisites

- **uv** and **Python 3.13**.
- **`pandaprobe` CLI** on PATH: `curl -fsSL https://cli.pandaprobe.com/install.sh | sh`
- **Credentials** (put in `benchmarks/.env`; see `.env.example`):
  - Vertex AI: `gcloud auth application-default login` + `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`
  - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (as needed)
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

## 2. Smoke test (pipeline check, no external harnesses)

```bash
make smoke        # 2 tasks/phase x 1 trial x both arms, all benchmarks, dry-run (mock model)
make report       # regenerate results/summary/
```

## 3. Run a benchmark

Model keys: `gemini-3.1-flash-lite`, `gemini-3.5-flash`, `gemini-3.1-pro`,
`gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`, `claude-sonnet-5`, `claude-haiku-4-5`.

**Knobs:**

| Knob | Meaning | Default |
|---|---|---|
| `ARM` | `baseline` (no harness) or `harness` | `baseline` |
| `MODEL` | a model key from the list above | `gemini-3.1-flash-lite` |
| `SEED` | shuffles task order; run several (1, 2, 3) as replicates for statistics | `1` |
| `K` | trials per task — `pass@1` = first trial passed, `pass^k` = all K passed | `4` |
| `DATASET` | override the configured task universe (for example, Terminal-Bench's 10-task sample) | benchmark config |
| `LIMIT` | max **tasks per phase**; **omit to run the whole split** | unset (all) |
| `MAXTURNS` | per-task **agent-turn cap** (how long the agent works on one task) | `study.yaml` `max_turns` (100 for all benchmarks) |
| `BACKEND` | **Claude only**: `vertex_ai` or `anthropic` | model's `default_backend` |

- **`LIMIT` controls the number of tasks.** It is applied independently after the
  seeded learning/eval split: `LIMIT=5` with both phases runs up to 5 learning + 5 eval
  tasks (10 total), and `K=4` makes that up to 40 trials per arm. On the raw CLI, use
  the equivalent `--limit 5`.
- **`LIMIT` ≠ task length.** To make each task run longer, raise `MAXTURNS`, e.g.
  `MAXTURNS=60`, or bump `max_turns` in `configs/study.yaml` for that benchmark.
- For a paired A/B comparison, keep `MODEL`, `DATASET`, `SEED`, `K`, `LIMIT`, and
  `MAXTURNS` identical; change only `ARM`.
- **OpenAI / Gemini route automatically** by their `models.yaml` prefix
  (`openai/…` → OpenAI API via `OPENAI_API_KEY`; `vertex_ai/…` → Vertex). Only set
  `BACKEND` for Claude — passing it to an OpenAI/Gemini model errors.

The examples below are paired medium-pilot runs: 5 learning + 5 evaluation tasks,
once per arm.

### AppWorld

Needs the isolated env from `make setup` (`PANDABENCH_APPWORLD_PYTHON`, `APPWORLD_ROOT`).

```bash
make appworld ARM=baseline MODEL=gpt-5.6-terra DATASET=dev \
  SEED=1 K=1 LIMIT=5
make appworld ARM=harness MODEL=gpt-5.6-terra DATASET=dev \
  SEED=1 K=1 LIMIT=5
```

### Terminal-Bench (Harbor)

Needs **Docker running**. Harbor is installed into this project so its CLI can import
the custom `pandabench` agent. Real runs are driven by the Terminal-Bench runner.

```bash
make terminal ARM=baseline MODEL=gpt-5.6-terra \
  DATASET=terminal-bench-sample@2.0 SEED=1 K=1 LIMIT=5
make terminal ARM=harness MODEL=gpt-5.6-terra \
  DATASET=terminal-bench-sample@2.0 SEED=1 K=1 LIMIT=5
```

The sample dataset has 10 tasks. With its 50/50 split, `LIMIT=5` runs all 5 learning
and all 5 evaluation tasks. Omit `DATASET` to use the configured 89-task
`terminal-bench@2.0` dataset.

### τ²-bench (retail)

Install the optional dependency with `uv sync --extra tau2` and set `TAU2_DATA_DIR`
(the data is not shipped). tau2 runs in the same Python 3.13 environment as PandaBench.

```bash
make tau2 ARM=baseline MODEL=gpt-5.6-terra DATASET=retail \
  SEED=1 K=1 LIMIT=5
make tau2 ARM=harness MODEL=gpt-5.6-terra DATASET=retail \
  SEED=1 K=1 LIMIT=5
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
- `headline.csv` — benchmark × model × arm: `pass@1`, `pass^k`, mean cost, tokens.
- `harness_telemetry.csv` — rules active/candidate/retired, notices, breach rate (arm B).
- `report.md` — headline table + harness-vs-baseline paired delta (bootstrap CI + McNemar
  p) + cost/overhead + methodology notes.
- `learning_curve.png` — arm-B pass rate across the learning phase.

With no records yet it writes an empty summary — that's expected before any run.

## 5. Calibrate — Checkpoint 1 (metric ↔ failure correlation)

`make calibrate BENCH=<name>` verifies that the PandaProbe metrics actually correlate
with *this benchmark's* task failures before you trust the harness arm. It finds the
**latest `harness`-arm run** for the benchmark, turns its **learning-phase** records
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
