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
make setup                    # uv sync, harbor tool, isolated AppWorld env, preflight
uv run pandabench-run --preflight   # re-check tools + creds + a 1-token ping
```

`make setup` prints two env vars for AppWorld — add them to `.env`:

```bash
export PANDABENCH_APPWORLD_PYTHON=$HOME/.pandabench/awenv/bin/python
export APPWORLD_ROOT=$HOME/.pandabench/appworld
```

## 2. Smoke test (pipeline check, no external harnesses)

```bash
make smoke        # 2 tasks x 1 trial x both arms, all benchmarks, dry-run (mock model)
make report       # regenerate results/summary/
```

## 3. Run a benchmark

Model keys: `gemini-3.1-flash-lite`, `gemini-3.5-flash`, `gemini-3.1-pro`,
`gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.6-luna`, `claude-sonnet-5`, `claude-haiku-4-5`.

**Knobs:**

| Knob | Meaning | Default |
|---|---|---|
| `ARM` | `baseline` (no harness) or `harness` | `baseline` |
| `MODEL` | a model key from the list above | `gemini-3.1-flash-lite` |
| `SEED` | shuffles task order; run several (1, 2, 3) as replicates for statistics | `1` |
| `K` | trials per task — `pass@1` = first trial passed, `pass^k` = all K passed | `4` |
| `LIMIT` | max **tasks per phase**; **omit to run the whole split** | unset (all) |
| `MAXTURNS` | per-task **agent-turn cap** (how long the agent works on one task) | `study.yaml` `max_turns` (100 for all benchmarks) |
| `BACKEND` | **Claude only**: `vertex_ai` or `anthropic` | model's `default_backend` |

- **`LIMIT` ≠ task length.** Unset `LIMIT` runs every task in the split (e.g. AppWorld
  `dev` ≈ 20 learning / 37 eval). To make each task run *longer* (what makes the harness
  matter), raise `MAXTURNS`, e.g. `MAXTURNS=60`, or bump `max_turns` in
  `configs/study.yaml` for that benchmark.
- **OpenAI / Gemini route automatically** by their `models.yaml` prefix
  (`openai/…` → OpenAI API via `OPENAI_API_KEY`; `vertex_ai/…` → Vertex). Only set
  `BACKEND` for Claude — passing it to an OpenAI/Gemini model errors.

### AppWorld

Needs the isolated env from `make setup` (`PANDABENCH_APPWORLD_PYTHON`, `APPWORLD_ROOT`).

```bash
make appworld ARM=baseline MODEL=gemini-3.1-flash-lite SEED=1 K=4 LIMIT=5
make appworld ARM=harness  MODEL=claude-sonnet-5 SEED=1 BACKEND=vertex_ai K=4 LIMIT=5
```

### Terminal-Bench (Harbor)

Needs **Docker running** + `harbor` installed in an env that also has `pandabench`
(see IMPLEMENTATION_NOTES.md). Real runs are Harbor-driven (see the runner's message
for the exact `harbor run …` command).

```bash
make terminal ARM=baseline MODEL=gemini-3.1-pro SEED=1 K=4
```

### τ²-bench (retail)

Needs its own isolated venv (`git+…/tau2-bench.git@v0.2.0` + `pandabench`) and
`TAU2_DATA_DIR` (data is not shipped). See IMPLEMENTATION_NOTES.md for the recipe.

```bash
make tau2 ARM=harness MODEL=gpt-5.4-mini SEED=1 K=4
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
  and re-run, or record the null result and stop. See the brief §5.1.

## 6. Full study (all models × seeds × arms)

`make matrix` is **not yet wired to execute** — it prints guidance and exits, so it
never silently spends budget. Run the full study with an explicit loop over the per-arm
commands (edit the lists to your models/seeds/benchmarks):

```bash
for bench in appworld terminal tau2; do
  for model in claude-sonnet-5 gpt-5.4-mini gemini-3.1-pro; do
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
