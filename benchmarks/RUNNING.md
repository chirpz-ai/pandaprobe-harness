# Running PandaBench

Command-first guide. Run everything from `benchmarks/`. Each benchmark runs two
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

Knobs: `ARM=baseline|harness` · `MODEL=<models.yaml key>` · `SEED=<int>` ·
`BACKEND=vertex_ai|anthropic` (Claude only) · `K=<trials>` · `LIMIT=<max tasks/phase>`.
Model keys: `gemini-3.1-flash-lite`, `gemini-3.5-flash`, `gemini-3.1-pro`,
`gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.6-luna`, `claude-sonnet-5`, `claude-haiku-4-5`.

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

## 4. Report & checkpoints

```bash
make report                   # results/summary/{all_records.csv, headline.csv,
                              #   harness_telemetry.csv, report.md, learning_curve.png}
make calibrate BENCH=appworld # Checkpoint 1: metric<->failure calibration (harness arm)
```

## 5. Full matrix

```bash
make matrix                   # the full study from configs/study.yaml — spends real
                              # budget; left for the operator to launch deliberately
```

## Notes

- **Resumable:** rerun with the same `RUN_ID` to skip task-trials already recorded.
- **Dry-run anything:** append `--dry-run` to any `uv run pandabench-run …` (mock model,
  no API calls) to validate wiring.
- **Everything is a plain CLI command** — the Makefile is sugar over
  `uv run pandabench-run …` / `pandabench-report` / `pandabench-calibrate`.
