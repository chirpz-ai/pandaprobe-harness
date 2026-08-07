# PandaBench

An A/B benchmark study measuring the **PandaProbe Harness** effect on agent
reliability across **AppWorld**, **Terminal-Bench 2.x (via Harbor)**, and
**τ²-bench**. Each benchmark runs in two arms with identical agent code, model,
prompts, and task sets — the only difference is harness wiring:

| Arm | Description |
|---|---|
| `baseline` | Plain tool-calling loop; no harness. |
| `harness` | Learning evaluates the developer-owned task agent and uses the package-owned managed repair agent. Eval uses one hashed, read-only snapshot of the learning rules and keeps native benchmark grading active without trace evaluation or repair. |

The benchmark override is `gate_window: 10`: a Tier-1 STALL needs ten
consecutive non-improving evaluated trace updates during learning. REGRESSION
behavior is unchanged. At the phase boundary, PandaBench drains the bounded
learning settlement, writes `frozen-rules.json`, and reuses that exact hash for
every harness-arm eval trial. Eval tracing may remain enabled for later
inspection, but no PandaProbe trace listing, scoring, notices, rule mutation,
validation, replay, or settle barrier runs.

In both learning and frozen eval, PandaBench attaches only a short stable
capability preamble and four read-only tools: `harness_rules_list`,
`harness_rules_read`, `harness_rules_search`, and `harness_rule_status`. It does
not put rule bodies or the expanded index in the prompt and performs no rule
lookup before the task model chooses one. After a learning turn, related notices
form one repair episode; managed repair may add at most one provisional
candidate, and the next turn can discover it on demand.
`harness_rules_list` returns the canonical task-facing `harness_guide.md`: SKILL-style
frontmatter and a stable pull workflow followed by generated scope references.

Hosts supply deterministic semantic scopes without another model call:
AppWorld derives application names from its safe API/task metadata; tau2 supplies
its domain plus available workflow metadata; Terminal-Bench uses Harbor category
or task-family metadata when present. The generic root package contains no
benchmark-specific application mapping.

Self-contained uv project currently locked to the locally built root wheel via
`[tool.uv.sources]`. This validates the distributable managed-repair candidate
without an editable import or a premature release. After publication, remove the
source override and update the exact version pin. See `RUNNING.md` for the workflow.

## Prerequisites

- Python 3.13 + [uv](https://docs.astral.sh/uv/).
- The `pandaprobe` CLI on PATH (`curl -fsSL https://cli.pandaprobe.com/install.sh | sh`).
- LLM credentials for the providers you use (see `.env.example`): Vertex AI ADC
  (`gcloud auth application-default login` + `VERTEXAI_PROJECT`), `OPENAI_API_KEY`,
  or `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION` for the default Claude backend.
  `ANTHROPIC_API_KEY` enables the optional Claude fallback; the harness arm also
  needs `PANDAPROBE_API_KEY`.
- Docker running (Terminal-Bench only). Harbor is installed with this project.
- AppWorld isolated env (`make setup` provisions it; ~183 MB data).
- A current root wheel at `../dist/pandaprobe_harness-0.8.0-py3-none-any.whl`.

## Setup

```bash
cd .. && uv build --wheel && cd benchmarks
cp .env.example .env      # fill in credentials
make setup                # uv sync (including Harbor), isolated AppWorld env, preflight
uv run pandabench-run --preflight   # validate tools + creds + a 1-token ping
```

`make setup` prints two env vars to export for real AppWorld runs
(`PANDABENCH_APPWORLD_PYTHON`, `APPWORLD_ROOT`) — add them to `.env`.

During learning, the benchmark reuses the run's resolved task model for managed
repair unless `harness.repair_model` selects another current model.
`repair_reasoning_effort: "none"` keeps current OpenAI reasoning models on the
tool-capable chat-completions path wrapped by PandaProbe. Repair is reported
separately in harness telemetry.

## Running

See **[RUNNING.md](RUNNING.md)** for the full command-first, step-by-step guide.

```bash
make smoke                # dry-run pipeline gate: both arms x tiny task set, all benchmarks
make report               # regenerate results/summary/ from results/runs/

# One real arm of one benchmark (needs that harness provisioned + creds):
make appworld ARM=harness  MODEL=claude-sonnet-5 SEED=1 BACKEND=bedrock K=4 LIMIT=5
make terminal ARM=baseline MODEL=gemini-3.1-pro  SEED=1
make tau2     ARM=harness   MODEL=gpt-5.6-terra    SEED=1

make calibrate BENCH=appworld   # Checkpoint 1: metric<->failure calibration
make matrix                     # full study matrix (left for the operator; spends budget)
make check                      # ruff + mypy (strict) + offline unit tests
```

Every target is sugar over a documented CLI command
(`uv run pandabench-run --benchmark appworld --arm harness ...`). From the repo
root the same targets are `make bench-setup`, `make bench-smoke`,
`make bench-report`, `make bench-check`.

Runs are **resumable**: rerun with the same `--run-id` (or `make ... RUN_ID=...`)
and task-trials already in `records.jsonl` are skipped.

## Outputs

```
results/runs/<run_id>/
  manifest.json     # resolved config, versions, git SHA, env fingerprint, learning_outcome
  records.jsonl     # one row per task-trial (schema in results.py::TrialRecord)
  frozen-rules.json # arm B: immutable learning-boundary rules + stable SHA-256
  harness/          # arm B: archived HARNESS_ROOT (index, scopes, JSONL, journal, mailbox)
  raw/              # benchmark-native artifacts
results/summary/    # committed, regenerated by `make report`
  all_records.csv   headline.csv   harness_telemetry.csv   report.md   learning_curve.png
```

`headline.csv` is the benchmark × model × arm view (pass@1, pass^k, cost);
`report.md` adds the harness-vs-baseline paired delta (bootstrap CI + McNemar p),
telemetry, cost/overhead, and the methodology caveats (statistical power,
nondeterminism, preamble token overhead).

## Layout

```
src/pandabench/
  providers/   models.py (registry) · litellm_client.py (the one LLM path) · tracing.py
  agents/      loop.py (shared loop) · harness_wiring.py (live learning) · frozen_wiring.py
  runners/     base.py · appworld.py + appworld_env.py · terminal_bench.py · tau2.py · mock.py
  adapters/    harbor_agent.py · tau2_agent.py
  frozen_rules.py  harness_glue.py  results.py  metrics.py  report.py  checkpoints.py  cli.py
configs/       models.yaml · study.yaml · benchmarks/*.yaml
scripts/       labels_from_records.py · setup_appworld.sh
```
