# PandaBench

An A/B benchmark study measuring the **PandaProbe Harness** effect on agent
reliability across **AppWorld**, **Terminal-Bench 2.x (via Harbor)**, and
**τ²-bench**. Each benchmark runs in two arms with identical agent code, model,
prompts, and task sets — the only difference is harness wiring:

| Arm | Description |
|---|---|
| `baseline` | Plain tool-calling loop; no harness. |
| `harness` | The harness is **live for the benchmark's entire dataset**: every turn is evaluated, notices go to the package-owned managed repair agent, and candidate rules are validated — from the first task to the last. |

Self-contained uv project that installs `pandaprobe-harness==0.9.0` from PyPI — not
an editable import of `../src`. A study run therefore measures the same artifact a
user installs. Bump the pin and the lockfile together when moving to a new release.

## Harness live throughout

There is **no learning/eval split and no frozen ruleset**. The claim under test is
*in-session healing* — a rule learned at task N helping task N+1 of the same run — so
a run is one continuous pass over the whole configured dataset. Each benchmark's own
dataset is used as given (AppWorld's native `dev` / `test_normal` / … are never
re-partitioned, which keeps our numbers comparable to published results for that
split).

Because the harness is live throughout, **task order is load-bearing**: it decides
what has been learned by the time task N runs. Order is a pure function of
`(dataset, seed)` and is **identical in both arms**, without which the paired
per-task comparison would be invalid. Vary `--seed` to counterbalance which tasks the
harness sees early.

## What the agent sees

PandaBench attaches only a short stable capability preamble and four read-only tools:
`harness_rules_list`, `harness_rules_read`, `harness_rules_search`, and
`harness_rule_status`. It does not put rule bodies or the expanded index in the
prompt, and performs no rule lookup before the task model chooses one.
`harness_rules_list` returns the canonical task-facing `rules.md`: SKILL-style
frontmatter and a stable pull workflow, followed by generated scope references.

After a turn, related notices form one repair episode; managed repair may add at most
one provisional candidate, and the next turn can discover it on demand. Repair also
chooses each rule's scope from the failure evidence, as part of the call it already
makes — no extra model round, and no benchmark-specific application mapping in the
root package. Benchmarks feed that decision deterministic context, never a naming
rule: AppWorld passes the task instruction plus application names from safe API
metadata, tau2 its domain plus the user scenario, Terminal-Bench the Harbor task
statement plus category metadata. A benchmark's own label for itself is rejected as a
scope.

Validation, never repair, promotes or retires a candidate. Replay is the strong path
and sees only the candidate under test; a bounded per-round replay budget falls back
to a cheap forward trial so every candidate reaches a verdict rather than sitting
undecided. Validation runs *during* the pass, which is what allows a rule to be
promoted mid-run; a single bounded settlement at the **end of the run** then drains
outstanding evals and in-flight validation before the workspace is archived.

## Metrics

`TrialRecord.passed` is always the **benchmark's own** verdict (AppWorld: every test
passes; τ²: `is_successful(reward)`; Terminal-Bench: `reward >= 1.0`), applied
identically in both arms, and the only figure comparable to published numbers.

That verdict is floored in practice — in a measured 672-trial AppWorld `test_normal`
run, 68% of trials failed by exactly one test, and in 97% of those the single failing
test was `assert answers match`. Records therefore also keep each benchmark's raw
signal: per-test counts, and `native_metrics.failing_tests` for *which* AppWorld tests
failed (bounded, `requirement` text only, never the traceback). If most tasks always
miss the same test, that is either one systematic agent behavior a rule could fix or an
environment artifact, and the counts alone cannot tell them apart.

Everything beyond the native verdict is derived in the report rather than written into
a record, so additional views cost nothing and require no re-run.

## Prerequisites

- Python 3.13 + [uv](https://docs.astral.sh/uv/).
- The `pandaprobe` CLI on PATH (`curl -fsSL https://cli.pandaprobe.com/install.sh | sh`).
- LLM credentials for the providers you use (see `.env.example`): Vertex AI ADC
  (`gcloud auth application-default login` + `VERTEXAI_PROJECT`), `OPENAI_API_KEY`, or
  `AWS_PROFILE_NAME` + `AWS_REGION` for the default Claude backend (any AWS
  profile; boto3 auto-refreshes it, including SSO).
  `ANTHROPIC_API_KEY` enables the optional Claude fallback; the harness arm also needs
  `PANDAPROBE_API_KEY`.
- Docker running (Terminal-Bench only). Harbor is installed with this project.
- AppWorld isolated env (`make setup` provisions it; ~183 MB data).

## Setup

```bash
cp .env.example .env      # fill in credentials
make setup                # uv sync (including Harbor), isolated AppWorld env, preflight
uv run pandabench-run --preflight   # validate tools + creds + a 1-token ping
```

`make setup` prints two env vars to export for real AppWorld runs
(`PANDABENCH_APPWORLD_PYTHON`, `APPWORLD_ROOT`) — add them to `.env`.

The benchmark reuses the run's resolved task model for managed repair unless
`harness.repair_model` selects another. `repair_reasoning_effort: "none"` keeps current
OpenAI reasoning models on the tool-capable chat-completions path wrapped by
PandaProbe. Repair is reported separately in harness telemetry.

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
(`uv run pandabench-run --benchmark appworld --arm harness ...`). From the repo root
the same targets are `make bench-setup`, `make bench-smoke`, `make bench-report`,
`make bench-check`.

Runs are **resumable**: rerun with the same `--run-id` (or `make ... RUN_ID=...`) and
task-trials already in `records.jsonl` are skipped.

## Outputs

```
results/runs/<run_id>/
  manifest.json     # resolved config, versions, git SHA, env fingerprint, rules_outcome
  records.jsonl     # one row per task-trial (schema in results.py::TrialRecord)
  harness/          # harness arm: archived HARNESS_ROOT (index, scopes, JSONL, journal, mailbox)
  raw/              # benchmark-native artifacts
results/summary/    # regenerated by `make report`
  all_records.csv   headline.csv   harness_telemetry.csv   report.md   learning_curve.png
```

`manifest.json`'s `rules_outcome` is the end-of-run rule lifecycle stamp (`active=N`,
`active=N,pending=M`, `pending=M`, or `no_rules`) — it distinguishes "produced nothing"
from "produced candidates validation never decided".

`headline.csv` is the benchmark × dataset × model × arm view over the **whole run**,
strictest metric first, plus cost and tokens. `report.md` adds the harness-vs-baseline
paired deltas (bootstrap CI + McNemar p), harness telemetry, cost/overhead, and the
methodology caveats — statistical power, nondeterminism, preamble token overhead, and
which metrics are publishable.

## Layout

```
src/pandabench/
  providers/   models.py (registry) · litellm_client.py (the one LLM path) · tracing.py
  agents/      loop.py (shared loop) · harness_wiring.py (live) · frozen_wiring.py (unwired)
  runners/     base.py · appworld.py + appworld_env.py · terminal_bench.py · tau2.py · mock.py
  adapters/    harbor_agent.py · tau2_agent.py
  frozen_rules.py  harness_glue.py  results.py  metrics.py  report.py  checkpoints.py  cli.py
configs/       models.yaml · study.yaml · benchmarks/*.yaml
scripts/       labels_from_records.py · setup_appworld.sh
```

`frozen_rules.py` and `agents/frozen_wiring.py` are **currently unwired**: nothing
constructs them now that the harness is live throughout. They are retained with their
tests because transfer/generalization ("does a static learned ruleset help on unseen
tasks?") is a separate question we may want to measure later.
