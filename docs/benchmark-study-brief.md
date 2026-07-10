# PandaProbe Harness — Benchmark A/B Study Implementation Brief

> **Audience:** a coding agent implementing this end-to-end. Read this document fully before
> writing any code. Everything you need is specified here or reachable from the links in §2.

## 0. Mission

Build a clean, reproducible benchmark suite inside this repository that measures the effect of the
**PandaProbe Harness** (a self-healing envelope for LLM agents) on agent reliability. Three public
benchmarks — **τ²-bench**, **Terminal-Bench 2.x (via Harbor)**, and **AppWorld** — each run in two
arms: an identical agent **without** the harness (baseline) and **with** the harness (treatment).
The output is a paper-ready set of results: per-task records, aggregate `pass@1` / `pass^k`
metrics, harness telemetry, and token/cost accounting, all runnable from the repo root via `make`.

Non-goals: do not modify the harness package itself (`src/`), do not invent new benchmarks, do not
build a web UI. Only if you find a harness bug, you're allowed to patch `src/` to fix it.

## 1. What PandaProbe and the Harness are

**PandaProbe** is an agent-engineering platform: an SDK traces agent sessions, and the platform
computes session-level evaluation metrics. The two metrics used here are `agent_reliability` and
`agent_consistency` — each scored 0–1 per session, where a score below the configured threshold
(default 0.5) is a **breach**.

**PandaProbe Harness** (`pandaprobe-harness` on PyPI, Python ≥3.13, zero runtime deps) wraps an
existing agent loop — any loop — and closes a self-heal cycle around it:

1. **Detect** — at each turn end, the harness (non-blocking, in the background) pulls the session's
   metric scores from the platform via the `pandaprobe` CLI. Degradations become
   `DiagnosticNotice`s posted to a filesystem **mailbox** under the harness workspace
   (`HARNESS_ROOT`, default `/harness` — always override this per run).
2. **Inform** — `harness.system_context(task_hint=...)` returns a system-prompt preamble containing
   the learned operating rules (task-conditioned top-k retrieval) plus pending notices. The agent
   injects this into its own prompt.
3. **Self-correct** — the agent gets a 14-op toolset (`harness.toolset`, exportable via
   `as_anthropic_tools()` / `as_openai_function_tools()` / `as_langchain_tools()`) to read notices,
   inspect scores, and write its own rules (`harness_rule_add`).
4. **Validate** — new rules enter as `candidate` and are promoted to `active` only on evidence:
   either by replaying a captured failure case (developer supplies a `ReplayFn`) or by a forward
   trial over live sessions. Regressing rules are retired. Breaching sessions are captured into a
   replayable **eval set** (`capture_eval_cases=True`), and `harness.run_regression()` re-runs the
   corpus against the current rule set.

Key facade API (all you need for this study):

```python
from pandaprobe_harness import Harness, HarnessConfig

harness = Harness.create(config, replay=my_replay_fn)   # config: HarnessConfig(root=..., ...)
preamble = harness.system_context(task_hint="...")       # inject into the agent's system prompt

async with harness.turn(session_id):                     # wrap each agent turn
    ...  # run one agent step
# or, for loops you can't wrap: harness.on_turn_end({"session_id": ..., "turn_index": ..., "end_state": {...}})

tools = harness.toolset.as_openai_function_tools()       # or as_anthropic_tools()
await harness.drain_validation()                         # deterministic: wait for rule validation
report = await harness.run_regression(sample=0)          # replay the eval set
```

The harness **requires** the PandaProbe platform: `PANDAPROBE_API_KEY` and
`PANDAPROBE_PROJECT_NAME` env vars, the `pandaprobe` CLI on PATH, and the agent's LLM calls traced
with the PandaProbe SDK **using the same session id** the harness sees. Config knobs are mirrored
as env vars: `HARNESS_ROOT`, `HARNESS_RULE_VALIDATION`, `HARNESS_RULE_RETRIEVAL`,
`HARNESS_CAPTURE_EVAL_CASES`, `HARNESS_REPLAY_TIMEOUT_S`, etc.

**Authoritative documentation — read these before integrating:**

- Docs site: `https://docs.pandaprobe.com/harness/get-started/quickstart` (and the whole Harness
  tab: concepts, how-it-works, closed-loop pages, integrations/custom-loops, reference/configuration).
- This repo: `src/pandaprobe_harness/harness.py` (facade), `agent_tools/toolset.py` (14 ops),
  `examples/` (offline and closed-loop examples), `README.md`.
- Do **not** import from `src/` in benchmark code — install the released package (§3).

## 2. Repository context

This repo (`chirpz-ai/pandaprobe-harness`) is the harness package itself:

```
pandaprobe-harness/
├── src/pandaprobe_harness/    # the package — DO NOT TOUCH
├── tests/                     # package tests (offline, 348 passing) — DO NOT TOUCH
├── examples/                  # package examples — DO NOT TOUCH
├── docs/                      # internal briefs (this file lives here)
├── pyproject.toml             # uv-managed; mypy --strict; ruff E,F,I,UP,B,ASYNC line 100
├── Makefile                   # make lint / typecheck / test / example
└── .github/workflows/         # ci.yml (gate), release.yml (PyPI on _version.py bump)
```

Invariants you must preserve: `make lint`, `make typecheck`, and `make test` at the repo root must
pass exactly as before your changes. Package CI must not start executing benchmark code.

## 3. Isolation requirements (hard constraints)

1. **All new code lives under a new top-level `benchmarks/` directory.** Nothing outside it may
   change except: (a) the root `Makefile` gains thin delegating targets, (b) the root
   `pyproject.toml` ruff config gains `extend-exclude = ["benchmarks"]` (so `ruff check .` keeps
   passing without linting the sub-project with the package's rules), (c) `.gitignore` gains
   benchmark artifacts (`benchmarks/results/**/raw/`, harness workspaces, caches), and (d)
   `README.md` may gain a one-paragraph pointer.
2. **`benchmarks/` is its own uv project** — own `pyproject.toml`, own `uv.lock`, own virtualenv.
   It must **not** be a workspace member of the root project. It depends on
   `pandaprobe-harness>=0.6` **from PyPI** — never a path/editable dependency on `../src`. Pin the
   exact version in the lock file.
3. Benchmark deps (`tau2`/τ²-bench, `appworld`, provider SDKs, `pandas`, `matplotlib`, …) exist
   only in `benchmarks/`. Harbor is a standalone tool: install with `uv tool install harbor`
   (document it; also add a `make setup` step).
4. `benchmarks/` has its own lint/typecheck config (ruff + mypy; strictness at your discretion but
   keep it clean) and its own `make check` — do not wire benchmark checks into package CI.

## 4. Target layout

```
benchmarks/
├── README.md                    # how to set up, run, and interpret everything
├── pyproject.toml               # uv project: pandaprobe-harness (PyPI) + benchmark deps
├── Makefile                     # all benchmark targets (root Makefile delegates here)
├── .env.example                 # every env var needed, documented
├── configs/
│   ├── models.yaml              # the model matrix: id, provider, per-benchmark overrides
│   ├── study.yaml               # arms, seeds, k, task subsets, phase definitions, cost caps
│   └── benchmarks/{tau2,terminal_bench,appworld}.yaml
├── src/pandabench/              # one installable package for shared code
│   ├── providers/               # unified LLM client layer (§7)
│   ├── agents/                  # the shared tool-calling agent loop + harness wrapper (§6)
│   ├── harness_glue.py          # Harness construction, session-id plumbing, ReplayFn per benchmark
│   ├── runners/                 # per-benchmark runners: tau2.py, terminal_bench.py, appworld.py
│   ├── results.py               # run manifest, per-task record schema, writers
│   ├── metrics.py               # pass@1, pass^k, bootstrap CIs, McNemar
│   └── report.py                # aggregation → CSV + markdown tables (+ optional plots)
├── adapters/
│   ├── tau2_agent.py            # τ²-bench custom-agent class (registered with their registry)
│   └── harbor_agent.py          # Harbor BaseAgent subclass (used via --agent-import-path)
├── scripts/                     # thin CLI entry points (also exposed as console scripts)
└── results/                     # gitignored raw outputs; committed: aggregated CSV/MD summaries
```

## 5. Study design

**Arms** (identical agent code, models, prompts, and task sets across arms — the only difference
is harness wiring):

| Arm | Name | Description |
|---|---|---|
| A | `baseline` | Agent loop with no harness. No preamble, no toolset, no PandaProbe tracing overhead beyond what's needed for cost accounting. |
| B | `harness` | Full harness: preamble injection, harness toolset added to the tool list, `turn()` wrapping, `capture_eval_cases=True`, replay wired, rule validation on. |
| B′ | `harness-noval` (optional, flag-gated) | Like B but `HARNESS_RULE_VALIDATION=false` — rules activate immediately. Ablation isolating the validation loop's contribution. Implement the flag; running it is optional. |

**Protocol per (benchmark × model):**

1. **Learning phase** — run the harness arm over a designated learning split (a fixed, seeded
   subset of tasks, ~30–40% of the selected task set) so the harness accumulates and validates
   rules. The baseline arm runs the same split (for symmetry of cost/exposure, and its results are
   kept but flagged `phase=learning`).
2. **Frozen eval phase** — run both arms over the held-out eval split. For arm B, the harness
   workspace carries over from the learning phase (rules learned there apply here); set
   `HARNESS_CAPTURE_EVAL_CASES=false` and do not add new rules during eval if the harness exposes
   a switch — otherwise document that learning continues and report both interpretations.
   **Do not start the frozen eval phase until both checkpoints below (§5.1) have passed or their
   failure has been explicitly recorded.**
3. **Trials** — every eval task runs `k` trials (default k=4; τ² supports its native `--num-trials`;
   Harbor has `-k`; AppWorld: loop yourself). `pass@1` = fraction of tasks whose first trial
   passed; `pass^k` = fraction whose *all k* trials passed. Compute both from the same per-trial
   records.
4. **Seeds / ordering** — ≥3 study seeds; each seed shuffles task order (counterbalancing order
   effects on the harness's learning). Note: current Claude models (4.6+) reject `temperature`, so
   trial-to-trial variance comes from the models' natural nondeterminism — do not try to force
   sampler seeds; record this in the methodology notes.
5. **Statistics** — paired per-task comparison between arms: McNemar's test on pass/fail pairs,
   plus bootstrap 95% CIs on the pass@1 / pass^k deltas. Implement in `metrics.py` with stdlib +
   numpy/scipy only. **Power caveat (state it in the report):** at ~30–40 eval tasks per
   benchmark, McNemar detects large deltas (~10+ points) but is underpowered for small ones even
   pooling seeds — frame results as directional with CIs, and scale subsets up only where early
   deltas look promising. Do not oversell small effects.

### 5.1 Required checkpoints (the study's load-bearing assumptions — do not skip)

The harness only helps if a long causal chain fires: task failure → PandaProbe metric breach →
notice → agent writes a rule → validation promotes it → the rule generalizes. Two links are
unverified going in, so gate the expensive runs on them:

- **Checkpoint 1 — metric↔failure calibration.** The PandaProbe metrics (`agent_reliability`,
  `agent_consistency`) were designed for production sessions; nobody has verified they correlate
  with *benchmark task failure*. After the first learning-phase run of each benchmark, run
  `pandaprobe-harness-calibrate --labels <path>` (the harness package ships this CLI) with labels
  derived from the benchmark's own pass/fail results (build a small
  `scripts/labels_from_records.py` that converts `records.jsonl` into the CLI's label format).
  Record precision/recall/F1 in `IMPLEMENTATION_NOTES.md` and in `summary/report.md`. If the
  metrics barely breach on failed tasks, adjust the breach threshold per the sweep output (record
  the chosen threshold in `study.yaml` — same threshold for all arms/seeds of that benchmark) and
  re-run the learning phase. If no threshold yields a usable signal, **stop and report** — arm B
  would be inert and the full matrix would waste the budget; that null mechanism is itself a
  finding and must be documented, not papered over.
- **Checkpoint 2 — rules actually promoted before eval.** With ~10–15 learning tasks,
  `rule_trial_min_sessions=5`, and replay validation needing captured cases, it is plausible the
  learning phase ends with zero *active* (promoted) rules — making arm B baseline-plus-overhead.
  After each learning phase, inspect the archived harness workspace (`rules.jsonl` / journal) and
  record `rules_candidate` / `rules_active` counts. If zero rules were promoted: first ensure the
  operational fix below is in place, then consider extending learning exposure (more trials per
  learning task — each trial is a distinct session and counts toward forward trials) or lowering
  `HARNESS_RULE_TRIAL_MIN_SESSIONS` for the study (record it in config). If it still ends at
  zero, run the frozen eval anyway but flag the run `learning_outcome=no_rules` in its manifest
  so the analysis can separate "harness learned nothing" from "harness rules didn't help".

**Operational fix required for arm B (implements the pacing both checkpoints depend on):**
benchmark trials run much faster than production sessions, so platform scoring can lag turn
ends — a session may finish before its scores exist, yielding no notices and no learning. In the
arm-B runner, between task-trials, call `await harness.refresh(session_id)` for the
just-finished session and then `await harness.drain_validation()` (both bounded/never-raising)
so notices land and candidate validation completes *during* the learning phase rather than after
it. Budget this into per-task wall time.

**Confound to report, not fix:** the harness preamble + 14 extra tools cost context and tokens
on every arm-B turn, which can depress arm B on long tasks independent of rule quality. The
cost/overhead table (§9) quantifies it; `summary/report.md` must discuss it explicitly in the
methodology notes.

**Metrics collected per trial** (the row schema, see §9): task id, benchmark, arm, model,
provider, seed, trial index, phase, pass/fail (benchmark-native), benchmark-native sub-metrics
(τ²: reward + pass components; AppWorld: TGC/SGC + no-collateral-damage; TB: resolved), wall time,
input/output tokens per provider, estimated cost, and — for harness arms — harness telemetry
(rules added/promoted/retired so far, notices posted this session, PandaProbe
reliability/consistency scores, breach flags).

**Sizing / budget discipline:** default task subsets, not full suites — τ²: one domain (retail)
full task set; Terminal-Bench: a stratified ~30-task subset; AppWorld: the `dev` split (or a
~40-task stratified subset of test-normal). All subsets fixed by seed and recorded in
`study.yaml`. Every runner takes `--limit N` and `--dry-run` (mock model, zero API calls). Add a
`make smoke` target: 2 tasks × 1 trial × both arms × 1 cheap model per benchmark, asserting the
full pipeline (run → records → report) end to end.

## 6. The agent under test

One shared, minimal **tool-calling loop** in `src/pandabench/agents/` used by all three
benchmarks (adapted per benchmark's tool surface), so the arms differ only in harness wiring:

- System prompt = benchmark-required prompt (+ harness preamble in arm B, refreshed each turn via
  `system_context(task_hint=<task instruction>)`).
- Tools = benchmark's tools (+ harness toolset in arm B, namespaced `harness_*` so they can't
  collide).
- Loop: call model → execute tool calls → repeat until final answer / max turns. Per-benchmark max
  turns from config.
- Arm B wraps each iteration in `async with harness.turn(session_id)` and passes
  `end_state={"task_id": ..., "messages": <replayable snapshot>}` so eval-case capture works, and
  the LLM calls are traced with the PandaProbe SDK under that same session id (one session per
  task-trial; session id format: `{benchmark}-{task_id}-{arm}-{model}-{seed}-t{trial}` sanitized).

**ReplayFn per benchmark** (`harness_glue.py`): given an `EvalCase` (whose `replay_input` carries
the task id) and a system context string, re-run that benchmark task once with the provided
context and return the NEW session id. Keep replays cheap: cap max turns, and honor
`HARNESS_REPLAY_TIMEOUT_S`.

## 7. LLM provider layer (hard constraints)

**Use LiteLLM (`litellm`) as the single, core LLM provider layer for the entire study.** Every
model call in every benchmark, both arms, the τ² user simulator, and every ReplayFn goes through
one thin wrapper in `src/pandabench/providers/` built on `litellm.acompletion(...)`. Do not call
provider SDKs directly anywhere else — one code path means uniform tool-calling semantics, usage
accounting, retries, and logging across all providers, and switching a model or backend is a
config-string change, not a code change.

**Routing (LiteLLM model strings — these encode the provider, so the study's backend constraints
are enforced entirely in `configs/models.yaml`):**

1. **Vertex AI is the primary platform** (the user's credits live there).
   - **Gemini via Vertex:** `vertex_ai/gemini-2.5-pro`, `vertex_ai/gemini-2.5-flash` (verify
     current IDs at implementation time). Auth: ADC (`gcloud auth application-default login`) +
     `VERTEXAI_PROJECT` / `VERTEXAI_LOCATION` env vars (or per-call `vertex_project` /
     `vertex_location` kwargs — pick one mechanism and use it everywhere).
2. **OpenAI models via OpenAI's own API** (Vertex does not serve them): `openai/gpt-...` with
   `OPENAI_API_KEY`. Suggested: one frontier model (e.g. `gpt-5.1`) and one mini-tier — verify
   current IDs at implementation time.
3. **Claude models must support BOTH backends, switchable per run** (credits exist on both) —
   with LiteLLM this is purely a prefix swap on the same bare model id:
   - Anthropic API: `anthropic/claude-sonnet-4-6`, `anthropic/claude-opus-4-8`
     (`ANTHROPIC_API_KEY`).
   - Vertex AI: `vertex_ai/claude-sonnet-4-6`, `vertex_ai/claude-opus-4-8` (same ADC +
     project/location as Gemini; verify the Vertex partner-model form against the installed
     LiteLLM version's docs).
   - Selection: each Claude entry in `models.yaml` declares `backends: [anthropic, vertex_ai]`
     with a default; overridable via `CLAUDE_BACKEND` env var and a `BACKEND=` make variable. The
     resolved full LiteLLM string is recorded in every run manifest and record row.
   - Suggested models: `claude-sonnet-4-6` (workhorse), `claude-opus-4-8` (frontier). Bare ids —
     no date suffixes.

**Wrapper requirements** (the only place LiteLLM is touched directly):

- One async chat-with-tools interface: OpenAI-format messages + tool schemas in; normalized
  assistant message, parsed tool calls, and usage out. LiteLLM normalizes tool calling to the
  OpenAI shape across providers — the shared agent loop consumes only that shape.
- **Usage & cost:** read `response.usage` on every call and accumulate per session/trial; compute
  cost via `litellm.completion_cost(response)`, with the `models.yaml` price table as fallback
  when LiteLLM lacks a price for a model. Record both the configured model key and the resolved
  LiteLLM model string in every record.
- **Retries/timeouts:** set `num_retries` and a per-request `timeout` on each call (values from
  config), plus a loop-level cap; on exhausted retries, record the trial as `error`, never crash
  the run. Do not stand up a LiteLLM proxy server — use the SDK in-process; `litellm.Router` is
  optional and only if fallback routing proves necessary (document if used).
- **Provider quirks still apply through LiteLLM** — it passes params through, it doesn't sanitize
  them: for Claude 4.6+ do not send `temperature`/`top_p`/`top_k` (400 on Opus 4.8); keep
  per-model param allowlists in `models.yaml` (or use `litellm.drop_params` deliberately and
  document it — don't rely on it silently). Parse tool-call arguments as JSON, never string-match.
- **Determinism of routing:** pin the `litellm` version in `uv.lock`; disable any implicit
  fallbacks — a run must fail loudly rather than silently answer from a different provider.
- Keep the wrapper mockable: unit tests patch the single wrapper entry point, not `litellm`
  internals.

One env template documents everything (`benchmarks/.env.example`): `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, `CLAUDE_BACKEND`, plus PandaProbe
vars (§10).

## 8. Benchmark integrations

### 8.1 τ²-bench (`sierra-research/tau2-bench`)

- **Why:** the pass^k reliability benchmark — multi-turn customer-service agent + LLM-simulated
  user + domain policy. Its metrics are the study's headline. Use τ²-bench, not the original
  `tau-bench` (whose README marks its tasks outdated).
- **Integration path (important):** τ²'s orchestrator owns the conversation loop and drives an
  agent object against the environment + user simulator. Implement a **custom agent class** in
  `adapters/tau2_agent.py` conforming to τ²'s agent interface (subclass its agent base /
  registry-register it — verify the exact interface from the installed package's docs and source;
  do not fight the orchestrator or reimplement the user simulator). Your agent class delegates to
  the shared pandabench loop's policy: given history + tools, produce the next assistant
  message/tool call via **our** LiteLLM wrapper (§7). τ²'s built-in agents also use LiteLLM
  internally, but do not reuse their call path — route through the pandabench wrapper so usage
  accounting, retries, harness wiring, and model routing stay identical across all three
  benchmarks.
- **Harness wiring:** one τ² simulation = one harness session. Inject `system_context()` into the
  agent's system prompt (τ² composes domain policy + your prompt — append, never replace policy);
  wrap each policy invocation in `harness.turn(...)`; expose harness tools alongside domain tools.
  The user simulator runs on a fixed cheap model (e.g. `gemini-2.5-flash` or `gpt-*-mini`) —
  identical across arms, recorded in config.
- **Runs:** domain `retail` (add `airline`/`telecom` only if budget allows), native trials support
  for k, both arms, all study seeds.
- **Scoring:** τ²'s native task-success reward is ground truth; record its full per-task result
  object into `raw/`.

### 8.2 Terminal-Bench 2.x via Harbor

- **Why:** infrastructural fit — sandboxed terminal tasks, first-class custom agents, repeated
  trials built in; the public leaderboard shows a median ~13.6-point within-model spread across
  harnesses, so harness effects are measurable here.
- **Integration path:** `uv tool install harbor` (Docker required). Implement
  `adapters/harbor_agent.py` subclassing Harbor's `BaseAgent`, run via
  `harbor run -d terminal-bench/terminal-bench-2 --agent-import-path pandabench.adapters.harbor_agent:PandaBenchAgent -m <model> -k <k> ...`
  (verify exact flags against the installed Harbor version). The agent receives the task
  environment; inside it, run the shared loop with a bash/terminal tool surface.
- **Harness wiring:** one TB task attempt = one harness session. Arm selection via env var passed
  through Harbor into the agent (document how Harbor propagates env/config to agents; if it
  doesn't cleanly, select the arm via a small config file path in an env var). `HARNESS_ROOT` must
  point at a host-persistent path shared across attempts within a (model × arm × seed) run so
  learning accumulates.
- **Scoring:** Harbor's per-attempt records give resolved/unresolved; compute pass@1 and pass^k
  from per-attempt data yourself (don't rely on its aggregate only). Copy Harbor's run artifacts
  into our `results/<run_id>/raw/`.

### 8.3 AppWorld (`StonyBrookNLP/appworld`)

- **Why:** richest failure-detection metrics — TGC/SGC plus database-state unit tests including
  **no-collateral-damage** checks, i.e. it detects *harmful* wrong actions, not just missing ones.
- **Integration path:** pure library — you own the loop. `pip install appworld` (into the
  benchmarks venv), `appworld install`, `appworld download data` in `make setup`. Runner:
  `from appworld import AppWorld, load_task_ids`; for each task,
  `with AppWorld(task_id=..., experiment_name=<run_id>) as world:` drive the shared loop where the
  agent's tool is `world.execute(code)` (its API-calling environment), then `world.evaluate()` for
  TGC/SGC/test results. Verify the exact evaluation API against the installed version (v0.1.3+).
- **Harness wiring:** one task-trial = one harness session; straightforward since we own the loop.
- **Runs:** `dev` split (or stratified subset), k trials via your own loop, both arms, all seeds.

## 9. Results collection (paper-ready)

**Directory layout:**

```
benchmarks/results/
├── runs/<run_id>/               # run_id = {benchmark}_{model}_{arm}_{seed}_{YYYYMMDD-HHMMSS}
│   ├── manifest.json            # full resolved config, package versions (uv.lock hash,
│   │                            #   pandaprobe-harness version), git SHA, env fingerprint
│   ├── records.jsonl            # one row per task-trial (schema below)
│   ├── harness/                 # arm B: the entire HARNESS_ROOT workspace archived at run end
│   │   └── ...                  #   (rules.jsonl, journal, evalset, mailbox — the telemetry gold)
│   └── raw/                     # benchmark-native outputs (τ² results, Harbor artifacts, AppWorld evals)
└── summary/                     # committed to git
    ├── all_records.csv          # flattened union of every run's records
    ├── headline.csv             # benchmark × model × arm: pass@1, pass^k, CIs, p-values, cost
    ├── harness_telemetry.csv    # rules promoted/retired, notices, breach rates per run
    └── report.md                # human-readable tables + methodology notes
```

**`records.jsonl` row schema** (define as a typed dataclass in `results.py`; version the schema
with a `schema_version` field):

```json
{"schema_version": 1, "run_id": "...", "benchmark": "tau2", "task_id": "...", "arm": "harness",
 "model": "claude-sonnet-4-6", "provider": "vertex", "seed": 1, "trial": 0, "phase": "eval",
 "passed": true, "native_metrics": {...}, "turns": 14, "wall_time_s": 212.4,
 "usage": {"input_tokens": ..., "output_tokens": ..., "cost_usd": ...},
 "harness": {"session_id": "...", "reliability": 0.82, "consistency": 0.74, "breached": false,
             "rules_active": 3, "rules_candidate": 1, "notices": 0} | null,
 "error": null}
```

**Reporting:** `make report` regenerates everything in `summary/` from `runs/`. Tables must be
directly usable in a paper: one headline table (benchmark × model: baseline vs harness pass@1 and
pass^k with deltas + CIs + McNemar p), one learning-curve table/plot for arm B (pass rate vs task
index within the learning phase), one telemetry table, one cost/overhead table (token overhead of
the harness preamble+toolset vs baseline). Runs are **resumable**: a runner started with an
existing `run_id` skips task-trials already present in `records.jsonl`.

## 10. PandaProbe platform wiring

- Env: `PANDAPROBE_API_KEY`, and set `PANDAPROBE_PROJECT_NAME` per benchmark+arm (e.g.
  `bench-tau2-harness`) so platform data stays organized; put the naming convention in config.
- The `pandaprobe` CLI must be on PATH (`make setup` checks and prints install instructions:
  `curl -fsSL https://cli.pandaprobe.com/install.sh | sh`).
- Fresh `HARNESS_ROOT` per (benchmark × model × arm × seed) run under
  `results/runs/<run_id>/harness_root/`, archived into `harness/` at run end.
- Trace agent LLM calls with the PandaProbe SDK under the harness's session id (see the harness
  quickstart docs for the exact SDK session API). In arm A, tracing may be disabled entirely —
  but then compute cost from provider usage (we do anyway), and note the asymmetry; prefer
  tracing both arms identically if overhead is negligible.

## 11. Makefile / ergonomics (hard constraints)

Root `Makefile` delegates: `make bench-setup`, `make bench-smoke`, `make bench-run ...`,
`make bench-report` → `$(MAKE) -C benchmarks <target>`. Inside `benchmarks/Makefile`:

```
make setup                        # uv sync, harbor tool install, appworld data, CLI checks
make smoke                        # §5 smoke test, < ~10 min, cheap models
make tau2      ARM=harness MODEL=claude-sonnet-4-6 SEED=1 [BACKEND=vertex] [K=4] [LIMIT=]
make terminal  ARM=baseline MODEL=gemini-2.5-pro  SEED=1 ...
make appworld  ...
make matrix                       # the full study matrix from configs/study.yaml (prints a cost
                                  #   estimate and asks for confirmation before launching)
make report                       # regenerate summary/ from runs/
make check                        # lint + typecheck + unit tests for benchmarks/ code
```

Everything a target does must be reproducible by a plain documented CLI command
(`uv run pandabench-run --benchmark tau2 --arm harness ...`) — the Makefile is sugar, not logic.

## 12. Engineering standards

- Extremely clean and organized: typed Python, small modules, docstrings on every public function,
  no dead code, no notebook-style scripts. Config over code: nothing study-relevant hardcoded.
- Unit tests for the pure logic (metrics math, record schema round-trip, report aggregation,
  provider request shaping and model-string resolution with the LiteLLM wrapper mocked). No
  network in tests.
- Never commit secrets; `.env` loading via a tiny explicit loader or `python-dotenv`;
  `.env.example` documents every variable with a comment.
- Fail loud and early: `make setup` validates Docker, CLI, credentials (a 1-token ping per
  configured provider) before any expensive run.
- Log per-run to `results/runs/<run_id>/run.log`; console output is a compact progress line per
  task-trial.
- `benchmarks/README.md` is the single onboarding doc: prerequisites, setup, the exact command
  sequence to reproduce the full study, and how to read the outputs.

## 13. Deliverables & acceptance criteria

1. `benchmarks/` project as specified, with root-Makefile delegation; root `make lint typecheck
   test` still green and package CI untouched.
2. All three integrations working: for each benchmark, `make smoke` completes end-to-end in both
   arms with a cheap model — real benchmark harness, real model calls, real records written, and
   for arm B a non-empty archived harness workspace.
3. Provider layer: Gemini-on-Vertex, OpenAI-API, and Claude on **both** Anthropic API and Vertex
   verified by the setup ping + one smoke trial each.
4. `make report` produces the four summary artifacts from smoke data.
5. Resumability demonstrated (kill a run mid-way, rerun, verify skip-ahead).
6. Checkpoint tooling (§5.1) implemented and exercised in smoke: `scripts/labels_from_records.py`
   feeding `pandaprobe-harness-calibrate`, a `make calibrate BENCH=...` target, workspace
   rule-count inspection in the report, and the arm-B between-task `refresh` +
   `drain_validation` pacing.
7. `benchmarks/README.md` complete; every config file commented.
8. A short `IMPLEMENTATION_NOTES.md` in `benchmarks/` recording: exact versions pinned, any
   deviations from this brief and why, checkpoint results (§5.1) as they come in, and known sharp
   edges (e.g. how Harbor env propagation was solved).

## 14. Suggested build order

1. Scaffolding: uv project, configs, provider layer + setup pings, record schema, Makefiles.
2. AppWorld runner (you own the loop — easiest), baseline arm → harness arm → smoke.
3. Harbor agent adapter, both arms → smoke.
4. τ²-bench agent adapter, both arms → smoke.
5. Metrics + report pipeline; full smoke matrix; resumability; docs; `IMPLEMENTATION_NOTES.md`.

Work phase by phase; keep `make check` green at each step. Where this brief says "verify against
the installed version", actually read the installed package's source/docs before coding against it
— benchmark APIs move fast and this brief is not a substitute for their documentation.
