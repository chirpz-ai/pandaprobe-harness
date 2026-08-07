# Implementation notes

Engineering record for the PandaBench suite: pinned versions, benchmark-specific
deviations, checkpoint results as they arrive, and sharp edges discovered while
building. Read alongside `RUNNING.md`.

## Pinned versions (benchmarks/uv.lock)

- `pandaprobe-harness==0.8.0` remains exact while the pre-release candidate is
  resolved from `../dist/pandaprobe_harness-0.8.0-py3-none-any.whl` by
  `[tool.uv.sources]`. This is a non-editable built artifact. Remove the source
  override and update the pin after the managed-repair release is published.
- `harbor==0.18.0` — installed in this project so its custom-agent import can resolve
  `pandabench`.
- `pandaprobe>=0.5` (the SDK; native LiteLLM wrapper + session binding).
- `litellm` (>=1.55; lock pins the resolved version, currently 1.91.x).
- `pandas>=2.2`, `numpy>=2.0`, `scipy>=1.14`, `matplotlib>=3.9`, `tabulate`, `httpx`,
  `pyyaml`, `python-dotenv`.
- Python 3.13 (matches the harness). AppWorld alone uses an isolated environment;
  tau2 and Harbor run in this project's environment.

## Integration choices and deviations

1. **`adapters/` lives under `src/pandabench/adapters/`, not top-level.** Harbor's
   `-a pandabench.adapters.harbor_agent:PandaBenchAgent` requires `adapters` to be an
   importable subpackage of `pandabench`; a top-level directory would not import.
   `scripts/` stays top-level (thin CLI shims over package modules).

2. **AppWorld runs OUT-OF-PROCESS over HTTP; pandabench never imports it.**
   AppWorld (`0.1.3.post1`) pins `pydantic<2`, which is irreconcilable with modern
   LiteLLM (`pydantic>=2.10`) — uv cannot co-resolve them. So AppWorld is not a
   dependency of `pandabench`; instead it runs as its own *environment server*
   (`appworld serve environment`) in an isolated pydantic-v1 venv, and we drive it
   via the REST API AppWorld exposes for exactly this reason. `AppWorldServer` +
   `HttpAppWorldEnv` in `runners/appworld_env.py`. This keeps the pandabench core
   (LiteLLM, pydantic v2) conflict-free. **Verified end-to-end against the real
   server** (see Verification status).

3. **PandaProbe tracing uses the native LiteLLM wrapper (SDK >= 0.5).**
   `providers/tracing.py` calls `wrap_litellm(litellm)` once (when a client is
   available) to auto-trace every `litellm.acompletion` call, and binds each call
   to the harness session via `pandaprobe.session(session_id)`. The tracer guards on
   `get_client()` — arm A and offline tests (`PandaTracer.disabled()`) never patch
   LiteLLM or open a session. (Earlier builds used manual `start_trace`+span
   instrumentation because SDK 0.4 had no LiteLLM wrapper; 0.5 added it.)

4. **The managed-repair arm settles every completed learning turn.**
   `HarnessWiring.settle_turn`
   flushes traces, supplies a replayable `end_state`, calls `on_turn_end`, and waits
   for task evaluation plus package-owned managed repair before the next prompt. The
   task agent receives only `harness_rules_read/search/list/status`; mailbox,
   diagnostic, acknowledgement, and rule-mutation capabilities remain private to the
   package repair agent. Its preamble is capability-only: neither live nor frozen
   wiring inserts rules or an expanded index, and neither performs an automatic
   rule lookup. `harness_rules_list` returns the canonical task-facing `harness_guide.md`
   guide plus generated scope references. The runner performs an idempotent final-turn settle and the
   learning boundary drains outstanding validation. tau2
   crosses its synchronous worker-thread boundary with `run_coroutine_threadsafe`;
   Terminal-Bench performs the same learning barrier inside Harbor's custom agent.

5. **Terminal-Bench arm B has two explicit capability deviations.** It uses the
   learning per-turn barrier and a post-learning job settle, but candidate rules receive only
   forward-trial validation: replay would require another container build and full
   task run per candidate. It also has no live outcome verifier because Harbor's
   authoritative reward is produced after `agent.run()` returns. These are structural
   limitations, not parity with AppWorld/tau2.

6. **Harness eval is genuinely frozen.** After the bounded learning settlement,
   PandaBench snapshots every active, provisional candidate, and retired rule into
   `frozen-rules.json`. Canonical JSON plus deterministic rule ordering produces the
   recorded SHA-256. AppWorld and tau2 use `FrozenEvalWiring` directly; Harbor receives
   an explicit `phase=eval`, `frozen_eval=true`, and absolute snapshot path. Frozen
   wiring exposes only list/search/read/status operations, reports no pending notices,
   uses the same capability-only preamble, and never constructs a `Harness` or
   enters settlement. Native `/evaluate`, tau2 reward evaluation, and Harbor
   verification remain unchanged.

7. **The benchmark Tier-1 stall window is 10.** `study.yaml` propagates this through
   `build_harness_config`; it delays only learning-phase STALL detection. The locally
   built root candidate retains its package default of 5.

8. **Managed repair uses the root package's PandaProbe LiteLLM path.** The benchmark
   does not implement a repair loop. Null reuses the resolved task model, and the
   study explicitly sets `repair_reasoning_effort: "none"` so current OpenAI models
   accept function tools through the PandaProbe-wrapped LiteLLM chat-completions
   contract. The manifest records the effective model and limits, and live telemetry
   stores the structured `RepairResult`, including episode/group, scope, novelty,
   and resolution telemetry. Repair tracing remains under the distinct
   `repair-<task-session>-<episode>` identity owned by the package.

9. **Scope hints come from benchmark metadata, not classification.** AppWorld
   matches safe application names already exposed by its task/API descriptions;
   tau2 supplies the selected airline/retail/telecom domain and optional workflow
   metadata; Terminal-Bench passes safe Harbor category/task-family/workflow
   metadata when available. These hints travel on `TurnContext` into notices and
   repair assignments. A precise hint wins over `appworld`, `tau2`, or
   `terminal_bench`; `global` remains an explicit cross-domain applicability.

10. **Metrics/report/checkpoints were built alongside the AppWorld slice**, not in a
   separate later pass, because the vertical slice's acceptance gate is
   run → records → report end-to-end.

11. **Smoke (`make smoke`) runs in `--dry-run`** (mock model, mock benchmark envs) as
   the deterministic pipeline gate. Real per-benchmark smokes are separate targets
   that need each harness provisioned + live creds (see below). This matches the
   brief's `--dry-run` requirement and gives a dependency-free acceptance check.

## Sharp edges discovered

- **AppWorld CLI `--root` overrides `$APPWORLD_ROOT`** (default `.`). The server must
  be launched with `--root <isolated-root>` or it can't find `./data`. Handled in
  `AppWorldServer.start`.
- **AppWorld `/evaluate` needs `suppress_errors: true`** or it 500s on an incomplete
  task. Handled in `HttpAppWorldEnv.evaluate`.
- **AppWorld port default mismatch** (CLI `serve` defaults differ from `run()`); we
  always pass `--port` explicitly.
- **AppWorld holds one active world per server** — tasks run serially per server;
  concurrency would need multiple ports.
- **Claude is fixed to `max_tokens` only; GPT-5 rejects several sampler params** —
  `models.yaml` per-model `param_allowlist` drops them; we filter explicitly
  (not via `litellm.drop_params`).
- **`tau2` PyPI name is ambiguous** — install the Sierra benchmark from the pinned
  Git tag through PandaBench's `tau2` extra, never by an unqualified PyPI name.

## Setup for real runs

- **AppWorld**: `make setup` provisions an isolated venv + data via
  `scripts/setup_appworld.sh` and prints the two env vars to export
  (`PANDABENCH_APPWORLD_PYTHON`, `APPWORLD_ROOT`). ~183 MB data download.
- **Terminal-Bench**: needs Docker running; `uv sync` installs Harbor into this project.
- **tau2**: run `uv sync --extra tau2` and set `TAU2_DATA_DIR=<clone>/data`.
- **Providers**: export `VERTEXAI_PROJECT`/ADC and/or `OPENAI_API_KEY`. Claude
  defaults to AWS Bedrock: set a short-term `AWS_BEARER_TOKEN_BEDROCK` plus the
  matching `AWS_REGION`; set `ANTHROPIC_API_KEY` only for the optional
  `BACKEND=anthropic` fallback. Also set `PANDAPROBE_API_KEY` (+
  `PANDAPROBE_PROJECT_NAME`) for harness runs — or put everything in
  `benchmarks/.env`. `uv run pandabench-run --preflight` validates them.
- **Bedrock on-demand Claude calls require inference profiles.** The catalog's
  base `anthropic.*` IDs reject on-demand throughput for these models, so the
  registry calls AWS's corresponding `global.anthropic.*` system inference
  profiles. They retain the exact underlying model versions and use global/base
  pricing; use geography-specific profiles only if data residency is required.

## Benchmark integration recipes

All three integrations use verified installed APIs. AppWorld is fully wired and
real-verified; the verification status below distinguishes offline coverage from
paid live-model smokes for tau2 and Terminal-Bench.

- **Terminal-Bench via Harbor** — PyPI `harbor==0.18.0` provides the CLI (NOT
  `terminal-bench`). Custom agent = subclass `harbor.agents.base.BaseAgent` (impl
  `name`/`version`/`setup`/`run`); the sandbox is driven by
  `await environment.exec(cmd) -> ExecResult{stdout,stderr,return_code}`. The agent
  runs in-process on the host, so it reuses our loop + harness directly (bash tool =
  `environment.exec`). Config reaches it via `--agent-kwarg k=v` (JSON-typed →
  `__init__`) and `--agent-env K=V` (→ auto-injected into `exec`). Run:
  `harbor run -d terminal-bench@2.0 -a pandabench.adapters.harbor_agent:PandaBenchAgent
  -m <model> -k <k> -n 1 -o <dir> --ak arm=... --ak phase=... --ak frozen_eval=...
  --ak harness_root=...`. Frozen eval also receives
  `--ak frozen_rules_path=<absolute-path>`. Per-attempt
  result at `<dir>/<job>/<task>__<id>/result.json` →
  `verifier_result.rewards: dict[str, float | int]`; TB2 normally uses
  `{"reward": 0|1}`, with first-value fallback for compatible verifiers. GATES:
  Docker running; Harbor and pandabench are co-installed by this project. The runner
  resolves the `harbor` entry point beside `sys.executable`, not an arbitrary executable
  on `PATH`: a `uv tool install harbor` environment cannot import this custom agent.

- **tau2-bench** — PyPI `tau2` is a DECOY (a magnetics package). Real install:
  `git+https://github.com/sierra-research/tau2-bench.git@v0.2.0` (→ `tau2==0.2.1.dev0`,
  import `tau2`, Python >=3.10). Data is NOT shipped: set `TAU2_DATA_DIR=<clone>/data`
  *before the first tau2 import* — tau2 reads it at import time. Its only relevant
  constraint is `litellm>=1.65.0` with no upper bound, so it installs into THIS venv
  as the `tau2` extra (`uv sync --extra tau2`); no isolated interpreter. Custom agent = subclass `tau2.agent.llm_agent.LLMAgent`,
  override `generate_next_message` to route through our wrapper (done). tau2's
  `run_task` hardcodes the `LLMAgent(tools, domain_policy, llm, llm_args)` constructor,
  so to attach the harness wiring we drive `tau2.orchestrator.Orchestrator` per
  (task×trial),
  keeping the user simulator on tau2's stock `generate()` (fixed model, arm-independent).
  Reward: `Orchestrator.run()` does NOT grade (it returns `reward_info=None`) — call
  `tau2.evaluator.evaluator.evaluate_simulation(..., evaluation_type=EvaluationType.ALL)`
  separately. Use `ALL`, never `ALL_WITH_NL_ASSERTIONS` (that one calls an LLM);
  all three official domains grade deterministically: airline/retail use
  `[DB, COMMUNICATE]`, while telecom uses `[ENV_ASSERTION]` with some `[ACTION,
  ENV_ASSERTION]` tasks. Grading is therefore free and doubles as the harness's gold
  outcome signal. `DATASET=airline|retail|telecom` switches the task set, environment,
  policy, tools, and evaluator as one unit. `passed` = `is_successful(reward)` (== 1.0
  within 1e-6, not a threshold). `Orchestrator.run()` is blocking, so the runner drives
  it in a worker thread and the agent submits chat plus the learning per-turn barrier
  back to the runner's loop. The package-owned repair agent consumes any notice during
  settlement through its own PandaProbe LiteLLM wrapper and distinct repair session;
  tau2's adapter contains no mailbox or rule-authoring model loop. The next tau2 turn
  receives a capability-only `Harness.system_context` plus the four read-only tools.
  Live and frozen agents choose whether to list/search/read the immutable live rules;
  nothing embeds them in the domain prompt. Administrative calls are rejected at
  dispatch and cannot contaminate the domain transcript or grading.
  GATES: `uv sync --extra tau2` + `TAU2_DATA_DIR` + live creds (incl. Vertex ADC for
  the user simulator).

## Verification status (this build)

- **Current managed-repair migration**: the benchmark installs the local root wheel,
  exposes only read-only task tools, delegates repair to the package, records structured
  repair outcomes, and preserves frozen eval. Current gate results and live K=1/LIMIT=1
  run identifiers are recorded in the implementation report for this change.
- **tau2 paid smoke** (`tau2_gpt-5.6-terra_harness_1_20260730-202115`): four real
  retail episodes completed without integration errors (2 passed, $0.2402 recorded
  agent cost). Every session has a trace series longer than one (9–30 samples), the
  journal contains Tier-1 regressions plus Tier-2 breach signatures, and records carry
  real rewards, 8–21 turns, and non-null trace scores. Replay validation promoted two
  rules and retired eight with trace-metric reasons. This small sample learned only
  scoped rules, so `rules/global.md` was not created; the strict-pull index correctly
  references the populated scoped file.
- **Terminal-Bench paid smoke**
  (`terminal_bench_gpt-5.6-terra_harness_1_20260730-204311`): Harbor 0.18.0 ran four
  serial Docker trials without exceptions (eval 1/2 passed; $1.0258 agent cost).
  `result.json` supplied `{"reward": ...}` dictionaries, turns ranged 5–15, and all
  four sessions have 2–14 trace samples. The journal shows `trend`/`stall` escalation
  and Tier-2 breaches; both global and scoped provisional rule files are populated and
  indexed from `harness_guide.md`. The two-task-per-phase smoke did not supply the three
  subsequent live sessions required for a forward-only candidate to promote or retire,
  which is the documented Terminal-Bench validation deviation.
- **AppWorld real integration**: `AppWorldServer` + `HttpAppWorldEnv` + `AppWorldRunner`
  driven against the **live AppWorld environment server** (57 dev tasks; initialize /
  api_docs / execute real code / evaluate 1-of-2 tests / close; runner `run_once` with
  a scripted tool call). ✅ verified. A full live-model smoke additionally needs LLM
  creds exported in the shell.
- **Arm-B capture, providers, and metrics**: capture/replayable eval cases, provider
  routing and usage, pass@1/pass^k, McNemar, paired deltas, and bootstrap are covered by
  the offline suite.

## Checkpoint results

- **Checkpoint 1 (metric↔failure calibration)** still requires a labelled real arm-B
  learning run; the tooling is built (`pandabench-calibrate`,
  `scripts/labels_from_records.py`).
- **Checkpoint 2 (rule promotion)** engaged in the tau2 smoke: two replay-validated
  rules promoted and eight regressing candidates retired. Terminal-Bench created four
  forward-validation candidates but, as expected for a two-task-per-phase smoke, did
  not observe enough later live sessions to decide them.
