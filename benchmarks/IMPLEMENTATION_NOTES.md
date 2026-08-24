# Implementation notes

Engineering record for PandaBench: pinned versions, integration deviations, and
sharp edges found while building. Read alongside `RUNNING.md`.

## Pinned versions (benchmarks/uv.lock)

- `pandaprobe-harness==0.9.0`, from PyPI. Exact-pinned so a recorded run names the
  precise harness it measured, non-editable so a study cannot silently measure
  uncommitted local changes. Bump the pin and lockfile together.
- `harbor==0.18.0` — installed here so its custom-agent import can resolve `pandabench`.
- `pandaprobe>=0.5` (SDK: native LiteLLM wrapper + session binding), `litellm>=1.55`.
- `pandas`, `numpy`, `scipy`, `matplotlib`, `tabulate`, `httpx`, `pyyaml`, `python-dotenv`.
- Python 3.13. AppWorld alone needs an isolated environment; tau2 and Harbor run in
  this project's.

## Integration choices

1. **`adapters/` lives under `src/pandabench/adapters/`.** Harbor's
   `-a pandabench.adapters.harbor_agent:PandaBenchAgent` requires an importable
   subpackage; a top-level directory would not import.

2. **AppWorld runs out-of-process over HTTP; pandabench never imports it.** AppWorld
   pins `pydantic<2`, irreconcilable with modern LiteLLM (`pydantic>=2.10`) — uv cannot
   co-resolve them. It runs as its own environment server in an isolated venv, driven
   through the REST API it exposes for this purpose (`runners/appworld_env.py`).

3. **Tracing uses the native LiteLLM wrapper.** `providers/tracing.py` calls
   `wrap_litellm(litellm)` once and binds each call to the harness session. It guards on
   `get_client()`, so the baseline arm and offline tests never patch LiteLLM.

4. **The harness arm settles every completed turn.** `HarnessWiring.settle_turn` flushes
   traces, supplies a replayable `end_state`, calls `on_turn_end`, and waits for
   evaluation plus managed repair before the next prompt. The task agent gets only the
   four read-only rule tools; mailbox, diagnostic, and rule-mutation capabilities stay
   private to the package repair agent. tau2 crosses its synchronous worker-thread
   boundary with `run_coroutine_threadsafe`; Terminal-Bench barriers inside Harbor's
   custom agent.

5. **Terminal-Bench has two capability deviations.** Candidate rules get only
   forward-trial validation (replay would need another container build and full task run
   per candidate), and there is no live outcome verifier because Harbor's authoritative
   reward arrives after `agent.run()` returns. Structural limitations, not parity with
   AppWorld/tau2.

6. **The harness is live for the whole dataset; there is no frozen eval.** A run is one
   continuous pass with notices, repair, and validation active throughout, because the
   claim under test is in-session healing. The earlier design split each task set into a
   live learning phase and a frozen eval phase and reported only the latter, which
   measured a different and harder hypothesis. On a real 228-trial AppWorld run the
   paired eval delta was `-0.0037` (p=0.75) while the learning delta was `+0.0393`, and
   8 of 9 surviving rules were scoped to `spotify`/`venmo`, learned from 20 tasks and
   applied to 37 disjoint ones. τ² and Terminal-Bench have no native splits, so the
   partition was purely our imposition; AppWorld does ship splits, but we were
   re-partitioning *one of them*, making our numbers uncomparable to published results.
   Native splits are now used whole. `frozen_rules.py`, `agents/frozen_wiring.py`, and
   `results.frozen_harness_telemetry` are retained but **unwired**.

7. **`capture` is passed explicitly, not derived from a phase name.**
   `build_harness_config(capture=...)` gates `capture_eval_cases`, the sole switch on
   whether a breaching session is stored as a replayable failure case. It used to be
   inferred as `phase == "learning"`; with one phase named `"live"` that silently
   evaluates to `False`, which would have disabled capture for every trial and made
   replay-based promotion impossible while leaving every test green.

8. **Task order is load-bearing.** It decides what has been learned by task N, so it is
   a pure function of `(dataset, seed)` and `_tasks()` deliberately takes no `arm`
   argument — the two arms cannot diverge by construction.

9. **Replay budgets are per-benchmark, in each benchmark's own turn unit.** `max_turns`
   means whatever a runner passes it to, and tau2 forwards it as `max_steps` (message
   hops, not agent turns). One global value in agent-turn units truncated tau2 replays to
   roughly a quarter of a median episode, so 46% of validation cases came back
   `candidate_not_exercised` and every rule fell through to the weaker forward trial.
   `StudyConfig.replay_max_turns(benchmark)` resolves the override.

10. **The benchmark Tier-1 stall window is 10**, propagated from `study.yaml` through
   `build_harness_config`; it only delays STALL detection. The released package keeps its
   own default of 5, and a test pins that the study override does not leak into it.

11. **Managed repair uses the root package's PandaProbe LiteLLM path.** The benchmark
   implements no repair loop. Null reuses the resolved task model; the study sets
   `repair_reasoning_effort: "none"` so current OpenAI models accept function tools on
   the wrapped chat-completions contract. `models.yaml` can declaratively override
   compatibility settings while a task model repairs itself: both gpt-oss models omit
   that reasoning value because Bedrock Harmony rejects `none`; they, Qwen3 Coder, and
   Nemotron get 12 turns because six ended mid-protocol in live pipeline runs. A
   dedicated study repair model intentionally ignores task-model overrides. Effective
   settings are stamped in each manifest. Repair tracing keeps a distinct
   `repair-<task-session>-<episode>` identity owned by the package.

12. **Scope is a repair-model decision; benchmarks only supply context.** Repair picks
   the scope from the failure evidence inside the call it already makes — no
   classification model, no benchmark application mapping in the root package. AppWorld
   passes the task instruction plus safe application names; tau2 its domain plus the
   user scenario; Terminal-Bench the Harbor task statement plus safe category metadata.
   These travel on `TurnContext` as `rule_scope_hints` and `task_summary`. A benchmark's
   own name is rejected as a scope, since it says where the agent ran rather than what
   failed.

13. **Validation reaches every candidate, and the run's end waits for it.** Replay is the
   strong path and sees only the candidate under test, so a delta is attributable to it;
   a replay that never read the candidate yields no conclusive verdict.
   `validation_round_budget_s` bounds replay work per round, with remaining candidates
   falling back to the forward trial so a promotable candidate is actually promoted
   instead of accruing observations forever. `replay_env_wait_timeout_s` plus
   `ReplayRuleWiring.mark_environment_ready()` keep time queueing behind AppWorld's
   single-world lock out of the replay execution budget. Validation is spawned by the
   harness from each handled report — detached and single-flight — so it runs *during* the
   pass and a rule can be promoted mid-run. `BenchmarkRunner._settle` then runs **once at
   the end**, the only point where nothing holds an environment a replay needs.

14. **`passed` is always the benchmark's own verdict**, applied identically in both arms,
   and the only figure comparable to published numbers. It is floored in practice: in a
   measured 672-trial AppWorld `test_normal` run, 68% of trials failed by exactly one
   test, and in 97% of those the one failing test was `assert answers match` — the agent
   performs the state mutations and returns the wrong answer string. Records therefore
   also persist the raw per-test counts and `native_metrics.failing_tests` (bounded
   `requirement` texts, ≤12 entries and ≤240 chars each, never the traceback, which is a
   full stack dump with source context). Whether the same test always fails is the
   difference between one systematic agent behavior a rule could fix and an environment
   artifact, and the counts alone cannot say. Everything derived from those counts is
   computed in the report, not written into a record.

15. **Smoke (`make smoke`) runs in `--dry-run`** (mock model, mock benchmark envs) as the
   deterministic pipeline gate. Real per-benchmark smokes are separate targets needing
   each harness provisioned plus live creds.

## Sharp edges

- **AppWorld CLI `--root` overrides `$APPWORLD_ROOT`** (default `.`). The server must be
  launched with `--root <isolated-root>` or it can't find `./data`.
- **AppWorld `/evaluate` needs `suppress_errors: true`** or it 500s on an incomplete task.
- **AppWorld port defaults differ** between CLI `serve` and `run()`; always pass `--port`.
- **AppWorld holds one active world per server** — tasks run serially; concurrency needs
  multiple ports.
- **Claude and Bedrock open weights are fixed to `max_tokens` only; GPT-5 rejects
  several sampler params** — `models.yaml` per-model `param_allowlist` drops them
  explicitly, not via `litellm.drop_params`. LiteLLM 1.91.1 reports no
  `supports_temperature` metadata for any of the nine open weights, so unsupported
  sampling controls are not guessed. Their smallest output ceiling is Llama 4 Scout's
  4k, which matches the normal 4,096-token fallback. Live runs found that gpt-oss-20b
  and Qwen3 Next could spend all 4,096 tokens on internal reasoning before emitting a
  structured tool call, so their registry entries set an 8,192-token task/replay
  fallback (Qwen3 Next's full output ceiling). Caller-supplied budgets still win.
- **`tau2` on PyPI is a decoy** (a magnetics package). Install the Sierra benchmark from
  the pinned Git tag through the `tau2` extra, never by unqualified PyPI name.
- **`TAU2_DATA_DIR` must be set before the first tau2 import** — tau2 reads it at import
  time.

## Setup for real runs

- **AppWorld**: `make setup` provisions an isolated venv + data and prints the two env
  vars to export (`PANDABENCH_APPWORLD_PYTHON`, `APPWORLD_ROOT`). ~183 MB download.
- **Terminal-Bench**: needs Docker running; `uv sync` installs Harbor here.
- **tau2**: `uv sync --extra tau2` and `TAU2_DATA_DIR=<clone>/data`.
- **Providers**: `VERTEXAI_PROJECT`/ADC and/or `OPENAI_API_KEY`. AWS Bedrock hosts both
  the default Claude route and all nine open-weight models (`AWS_PROFILE_NAME` +
  `AWS_REGION`); `ANTHROPIC_API_KEY` is only for Claude's optional
  `BACKEND=anthropic` fallback. Harness runs also need `PANDAPROBE_API_KEY` and
  `PANDAPROBE_PROJECT_NAME`. `uv run pandabench-run --preflight` validates them.
- **Some Bedrock calls require inference profiles.** Claude's base `anthropic.*` IDs
  reject on-demand throughput, so the registry calls the corresponding
  `global.anthropic.*` profiles. Llama 4 Scout is `INFERENCE_PROFILE`-only and must use
  `us.meta.llama4-scout-17b-instruct-v1:0`; its bare ID also rejects on-demand calls.
- **gpt-oss is Bedrock-only in this registry.** OpenAI does not serve the open weights
  on its first-party API and LiteLLM has no `openai/gpt-oss-*` entry. Adding a
  third-party OpenAI-compatible host later would be a separate backend, credential,
  and pricing decision.
- **Tool support is a hard integration gate.** Mixtral 8x7B and Gemma 3 27B reject the
  `tools` parameter. Magistral Small 2509 is more dangerous: it accepts `tools` but
  emits `[TOOL_CALLS]...` as ordinary text, so a run silently records zero structured
  calls and misleading near-total failure. None are registered. Supporting Magistral
  would require text-format parsing, contrary to the client's never-string-match
  invariant.

## Benchmark integration recipes

- **Terminal-Bench via Harbor** — PyPI `harbor`, not `terminal-bench`. Custom agent =
  subclass `harbor.agents.base.BaseAgent`; the sandbox is driven by
  `await environment.exec(cmd) -> ExecResult`. The agent runs in-process on the host, so
  it reuses our loop and harness directly. Config arrives via `--agent-kwarg k=v`
  (JSON-typed) and `--agent-env K=V`. Per-attempt result at
  `<dir>/<job>/<task>__<id>/result.json` → `verifier_result.rewards`; TB2 normally uses
  `{"reward": 0|1}`. The runner resolves the `harbor` entry point beside
  `sys.executable`, not an arbitrary one on `PATH`: a `uv tool install harbor`
  environment cannot import this custom agent.

- **tau2-bench** — custom agent = subclass `tau2.agent.llm_agent.LLMAgent`, overriding
  `generate_next_message`. `run_task` hardcodes the `LLMAgent` constructor, so to attach
  harness wiring we drive `tau2.orchestrator.Orchestrator` per (task × trial). A paired
  user adapter preserves tau2's prompt/state/tool behavior while routing calls through
  PandaBench with the task agent's same resolved model and backend. Its distinct trace
  session prevents simulated-user spans from entering the harness trajectory. Some
  reasoning models can return `finish_reason=stop` with private reasoning but empty
  content; the adapter requires a visible customer turn and retries that invalid shape
  once, accumulating both attempts into tau2's `user_cost`.
  `Orchestrator.run()` does **not** grade (`reward_info=None`) — call
  `evaluate_simulation(..., evaluation_type=EvaluationType.ALL)` separately. Use `ALL`,
  never `ALL_WITH_NL_ASSERTIONS`, which calls an LLM. All three domains grade
  deterministically (airline/retail use `[DB, COMMUNICATE]`; telecom uses
  `[ENV_ASSERTION]`), so grading is free and doubles as the harness's gold outcome
  signal. `DATASET=airline|retail|telecom` switches task set, environment, policy, tools,
  and evaluator as one unit. `Orchestrator.run()` is blocking, so the runner drives it in
  a worker thread and the agent submits chat plus the per-turn barrier back to the
  runner's loop. Administrative calls are rejected at dispatch and cannot contaminate the
  domain transcript or grading.

## Verification status

- **Harness-live-throughout migration**: the learning/eval split, frozen eval, and
  `frozen-rules.json` are gone. Verified offline — full test suite, `ruff`, and
  `mypy --strict` clean, plus one dry run per benchmark confirming every trial has
  `phase == "live"`, the task set matches the dataset, no `frozen-rules.json` is written,
  and both AppWorld arms ran the identical seeded order.
- **AppWorld real integration**: server + HTTP env + runner driven against the live
  AppWorld environment server (initialize / api_docs / execute / evaluate / close). ✅
- **tau2 paid smoke**: four real retail episodes without integration errors; every
  session has a multi-sample trace series, the journal shows Tier-1 regressions plus
  Tier-2 breach signatures, and replay validation promoted two rules and retired eight.
- **Terminal-Bench paid smoke**: Harbor ran four serial Docker trials without
  exceptions; `result.json` supplied reward dictionaries and all four sessions have
  trace samples. Too few subsequent live sessions for a forward-only candidate to
  promote or retire, which is the documented Terminal-Bench deviation.
- **Not yet exercised by a paid run**: end-of-run settlement against real validation
  latency, and whether in-session promotion occurs often enough to move the metric.

## Checkpoint results

- **Checkpoint 1 (metric↔failure calibration)** still needs a labelled real harness run;
  the tooling exists (`pandabench-calibrate`, `scripts/labels_from_records.py`).
- **Checkpoint 2 (rule promotion)** engaged in the tau2 smoke: two replay-validated rules
  promoted, eight regressing candidates retired.
