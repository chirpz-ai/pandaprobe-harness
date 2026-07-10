# Implementation notes

Engineering record for the PandaBench suite: pinned versions, deviations from
`../docs/benchmark-study-brief.md` and why, checkpoint results as they come in,
and the sharp edges discovered while building. Read alongside the brief.

## Pinned versions (benchmarks/uv.lock)

- `pandaprobe-harness==0.6.3` (exact; only 0.6.3 is on PyPI) — from PyPI, never `../src`.
- `pandaprobe==0.4.0` (the SDK, for session binding + manual spans).
- `litellm` (>=1.55; lock pins the resolved version, currently 1.91.x).
- `pandas>=2.2`, `numpy>=2.0`, `scipy>=1.14`, `matplotlib>=3.9`, `tabulate`, `httpx`,
  `pyyaml`, `python-dotenv`.
- Python 3.13 (matches the harness); benchmark harnesses run in their own envs.

## Deviations from the brief (and why)

1. **`adapters/` lives under `src/pandabench/adapters/`, not top-level.** Harbor's
   `--agent-import-path pandabench.adapters.harbor_agent:PandaBenchAgent` requires
   `adapters` to be an importable subpackage of `pandabench`; a top-level dir would
   not import. `scripts/` stays top-level (thin CLI shims over package modules).

2. **AppWorld runs OUT-OF-PROCESS over HTTP; pandabench never imports it.**
   AppWorld (`0.1.3.post1`) pins `pydantic<2`, which is irreconcilable with modern
   LiteLLM (`pydantic>=2.10`) — uv cannot co-resolve them. So AppWorld is not a
   dependency of `pandabench`; instead it runs as its own *environment server*
   (`appworld serve environment`) in an isolated pydantic-v1 venv, and we drive it
   via the REST API AppWorld exposes for exactly this reason. `AppWorldServer` +
   `HttpAppWorldEnv` in `runners/appworld_env.py`. This keeps the pandabench core
   (LiteLLM, pydantic v2) conflict-free. **Verified end-to-end against the real
   server** (see Verification status).

3. **PandaProbe tracing is a MANUAL span per LiteLLM call.** The SDK auto-wraps
   only native clients (`wrap_openai/anthropic/gemini`); it has no LiteLLM wrapper.
   Since every study call goes through `litellm.acompletion`, `providers/tracing.py`
   opens `pandaprobe.session(sid)` + a `start_trace` + an `LLM`-kind span per call
   (recording messages/usage/cost) so sessions are scoreable. `start_trace` *raises*
   without creds, so the tracer guards on `get_client()` — arm A and offline tests
   never touch the SDK.

4. **`on_turn_end` is called ONCE per task-trial (by the runner), not per loop turn,
   and NOT via `harness.turn()`.** Eval-case capture stashes the non-empty
   `end_state` as `EvalCase.replay_input` (`hook/core.py:203-204`), and
   `harness.turn()` always sends `end_state={}` — so `turn()` can never capture. A
   task-trial IS one PandaProbe session (session metrics are holistic), and platform
   scoring lags past trial end, so per-turn detection would not fire mid-task anyway.
   The runner therefore calls `on_turn_end({... "end_state": <replay descriptor>})`
   then `refresh` + `drain_validation` once per trial. The harness preamble + tools
   are still injected on EVERY turn (arm B), so cross-task learning works. Unit-tested
   in `tests/test_glue_and_results.py::test_on_turn_end_capture_yields_replayable_eval_case`.

5. **Metrics/report/checkpoints were built alongside the AppWorld slice**, not in a
   separate later pass, because the vertical slice's acceptance gate is
   run → records → report end-to-end.

6. **Smoke (`make smoke`) runs in `--dry-run`** (mock model, mock benchmark envs) as
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
- **Claude 4.6+ reject `temperature`/`top_p`/`top_k`** — `models.yaml` per-model
  `param_allowlist` drops them; we filter explicitly (not via `litellm.drop_params`).
- **`tau2` PyPI name is ambiguous** — the `tau2` PyPI package is likely NOT the
  sierra-research benchmark; the real install is resolved in the tau2 phase (see
  below) and kept out of the pandabench deps (its own isolated env).

## Setup for real runs

- **AppWorld**: `make setup` provisions an isolated venv + data via
  `scripts/setup_appworld.sh` and prints the two env vars to export
  (`PANDABENCH_APPWORLD_PYTHON`, `APPWORLD_ROOT`). ~183 MB data download.
- **Terminal-Bench**: needs Docker running + `uv tool install harbor`.
- **tau2**: installed into its own env in the tau2 phase.
- **Providers**: export `VERTEXAI_PROJECT`/ADC, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  and `PANDAPROBE_API_KEY` (+ `PANDAPROBE_PROJECT_NAME`) — or put them in
  `benchmarks/.env`. `uv run pandabench-run --preflight` validates them.

## Benchmark integration recipes (verified APIs; real runs gated this session)

All three benchmark harnesses were installed and their real APIs read (the brief's
"verify against the installed version"). AppWorld is fully wired + real-verified.
Harbor and tau2 adapters are implemented against verified APIs
(`adapters/harbor_agent.py`, `adapters/tau2_agent.py`); their real *orchestration*
was not run this session because each needs an environment I could not fully
provision + verify here (Docker off; no live creds in-shell; isolated co-install
envs). The dry-run path keeps all three in the pipeline/smoke. The precise recipes:

- **Terminal-Bench via Harbor** — PyPI `harbor==0.18.0` provides the CLI (NOT
  `terminal-bench`). Custom agent = subclass `harbor.agents.base.BaseAgent` (impl
  `name`/`version`/`setup`/`run`); the sandbox is driven by
  `await environment.exec(cmd) -> ExecResult{stdout,stderr,return_code}`. The agent
  runs in-process on the host, so it reuses our loop + harness directly (bash tool =
  `environment.exec`). Config reaches it via `--agent-kwarg k=v` (JSON-typed →
  `__init__`) and `--agent-env K=V` (→ auto-injected into `exec`). Run:
  `harbor run -d terminal-bench@2.0 -a pandabench.adapters.harbor_agent:PandaBenchAgent
  -m <model> -k <k> -n 1 -o <dir> --ak arm=... --ak harness_root=...`. Per-attempt
  result at `<dir>/<job>/<task>__<id>/result.json` → `verifier_result.rewards`
  (single 0/1). GATES: Docker running; harbor installed in an env that also has
  pandabench (so the agent import resolves).

- **tau2-bench** — PyPI `tau2` is a DECOY (a magnetics package). Real install:
  `git+https://github.com/sierra-research/tau2-bench.git@v0.2.0` (→ `tau2==0.2.1.dev0`,
  import `tau2`, Python 3.12). Data is NOT shipped: set `TAU2_DATA_DIR=<clone>/data`.
  It pins `litellm<1.82.7` (conflicts with the core's 1.91) → runs in its OWN venv
  with pandabench co-installed. Custom agent = subclass `tau2.agent.llm_agent.LLMAgent`,
  override `generate_next_message` to route through our wrapper (done). tau2's
  `run_task` hardcodes the `LLMAgent(tools, domain_policy, llm, llm_args)` constructor,
  so to inject the harness we drive `tau2.orchestrator.Orchestrator` per (task×trial),
  keeping the user simulator on tau2's stock `generate()` (fixed model, arm-independent).
  Reward: `sim.reward_info.reward` (success ≈ 1.0); `compute_metrics` gives pass^k.
  GATES: isolated venv + `TAU2_DATA_DIR` + live creds.

## Verification status (this build)

- **Dry-run pipeline (all 3 benchmarks × both arms)**: run → records → manifest →
  resume-skip → report → summary artifacts. ✅ green (`make smoke`, `make report`,
  `tests/test_pipeline_smoke.py`).
- **AppWorld real integration**: `AppWorldServer` + `HttpAppWorldEnv` + `AppWorldRunner`
  driven against the **live AppWorld environment server** (57 dev tasks; initialize /
  api_docs / execute real code / evaluate 1-of-2 tests / close; runner `run_once` with
  a scripted tool call). ✅ verified. A full live-model smoke additionally needs LLM
  creds exported in the shell.
- **Arm-B capture path**: `on_turn_end` → breach → replayable `EvalCase`, with a fake
  `pandaprobe` CLI. ✅ unit-tested.
- **Provider layer**: model resolution, Claude backend switching, param allowlist,
  tool-call JSON parsing, cost fallback, usage. ✅ 16 unit tests.
- **Metrics**: pass@1, pass^k, McNemar (exact + chi-square), paired delta, bootstrap.
  ✅ unit-tested.
- **Root invariants**: `make lint`/`typecheck`/`test` at repo root stay green (348
  package tests pass; ruff `extend-exclude=["benchmarks"]` keeps `ruff check .` out).
  ✅ verified.

## Checkpoint results

- **Checkpoint 1 (metric↔failure calibration)** and **Checkpoint 2 (rule promotion)**
  require a real arm-B learning run (LLM + PandaProbe creds); the tooling is built
  (`pandabench-calibrate`, `scripts/labels_from_records.py`, `learning_outcome` in each
  manifest) and exercised structurally. Results are appended here as real runs land.
