# PandaProbe Harness

A **meta-cognitive diagnostic sandbox** for PandaProbe-instrumented agents.

Instead of managing a model's internal context window through invasive database
middleware, the harness gives an agent the tools to diagnose and repair *itself*:

1. **Lifecycle Hook & Context Injector** — a lightweight, non-blocking
   `on_turn_end` hook that evaluates each completed turn via the PandaProbe
   platform (`agent_reliability` / `agent_consistency`, both **session-level**
   metrics). On a threshold breach **or a declining trend** it dumps a verbose
   telemetry payload to the workspace and injects a *System Alert* into the
   agent's next-turn message queue. It never mutates framework checkpoints.
2. **Diagnostic Sandbox** — a restricted `bash` environment exposing the
   `pandaprobe` CLI natively to the agent as a tool.
3. **Diagnostic Filesystem** — a persistent `/harness/` workspace
   (`harness_rules.md` living rules + `traces/latest_eval.json` failure dumps +
   `state/score_history.json` trend state) that the agent reads and rewrites to
   **self-heal**. The learned rules are re-injected into the agent's startup
   context every run, closing the loop.

The harness sits on top of two pre-existing tools — the PandaProbe **Core SDK**
(tracing) and the PandaProbe **CLI** (`pandaprobe`). The core has **no runtime
dependencies** and only ever shells out to the CLI through one narrow, injectable
seam (`CliClient`) — including **monitor management** (`MonitorClient` wraps the
CLI's `evals monitors` commands), so authentication is always the CLI's job.

## Quick start

```bash
make install        # uv sync
make test           # offline unit + e2e suite
make lint typecheck # ruff + mypy --strict
```

## How evaluation works

Both metrics are **session-scoped**. Each completed turn triggers one batch run:

```
pandaprobe evals runs batch --target session --session-ids <id> \
    --metrics agent_reliability,agent_consistency
pandaprobe evals runs scores <run_id> --target session   # polled to terminal
```

Scores are `0.0–1.0` (higher is better). An **absolute breach** is `score <
threshold` (default `0.5`). Trace ingestion lags turn-end, so runs are retried
with backoff while transiently empty.

## Trend alerting

Beyond absolute breaches, the harness flags **gradual decline** even while a
score is still above its floor. The detector is a local **dual-EWMA crossover**
(`evaluation/trends.py`) fed by the score the harness already obtained — O(1),
no extra network call on the turn path, recency-weighted, noise-robust. An
optional **adaptive (relative) threshold** and a **percentile-over-window**
corroborator reuse the same local history store. A declining trend raises a
distinct, advisory `TREND ALERT`. Repeated alerts are de-duplicated per session
(with optional cooldown) and reset on recovery.

## Integrating with LangGraph (sketch)

```python
from pandaprobe_harness import PandaHarnessHook, HarnessConfig, HarnessFilesystem, SubprocessCliClient
from pandaprobe_harness.adapters import LangGraphAdapter

adapter = LangGraphAdapter()
hook = PandaHarnessHook(adapter, SubprocessCliClient(), config=HarnessConfig.from_env())
adapter.register(hook)
HarnessFilesystem(hook_config).provision()

handler = adapter.make_callback()        # fires on root chain end
state = {"messages": adapter.startup_messages() + [user_message]}  # rules preamble

# each turn (async):
await hook.drain_pending(session_id)     # inject prior breach/trend alert
adapter.drain_into(state["messages"])    # merge pending alerts as SystemMessages
result = await graph.ainvoke(state, config={"callbacks": [handler],
                                            "configurable": {"thread_id": session_id}})
```

Run it inside `with pandaprobe.session(session_id): ...` so the harness and the
SDK traces share a session id.

**Supported frameworks** (optional extras, mirroring the SDK's integrations):
`LangGraphAdapter`, `LangChainAdapter`, `DeepAgentsAdapter` share the LangChain
callback model (turn-end on root chain end; state-level `SystemMessage`
injection). `CrewAIAdapter` and `ClaudeAgentSDKAdapter` instrument via `wrapt`
(`.instrument()`); the Claude adapter can inject directly into the SDK's
conversation history. `OpenAIAgentsAdapter` instruments via the Agents SDK
`TracingProcessor` and is observation-only — alerts are surfaced as input items
for the developer to prepend to the next `Runner.run` (the framework exposes no
in-flight injection point). All reuse the SDK's session `ContextVar`.

## Architecture

```
src/pandaprobe_harness/
├── config.py          # HarnessConfig (paths, thresholds, trend/EWMA knobs, API auth)
├── cli/               # CliClient seam: subprocess client, errors, JSON models
├── evaluation/        # MetricEvaluator + ScoreHistoryStore + TrendDetector
├── hook/              # PandaHarnessHook + alert builders + rules→context
├── adapters/          # FrameworkAdapter protocol + RawLoop, LangGraph, LangChain,
│                       #   DeepAgents, CrewAI, Claude Agent SDK, OpenAI Agents
├── sandbox/           # RestrictedShellTool + ShellPolicy
├── filesystem/        # HarnessFilesystem provisioner + rules template
└── monitors/          # MonitorClient (wraps the CLI's `evals monitors`; scheduled eval monitors)
```
