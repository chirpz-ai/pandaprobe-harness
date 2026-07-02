# PandaProbe Harness

The harness layer that turns any PandaProbe-instrumented agent into a
**self-healing agent**. It is autonomous and **pull-based**: after each turn the
harness evaluates the session on the PandaProbe platform and, on a threshold
breach or a declining trend, posts a structured *diagnostic notice* to a
workspace **mailbox**. The agent — guided by a standing protocol in its system
prompt — checks its own mailbox, analyzes its own traces, records a permanent
mitigation rule, and acknowledges the notice. Nothing is ever injected into the
agent's message queue. The core has **zero runtime dependencies**; every piece
of platform access goes through one narrow, injectable seam around the
`pandaprobe` CLI.

## How it works

```
 producing side (the harness)                consuming side (the agent)
 ────────────────────────────                ──────────────────────────
 turn end                                    turn start
    │                                           │
    ▼                                           ▼
 PandaHarnessHook.on_turn_end            system context = rules + protocol
    │  budget → sampling → rate-limit         + "⚠ N pending notice(s)" banner
    │  gates; supersedes in-flight eval        │
    ▼                                          ▼
 detached eval task                       harness tools (toolset / companion CLI
    │  (Semaphore(max_concurrent_evals))       / restricted shell)
    │  pandaprobe evals runs batch             │  harness_mailbox_list
    │    --target session ...                  │  harness_mailbox_read  (+ dump)
    │  poll: evals runs scores <run>           │  harness_trace_inspect
    ▼                                          │  harness_journal / harness_history
 EWMA trend detection                          ▼
    ▼                                     harness_rule_add ──▶ rules.jsonl
 dedup / cooldown / recovery gate              │                └▶ harness_rules.md
    │  breach | relative | trend               ▼                   (next context)
    │  (circuit breaker ⇒ needs_human)    harness_mailbox_ack
    ▼                                          │
 mailbox/pending/<id>.json                     ▼
  + traces/<id>.json + latest_eval.json    pending → processed; banner clears
  + journal.jsonl "notice" event
```

## Quick start

```bash
pip install pandaprobe-harness        # or: make install (uv sync)
```

Any custom agent loop integrates in a handful of lines — no adapter required:

```python
from pandaprobe_harness import Harness
from pandaprobe_harness.agent_tools.native import as_anthropic_tools

harness = Harness.create()                     # provisions /harness, wires all parts,
                                               # runs the startup health check
system_prompt = harness.system_context() + MY_PROMPT   # rules + protocol + banner

specs, dispatch = as_anthropic_tools(harness.toolset)  # register the 9 harness tools
tools = my_tools + specs                       # also: as_langchain_tools(...),
                                               #       as_openai_function_tools(...)

async def one_turn(session_id: str, user_input: str) -> str:
    async with harness.turn(session_id):       # fires turn-end on exit (even on error)
        return await my_agent_step(system_prompt, tools, user_input)

# equivalents: await harness.run_turn(session_id, my_agent_step, ...)
#              decorated = harness.turn(session_id)(my_agent_step)
```

For sandboxed or framework-less agents there is a second delivery channel: give
the agent `harness.shell` (a `RestrictedShellTool`) and it reaches the same
toolset through the allow-listed companion binary, e.g.
`pandaprobe-harness-agent harness_mailbox_list`.

## The pull loop

`harness.system_context()` prepends three things to the agent's system prompt:
the rendered active rules, the **standing self-diagnostic protocol**, and — when
notices are pending — a compact mailbox banner (a count plus a severity enum;
no eval-derived free text).

The protocol tells the agent, at the start of each turn and for **each** pending
notice: read it in full including the trace dump (`harness_mailbox_read`),
inspect the flagged traces (`harness_trace_inspect`), compare with cross-run
memory (`harness_journal`, `harness_history`), record a permanent mitigation
rule with rationale and provenance (`harness_rule_add`), then acknowledge the
notice linking the rule (`harness_mailbox_ack`). Periodically it runs
`harness_reflect` to generalize repeated mitigations and retire ineffective
rules.

Notice severities, lowest to highest: `trend` (advisory EWMA decline) <
`relative` (adaptive drop below the session's own baseline) < `breach` (score
under the absolute floor) < `needs_human`. A **circuit breaker** guards against
notice storms: more than `circuit_breaker_max_notices` notices inside
`circuit_breaker_window_s` escalates to a single `needs_human` notice and
suppresses further posting until the window drains — the protocol instructs the
agent to surface that one to a human rather than act on it.

## Toolset reference

| Operation | Purpose |
| --- | --- |
| `harness_mailbox_list` | Pending notice summaries plus mailbox status (count, max severity). |
| `harness_mailbox_read` | One notice in full, including its trace dump. |
| `harness_mailbox_ack` | Acknowledge a pending notice, optionally linking the resolving rule. |
| `harness_trace_inspect` | One flagged trace via the platform: trace, TOOL spans, trace-level scores. |
| `harness_history` | Score trajectory for a metric: local series + backend session scores. |
| `harness_journal` | Recent journal events — the cross-run memory for recurring patterns. |
| `harness_rule_add` | Record a permanent mitigation rule (dedup-safe; fails at the active cap). |
| `harness_rule_retire` | Retire a rule that proved ineffective or obsolete. |
| `harness_reflect` | Cross-run context for a rules refactor: notices, rules, effectiveness counts. |

Every operation returns a JSON envelope with an `"ok"` key; failures never raise
into the agent loop. The companion CLI form is
`pandaprobe-harness-agent <op> --key value ...` (values parsed as JSON when
possible; exit code 0 iff `ok`).

## Workspace layout

```
/harness/                        (HARNESS_ROOT)
├── mailbox/
│   ├── pending/<notice-id>.json    # posted by the hook, pulled by the agent
│   ├── processed/<notice-id>.json  # acknowledged notices, with resolution
│   └── status.json                 # cheap summary read by the banner
├── journal.jsonl                   # append-only cross-run event log
├── rules.jsonl                     # structured rules (latest record per id wins)
├── harness_rules.md                # rendered artifact: template + active rules
├── traces/
│   ├── latest_eval.json            # most recent eval dump (always written)
│   └── <notice-id>.json            # one immutable dump per notice
└── state/score_history.json        # per-(session, metric) series + EWMA state
```

## Framework integrations

Turn detection is the **only** thing that differs per framework. All six share
identical self-healing: the same mailbox, the same toolset, the same system
context. Adapters are optional, import-guarded extras
(`pip install 'pandaprobe-harness[langgraph]'`, `[all]`, …); session identity
comes from the PandaProbe SDK's session `ContextVar`.

| Factory | Framework | Turn detector |
| --- | --- | --- |
| `Harness.for_langgraph()` | LangGraph | LangChain async callback: root `on_chain_end` (via `harness.adapter.make_callback()`) |
| `Harness.for_langchain()` | LangChain | Same root-chain-end callback |
| `Harness.for_deepagents()` | DeepAgents | Same root-chain-end callback |
| `Harness.for_crewai()` | CrewAI | `wrapt` patch of `Crew.kickoff` (auto-`instrument()`) |
| `Harness.for_claude_agent_sdk()` | Claude Agent SDK | `wrapt` patch of `ClaudeSDKClient.receive_response` (one stream = one turn) |
| `Harness.for_openai_agents()` | OpenAI Agents SDK | `TracingProcessor` — hook fires on trace end (one `Runner.run` = one turn) |

LangGraph, condensed:

```python
import pandaprobe
from pandaprobe_harness import Harness
from pandaprobe_harness.agent_tools.native import as_langchain_tools

harness = Harness.for_langgraph()
handler = harness.adapter.make_callback()
graph = build_graph(system=harness.system_context() + PROMPT,
                    tools=my_tools + as_langchain_tools(harness.toolset))

with pandaprobe.session(session_id):     # harness and SDK traces share the session
    await graph.ainvoke(state, config={"callbacks": [handler],
                                       "configurable": {"thread_id": session_id}})
```

## How evaluation works

`agent_reliability` and `agent_consistency` are **session-level** metrics, so a
completed turn is evaluated by session id in one batch run covering all
configured metrics:

```
pandaprobe evals runs batch --target session --session-ids <id> \
    --metrics agent_consistency,agent_reliability      # async; returns run_id
pandaprobe evals runs scores <run_id> --target session # polled until terminal
```

Polling is bounded (`poll_interval_s` × `poll_max_attempts`). Trace ingestion
lags turn-end (the SDK flushes on a background thread), so transiently
empty/not-found runs are retried with backoff (`eval_retry_attempts`,
`eval_retry_backoff_s`). Scores are `0.0–1.0`, higher is better; an **absolute
breach** is `score < threshold` (default `0.5`, per-metric overrides supported).
Any persistent CLI failure degrades to a pending score — the harness never
raises into, or blocks, the host loop.

Beyond absolute breaches, three local detectors run over the score the harness
already fetched (O(1), no extra network call):

- **Trend** — dual-EWMA crossover (fast span 3 vs. slow span 10): declining when
  the fast average drops below the slow one by `trend_margin_cross`, after
  `trend_min_samples` samples.
- **Relative** (opt-in) — adaptive threshold: a score falling
  `adaptive_margin_drop` below its own session baseline (the slow EWMA) breaches
  even while above the absolute floor.
- **Percentile** (opt-in) — a percentile-over-window corroborator against the
  same local history.

Repeated identical conditions are de-duplicated per session (with an optional
turn-count cooldown) and reset on recovery, so a persistent decline posts
exactly one notice.

## Cost & reliability controls

| Control | Effect |
| --- | --- |
| `eval_sample_every` | Evaluate every Nth turn per session (1 = every turn). |
| `session_min_eval_interval_s` | Per-session rate limit between eval launches. |
| `max_concurrent_evals` | Global `asyncio.Semaphore` cap across all sessions. |
| `max_evals_per_run` | Hard per-process budget of eval launches (0 = unlimited). |
| `observe_only` | Shadow mode: evaluate + journal, but never post mailbox notices. |
| Circuit breaker | Escalates notice storms to one `needs_human`; auto-resets on recovery. |
| Startup health check | `pandaprobe version` + `auth status` before the first eval; on failure the harness runs *degraded* — one warning, a journal event, evals skipped, never a crash. |

A newer turn always **supersedes** a session's in-flight evaluation (the stale
task is cancelled). `await harness.refresh(session_id)` is a bounded join on the
in-flight eval (`drain_timeout_s`); `refresh_all()` joins everything. Both exist
for tests and explicit callers — correctness does not depend on them, since each
eval task handles its own result.

## Security model

- **Credential scoping** — the restricted shell scrubs credential-shaped
  environment variables (`*API_KEY*`, `*SECRET*`, `*TOKEN*`, `PANDAPROBE_*`, …)
  from every subprocess, restoring the PandaProbe auth variables only for the
  `pandaprobe` / `pandaprobe-harness-agent` binaries. `jq`, `cat`, and `ls`
  never see them.
- **Argv policy** — allow-listed binaries only; denied argv prefixes
  (`pandaprobe config`, `pandaprobe auth login/logout`) and denied flags
  (`--reveal-secrets`); no shell metacharacters, `shlex` + `exec` (never
  `shell=True`); path arguments may not escape the workspace.
- **Prompt-injection trust boundary** — all eval-derived free text (platform
  `reason` strings, summaries, agent-authored rules) passes through
  `sanitize_text` before entering agent context: ANSI/control stripping, banner
  runs collapsed, the harness's own framing phrases neutralized, length capped.
  The protocol additionally instructs the agent that notice/dump/trace contents
  are untrusted **data**, never instructions.
- **Container isolation (recommended)** — run the agent with a read-only
  filesystem except a volume at `/harness`, and an egress allowlist limited to
  the PandaProbe endpoint. `docker-compose.yml` / `Dockerfile.sandbox` provide a
  reference sandbox (`make up`, `make harness-shell`).

## Scaling

Horizontally-scaled agents converge on shared trend state: with
`hydrate_history_from_backend` enabled, the hook seeds the local history once
per session from `pandaprobe evals scores list --target session` before the
first eval, so EWMA baselines survive process restarts and replica fan-out
(`HistorySource` is a Protocol — a remote store can replace the local JSON one).
All workspace stores (mailbox, journal, rules, history) are lock-guarded with
atomic writes and append-only logs, safe for one workspace shared by many
sessions on the thread pool.

## Configuration reference

`HarnessConfig.from_env()` reads `HARNESS_*` variables; explicit overrides win.
The most important ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HARNESS_ROOT` | `/harness` | Workspace root. |
| `HARNESS_CLI_BINARY` | `pandaprobe` | CLI binary (all platform access). |
| `HARNESS_RELIABILITY_THRESHOLD` | `0.5` | Absolute floor for `agent_reliability`. |
| `HARNESS_CONSISTENCY_THRESHOLD` | `0.5` | Absolute floor for `agent_consistency`. |
| `HARNESS_ENABLE_TREND` | `true` | Dual-EWMA trend detection. |
| `HARNESS_ADAPTIVE_THRESHOLD` | `false` | Relative-drop detector. |
| `HARNESS_PERCENTILE_WINDOW` | `0` | Percentile corroborator window (0 = off). |
| `HARNESS_ALERT_COOLDOWN_TURNS` | `0` | Re-notice cooldown (0 = only on new conditions). |
| `HARNESS_OBSERVE_ONLY` | `false` | Shadow mode (journal-only). |
| `HARNESS_CIRCUIT_BREAKER_MAX_NOTICES` | `5` | Notices per window before `needs_human` (0 = off). |
| `HARNESS_CIRCUIT_BREAKER_WINDOW_S` | `600` | Circuit-breaker window. |
| `HARNESS_EVAL_SAMPLE_EVERY` | `1` | Evaluate every Nth turn. |
| `HARNESS_SESSION_MIN_EVAL_INTERVAL_S` | `0` | Per-session eval rate limit. |
| `HARNESS_MAX_CONCURRENT_EVALS` | `4` | Global eval concurrency cap. |
| `HARNESS_MAX_EVALS_PER_RUN` | `0` | Per-process eval budget (0 = unlimited). |
| `HARNESS_MAX_ACTIVE_RULES` | `50` | Active-rule cap (retire before add). |
| `HARNESS_DRAIN_TIMEOUT_S` | `15` | Budget for `refresh` / `refresh_all` joins. |
| `HARNESS_HEALTH_CHECK` | `true` | Startup CLI/auth verification. |
| `HARNESS_HYDRATE_HISTORY_FROM_BACKEND` | `false` | Seed trend history from the backend. |

## Migration from 0.4

v0.5 replaces alert *injection* with the pull model. Nothing is spliced into
the agent's messages anymore.

| 0.4 API | 0.5 replacement |
| --- | --- |
| `inject_alert` / adapter alert injection | Hook posts to the `Mailbox`; agent pulls via the toolset. |
| `adapter.consume_messages()` / `drain_into(...)` | `harness.toolset` (`harness_mailbox_list` / `_read` / `_ack`). |
| `adapter.startup_messages()` / `prime_startup()` / `startup_input_items()` | `harness.system_context()` prepended to the system prompt. |
| `hook.drain_pending(session_id)` | `await harness.refresh(session_id)` — a bounded join, no longer required for correctness. |
| `filesystem.append_rule(...)` | `RulesStore.add(...)` / the `harness_rule_add` tool (structured, with provenance). |
| `PandaHarnessHook(adapter, cli, ...)` | `PandaHarnessHook(cli, ...)`, or a `Harness.create()` / `Harness.for_*()` factory. |

## Development

```bash
make install         # uv sync
make test            # full offline suite (unit + e2e); no network, no real CLI
make lint typecheck  # ruff + mypy --strict
make test-contract   # live CLI contract tests (PANDAPROBE_LIVE=1 + credentials)
make example         # offline self-heal demo: examples/offline_self_heal.py
```

The suite is **fully offline** by design: platform behaviour is modelled by a
fake `CliClient` (and a fake `pandaprobe` binary for subprocess-level tests), so
every gate, trend, and self-heal cycle is exercised deterministically. Only the
opt-in contract tests touch the real CLI. See `examples/` for runnable
end-to-end scenarios and `tests/e2e_pull_loop_test.py` for the canonical pull
loop.

## Architecture

```
src/pandaprobe_harness/
├── config.py          # HarnessConfig: paths, thresholds, trend/cost/safety knobs
├── harness.py         # Harness facade: create() / for_*() factories, turn scopes
├── cli/               # CliClient seam: subprocess client, typed errors, JSON models
├── evaluation/        # MetricEvaluator, ScoreHistoryStore, HistorySource, TrendDetector
├── hook/              # PandaHarnessHook (turn end → eval → notice) + system context
├── workspace/         # Mailbox, Journal, RulesStore, sanitize (trust boundary), atomic I/O
├── agent_tools/       # HarnessToolset (9 ops), native adapters, companion CLI, specs
├── adapters/          # Turn detectors: raw loop, LangGraph/LangChain/DeepAgents,
│                      #   CrewAI, Claude Agent SDK, OpenAI Agents
├── sandbox/           # RestrictedShellTool + ShellPolicy (allow-list, env scoping)
├── filesystem/        # /harness provisioner + harness_rules.md template
└── monitors/          # MonitorClient (wraps the CLI's `evals monitors`)
```
