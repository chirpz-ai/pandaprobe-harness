# PandaProbe Harness

A **meta-cognitive diagnostic sandbox** for PandaProbe-instrumented agents.

Instead of managing a model's internal context window through invasive database
middleware, the harness gives an agent the tools to diagnose and repair *itself*:

1. **Lifecycle Hook & Context Injector** — a lightweight, non-blocking
   `on_turn_end` hook that evaluates each completed turn via the PandaProbe
   platform (`agent_reliability` / `agent_consistency`). On a threshold breach it
   dumps a verbose telemetry payload to the workspace and injects a textual
   *System Alert* into the agent's next-turn message queue. It never mutates
   framework checkpoints.
2. **Diagnostic Sandbox** — a restricted `bash` environment exposing the
   `pandaprobe` CLI natively to the agent as a tool.
3. **Diagnostic Filesystem** — a persistent `/harness/` workspace
   (`harness_rules.md` living rules + `traces/latest_eval.json` failure dumps)
   that the agent reads and rewrites to **self-heal**.

The harness sits on top of two pre-existing tools — the PandaProbe **Core SDK**
(tracing) and the PandaProbe **CLI** (`pandaprobe`). It only ever shells out to
the CLI binary through one narrow, injectable seam (`CliClient`).

## Quick start

```bash
make install        # uv sync
make test           # offline unit + e2e suite
make lint typecheck # ruff + mypy --strict
```

## Architecture

```
src/pandaprobe_harness/
├── config.py          # HarnessConfig (paths, thresholds, eval flags)
├── cli/               # CliClient seam: subprocess client, errors, JSON models
├── evaluation/        # MetricEvaluator: run + poll + threshold checks
├── hook/              # PandaHarnessHook + SystemAlert builder
├── adapters/          # FrameworkAdapter protocol + RawLoop (+ thin stubs)
├── sandbox/           # RestrictedShellTool + ShellPolicy
└── filesystem/        # HarnessFilesystem provisioner + rules template
```

See `docs`/the approved plan for the full design rationale.
