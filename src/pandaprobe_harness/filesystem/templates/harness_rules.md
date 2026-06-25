# Harness Rules

These are the **living rules** of this diagnostic harness. They are read into
the agent's startup context on every run. The agent is expected to *extend*
this file — never to discard it — as it learns from its own failures.

## The self-healing contract

This harness passively evaluates every completed turn via the PandaProbe
platform using two metrics:

- **`agent_reliability`** (trajectory / TRACER) — worst-case failure risk across
  the turn's traces.
- **`agent_consistency`** (session) — overall stability across the session.

Both score in `[0.0, 1.0]` where **higher is better**. A score **below 0.5**
(the default critical threshold) is a *breach*.

When a breach is detected, the harness:

1. Writes a verbose diagnostic dump to `traces/latest_eval.json`.
2. Injects a `SYSTEM ALERT` into your next turn.

On receiving that alert you MUST, before continuing the user's task:

1. Read `traces/latest_eval.json`.
2. Use the `pandaprobe` CLI to inspect what went wrong (e.g.
   `pandaprobe evals scores get <trace-id>`, `pandaprobe traces get <trace-id>`).
3. Reason about the failure (looping, redundant tool calls, tool-alignment).
4. Append a permanent mitigation rule to the **Learned Mitigations** section
   below so the failure mode never recurs.

## Baseline rules

- Rule 1: Never repeat an identical tool call without first inspecting the
  result of the previous call.
- Rule 2: Prefer reading existing state before mutating it.

## Learned Mitigations

<!-- The agent appends timestamped, attributed mitigation rules below. -->
