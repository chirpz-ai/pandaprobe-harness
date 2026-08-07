# Managed repair architecture note

This breaking root-package change separates the developer-owned task agent from
PandaProbe-owned workspace administration.

## API and role boundary

- `Harness.task_tools` exposes only rule read, search, list, and status.
- `Harness.system_context(session_id, task_hint=...)` returns only a stable
  capability note. It renders no rules or expanded index and performs no rule
  retrieval.
- `Harness.settle(session_id)` waits for task evaluation and one package-owned
  repair attempt, then returns a structured repair result without failing the
  developer task.
- `HarnessConfig` owns repair-model, budget, tracing, and domain-policy
  settings. A mutating harness requires an explicit repair model.
- `ManagedRepairAgent` owns the repair prompt and bounded tool loop. A narrow
  completion transport invokes LiteLLM only through PandaProbe's official
  wrapper; tests replace that transport, not the orchestration.
- `repair_reasoning_effort` defaults to `"none"` and is forwarded only when
  LiteLLM's model registry reports support. This keeps current OpenAI reasoning models
  compatible with function tools on the wrapped chat-completions path; all
  provider identifiers still use the same transport and orchestration.

Task dispatch rejects every administrative operation. A repair-scoped
dispatcher can read its assigned episode notices and evidence, inspect existing
rules, add at most one candidate, and resolve the episode. It cannot call domain
tools, promote candidates, or retire rules. Validation/regression remains the
lifecycle authority for promotion and retirement.

## Per-turn sequence

1. The host runs and traces its own task-agent turn.
2. The host flushes tracing and calls `on_turn_end` with task/end-state data.
3. The existing exact-session evaluator and trajectory gate persist a notice.
4. Related notices for the session/turn form one evidence-overlapping repair
   episode, single-flighted under `repair-<task-session>-<episode-id>`.
5. Repair either writes a provisional candidate and acknowledges the notice,
   or records an explicit duplicate/no-proposal resolution.
6. `settle` returns the cached repair outcome; timeouts and failures leave the
   task successful and the notice recoverable.
7. Before the next host turn, `harness_guide.md` and its scope file have been
   refreshed. The candidate is discoverable through list/read but is not injected.
   Replay validation remains detached, exposes the same on-demand read tools, and
   may later promote or retire it.

Repair work never calls the task hook. Repair model calls run in a clean async
context, use a distinct SDK session, and are therefore excluded from exact
task-session trace discovery and trajectory history.

When repair tracing is enabled, the assignment exports a single trace named
`pandaprobe`. A `harness` CHAIN span owns alternating `repair-agent` and `tools`
AGENT spans. PandaProbe's LiteLLM wrapper contributes each nested LLM span,
while the package-owned loop contributes one TOOL child for each restricted
workspace call. With tracing disabled, the same SDK context is non-exporting
so the wrapper cannot create an accidental standalone trace.
