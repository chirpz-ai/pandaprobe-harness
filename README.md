# PandaProbe Harness

PandaProbe Harness evaluates any developer-owned, PandaProbe-instrumented task
agent and maintains a shared learned-rules workspace with a separate,
package-owned repair agent.

[![PyPI](https://img.shields.io/pypi/v/pandaprobe-harness)](https://pypi.org/project/pandaprobe-harness/)
[![CI](https://github.com/chirpz-ai/pandaprobe-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/chirpz-ai/pandaprobe-harness/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://pypi.org/project/pandaprobe-harness/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Ownership model

The developer owns the task agent, model, framework, prompts, domain tools,
execution loop, and environment. PandaProbe owns task instrumentation and
evaluation, trajectory detection, diagnostic notices, the managed repair-agent
loop, workspace administration, validation, and read-only rule delivery.

Task and repair activity are isolated:

```text
developer task agent → task trace → evaluation/gate → notice
                                                ↓
                                  PandaProbe managed repair
                                                ↓
                                     candidate rule
                                                ↓
developer task agent ── read-only tools ──→ shared workspace
```

The task agent never reads notices, inspects diagnostic traces, acknowledges
notices, writes or retires rules, or controls validation. Its harness tools are
exactly `harness_rules_read`, `harness_rules_search`, `harness_rules_list`, and
`harness_rule_status`, all read-only.

### Learned rules are read on demand, never injected

`system_context()` never includes rule bodies or an expanded rule-file index and
performs no implicit rule read or search. It contains only a stable note that
learned rules are available through those four tools. The task agent decides
whether and when to list scopes, search, or read one. Workspace
`rules.md` holds SKILL-style task-facing instructions and the compact
generated scope index; each entry carries a bounded description and
active/provisional counts, never rule text.

### Scope: where a rule is filed

Managed repair decides, from the failure evidence it already has, whether a rule
is broadly applicable or belongs to a specific context. That decision is part of
the existing repair call — no extra model round.

- `global` is the default: broadly reusable rules, not tied to one task,
  workflow, application, tool, or domain.
- A concise contextual name — an application, workflow, or domain drawn from the
  evidence — is preferred whenever the rule really belongs to that context. The
  catalog is open, and a new name simply creates its `rules/<scope>.md`.
- `scoped` is the fallback: the rule is specific, but no meaningful stable name
  could be determined.

Scope naming is generic and not tied to any benchmark or integration; a host's
own label for itself is rejected as a scope. Hosts may pass bounded
`RuleScopeHint` metadata and a short `task_summary` to inform the decision, but
neither dictates it. PandaProbe normalizes names only for filename safety, owns
the resulting path, and imposes no prefix convention.

### Validation decides promotion and retirement

A new rule enters as a provisional candidate and only validation promotes or
retires it — never the repair agent. Replay is the strong path; a cheap
forward trial over live sessions decides candidates replay cannot reach, so every
candidate reaches a verdict. Call `settle_validation()` at a phase boundary before
snapshotting or reporting a ruleset.

## Installation

```bash
pip install pandaprobe-harness
```

Managed repair uses PandaProbe's official LiteLLM wrapper. The same normalized
path accepts LiteLLM identifiers for OpenAI, Anthropic API, Anthropic on
Bedrock, and Gemini on Vertex AI:

```text
openai/...
anthropic/...
bedrock/anthropic....
vertex_ai/...
```

No potentially billable model is selected by default. Configure one explicitly
with `repair_model=` or `HARNESS_REPAIR_MODEL`. Provider credentials and cloud
settings follow LiteLLM conventions.

## Generic task-loop integration

```python
from pandaprobe_harness import Harness, HarnessConfig, RuleScopeHint
from pandaprobe_harness.agent_tools.native import as_anthropic_tools

harness = Harness.create(
    HarnessConfig(
        repair_model="openai/gpt-...",
        repair_timeout_s=60,
        repair_max_turns=6,
        repair_max_tokens=4096,
        repair_temperature=None,
        repair_reasoning_effort="none",  # current OpenAI reasoning models + tools
        trace_repair_agent=False,
        domain_policy="Describe authorized domain behavior here.",
    )
)

async def one_turn(session_id: str, user_input: str) -> str:
    context = harness.system_context(session_id, task_hint=user_input)
    rule_specs, rule_dispatch = as_anthropic_tools(harness.task_tools)

    # You still construct and run your own agent with your own domain tools.
    answer = await my_agent_step(
        system_prompt=context + MY_PROMPT,
        tools=[*my_domain_tools, *rule_specs],
        tool_dispatch=rule_dispatch,
        user_input=user_input,
    )

    # Flush/export task tracing first, then register the completed task turn.
    harness.on_turn_end({
        "session_id": session_id,
        "turn_index": next_index(),
        "end_state": end_state(),
        "rule_scope_hints": [
            RuleScopeHint(
                key="payments",
                description="Payment authorization and transaction workflows.",
            ).to_json()
        ],
    })
    settlement = await harness.settle(session_id)
    return answer
```

The host must settle before starting the next task turn when same-session repair
is desired. Settlement waits for task evaluation, notice persistence, and one
bounded managed repair attempt. It does not synchronously wait for domain replay
validation. A candidate is immediately discoverable through list/read after
settlement, but is never inserted into the next prompt; validation may promote or
retire it later.

Before a task turn, the host may attach the stable capability preamble and four
read-only tools. The harness injects no learned content and makes no automatic
rule-tool call. After the turn, tracing is flushed, evaluation/gating run, related
notices are grouped into one bounded repair episode, and managed repair may add at
most one provisional candidate or resolve without one. Successful settlement
atomically refreshes the index/scope artifacts before the next task turn can query
them; timeout or failure leaves every notice recoverable.

`settlement.repair` exposes status, repair/task session IDs, repair episode and
notice IDs, recommended/selected scope, considered/existing/candidate rule IDs,
suppression reason, model turns/tool calls, normalized usage when available, and
error category. Repair failure or timeout never fails the developer task.

Framework turn detectors remain available through `Harness.for_langgraph()`,
`for_langchain()`, `for_deepagents()`, `for_crewai()`,
`for_claude_agent_sdk()`, and `for_openai_agents()`.

## Configuration

Managed repair fields and environment equivalents are:

| Field | Environment |
| --- | --- |
| `repair_model` | `HARNESS_REPAIR_MODEL` |
| `repair_timeout_s` | `HARNESS_REPAIR_TIMEOUT_S` |
| `repair_max_turns` | `HARNESS_REPAIR_MAX_TURNS` |
| `repair_max_tokens` | `HARNESS_REPAIR_MAX_TOKENS` |
| `repair_temperature` | `HARNESS_REPAIR_TEMPERATURE` |
| `repair_reasoning_effort` | `HARNESS_REPAIR_REASONING_EFFORT` |
| `trace_repair_agent` | `HARNESS_TRACE_REPAIR_AGENT` |
| `domain_policy` | `HARNESS_DOMAIN_POLICY` |

`observe_only=True` remains non-mutating and does not require a repair model.
Managed repair requires `rule_validation=True`, so a repair-authored rule can
never skip the candidate lifecycle. Task tracing is unchanged when repair
tracing is disabled. When enabled, the
PandaProbe SDK records repair completions under
`repair-<task-session>-<episode-id>` with repair-role metadata; exact-session
task trace discovery excludes them.

Each enabled repair run exports one trace named `pandaprobe`. Its `harness`
CHAIN span contains repeated `repair-agent` and `tools` AGENT spans; each
`repair-agent` contains the official wrapper's `litellm-chat` LLM span, and
each `tools` span contains one TOOL child per restricted workspace operation.
No second tool-only trace is created.

## Offline examples and operator tools

```bash
uv run python examples/misc/offline_self_heal.py
uv run python examples/misc/closed_loop_self_heal.py
uv run python examples/misc/calibration_demo.py
```

The examples use deterministic completion fakes and require no credentials.
Operator-only CLIs remain `pandaprobe-harness-eval` for regression replay and
`pandaprobe-harness-calibrate` for threshold calibration. The former
task-administration companion CLI has been removed.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run mypy
```

See [CHANGELOG.md](CHANGELOG.md) for migration notes and
[CONTRIBUTING.md](CONTRIBUTING.md) for project invariants.

## License

[MIT](LICENSE) © [Chirpz AI](https://chirpz.ai)
