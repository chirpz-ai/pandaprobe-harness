# PandaProbe Harness

PandaProbe Harness evaluates any developer-owned, PandaProbe-instrumented task
agent and maintains a shared learned-guidance workspace with a separate,
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
                                      candidate guidance
                                                ↓
developer task agent ← read-only next-turn context ← shared workspace
```

The task agent never reads notices, inspects diagnostic traces, acknowledges
notices, writes or retires rules, or controls validation. Its optional harness
tools are exactly `harness_rules_read`, `harness_rules_search`,
`harness_rules_list`, and `harness_rule_status`.

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
from pandaprobe_harness import Harness, HarnessConfig
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
    harness.on_turn_end(
        {"session_id": session_id, "turn_index": next_index(), "end_state": end_state()}
    )
    settlement = await harness.settle(session_id)
    return answer
```

The host must settle before starting the next task turn when same-session repair
is desired. Settlement waits for task evaluation, notice persistence, and one
bounded managed repair attempt. It does not synchronously wait for domain replay
validation, so a candidate may appear provisionally on the next turn and be
promoted or retired later.

`settlement.repair` exposes status, repair/task session IDs, notice ID, model
turns, tool calls, candidate IDs, normalized token/cost usage when available,
and error category. Repair failure or timeout never fails the developer task.

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
Managed repair requires `rule_validation=True`, so repair-authored guidance can
never skip the candidate lifecycle. Task tracing is unchanged when repair
tracing is disabled. When enabled, the
PandaProbe SDK records repair completions under
`repair-<task-session>-<notice-id>` with repair-role metadata; exact-session
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
