# PandaProbe Harness

**Self-healing for AI agents.** The harness wraps any
[PandaProbe](https://github.com/chirpz-ai/pandaprobe)-instrumented agent in an
operational envelope that evaluates every turn, alerts the agent when quality
degrades, and lets it diagnose its own failures and write — and *prove* — its
own operating rules. Fully automatic, no human in the healing loop.

[![PyPI](https://img.shields.io/pypi/v/pandaprobe-harness)](https://pypi.org/project/pandaprobe-harness/)
[![CI](https://github.com/chirpz-ai/pandaprobe-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/chirpz-ai/pandaprobe-harness/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://pypi.org/project/pandaprobe-harness/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

📚 **[Documentation](https://docs.pandaprobe.com/harness/get-started/quickstart)** ·
📦 **[PyPI](https://pypi.org/project/pandaprobe-harness/)** ·
💬 **[Discussions](https://github.com/chirpz-ai/pandaprobe-harness/discussions)**

## How it works

1. **Evaluate** — after each turn, the harness scores the session on the
   PandaProbe platform (`agent_reliability`, `agent_consistency`) in a
   detached task that never blocks your agent.
2. **Notice** — breaches and declining trends post a structured *diagnostic
   notice* to a workspace mailbox. Nothing is ever injected into the agent's
   conversation.
3. **Heal** — guided by a standing protocol in its system prompt, the agent
   pulls the notice, inspects its own flagged traces, and records a mitigation
   rule.
4. **Validate** — the rule enters as a *candidate*: the harness replays the
   captured failure (or watches the next live sessions) and promotes it only
   when it demonstrably helps. Validated rules re-enter the prompt on every
   future run; a replayable eval-set guards old wins against regressions.

The core has **zero runtime dependencies** and reaches the platform exclusively
through the `pandaprobe` CLI.

## Installation

```bash
pip install pandaprobe-harness
# framework adapters are optional extras:
pip install "pandaprobe-harness[langgraph]"     # [langchain] [deepagents] [crewai]
                                                # [claude-agent-sdk] [openai-agents] [all]
```

You'll also need the [`pandaprobe` CLI](https://docs.pandaprobe.com/introduction/cli)
installed and authenticated, and an agent traced with the
[PandaProbe SDK](https://docs.pandaprobe.com/tracing/get-started/quickstart).

## Quickstart

```python
from pandaprobe_harness import Harness
from pandaprobe_harness.agent_tools.native import as_anthropic_tools

harness = Harness.create()                              # provisions the workspace

system_prompt = harness.system_context() + MY_PROMPT    # rules + protocol + banner
specs, dispatch = as_anthropic_tools(harness.toolset)   # the 14 self-diagnostic tools
tools = my_tools + specs

async def one_turn(session_id: str, user_input: str) -> str:
    async with harness.turn(session_id):                # evaluates on exit
        return await my_agent_step(system_prompt, tools, user_input)
```

Using a framework? `Harness.for_langgraph()`, `for_langchain()`,
`for_deepagents()`, `for_crewai()`, `for_claude_agent_sdk()`, and
`for_openai_agents()` wire turn detection for you.

➡ Full guides, concepts, and the configuration reference live in the
**[documentation](https://docs.pandaprobe.com/harness/get-started/quickstart)**.

## Try it offline

The `examples/` directory ships fully-offline, credential-free demos:

```bash
make example                                        # the pull loop, end to end
uv run python examples/closed_loop_self_heal.py     # candidate → validate → promote → regression
uv run python examples/calibration_demo.py          # threshold calibration
```

## Operator CLIs

| Command | Purpose |
| --- | --- |
| `pandaprobe-harness-agent` | The agent-facing toolset for sandboxed shells. |
| `pandaprobe-harness-eval` | Replay the eval-set against the current rules — the regression guard. |
| `pandaprobe-harness-calibrate` | Measure and tune the breach thresholds, with or without labels. |

## Benchmarks

An A/B study measuring the harness's effect on agent reliability across
AppWorld, Terminal-Bench (via Harbor), and τ²-bench lives in
[`benchmarks/`](benchmarks/) — a self-contained uv project that installs the
released harness from PyPI. Run it from the repo root with `make bench-setup`,
`make bench-smoke`, and `make bench-report` (see
[`benchmarks/README.md`](benchmarks/README.md)).

## Development

```bash
make install         # uv sync
make test            # full offline suite — no network, no real CLI
make lint typecheck  # ruff + mypy --strict
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the project invariants and PR
process, and [CHANGELOG.md](CHANGELOG.md) for release history.

## License

[MIT](LICENSE) © [Chirpz AI](https://chirpz.ai)
