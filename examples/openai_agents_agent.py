"""OpenAI Agents SDK + PandaProbe Harness — a documented integration sketch.

Requires the ``openai-agents`` extra and real credentials (an authenticated
``pandaprobe`` CLI plus ``OPENAI_API_KEY``):

    pip install 'pandaprobe-harness[openai-agents]'
    python examples/openai_agents_agent.py

The wiring, in order:

1. ``Harness.for_openai_agents(...)`` provisions the workspace and installs a
   ``TracingProcessor`` — one completed ``Runner.run`` trace fires
   ``hook.on_turn_end`` (one run == one evaluated agent turn).
2. ``harness.system_context()`` is prepended to the agent instructions and
   re-read each turn, so the mailbox banner surfaces new notices.
3. ``as_openai_function_tools(harness.toolset)`` registers the agent's
   self-diagnostic tools (mailbox, trace inspection, rules, journal) as
   native ``FunctionTool``s.

For the fully offline, credential-free version of this loop, see
``examples/offline_self_heal.py``.
"""

from __future__ import annotations

import asyncio
import sys

from pandaprobe_harness import Harness
from pandaprobe_harness.agent_tools.native import as_openai_function_tools

try:
    from agents import Agent, Runner
except ImportError:
    sys.exit("missing extra — install with: pip install 'pandaprobe-harness[openai-agents]'")

SESSION_ID = "s-openai-agents-demo"
BASE_INSTRUCTIONS = "You are a payments support agent. Use your tools carefully."


async def main() -> None:
    # One factory call: workspace + hook + toolset + the tracing processor.
    harness = Harness.for_openai_agents(session_id=SESSION_ID)

    # Self-diagnostic tools, registered next to your own function tools.
    tools = as_openai_function_tools(harness.toolset)  # + your own tools

    for user_input in (
        "Charge customer 42 the monthly fee.",
        "Now charge customer 43 as well.",
    ):
        # Rebuild the agent each turn so instructions re-read the harness
        # system context: after a breach, the '⚠ HARNESS' mailbox banner shows
        # up here and the standing pull protocol drives the agent to pull the
        # notice, record a mitigation rule, and acknowledge it.
        agent = Agent(
            name="support-agent",
            instructions=harness.system_context() + "\n" + BASE_INSTRUCTIONS,
            tools=tools,
        )
        result = await Runner.run(agent, user_input)
        print(result.final_output)
        # Optional: join the in-flight evaluation (bounded by drain_timeout_s)
        # so this simple loop observes each turn's outcome before continuing.
        await harness.refresh(SESSION_ID)


if __name__ == "__main__":
    asyncio.run(main())
