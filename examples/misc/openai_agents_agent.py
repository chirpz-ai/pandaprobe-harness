"""OpenAI Agents SDK + PandaProbe Harness — a documented integration sketch.

Requires the ``openai-agents`` extra and real credentials (an authenticated
``pandaprobe`` CLI plus ``OPENAI_API_KEY``):

    pip install 'pandaprobe-harness[openai-agents]'
    python examples/misc/openai_agents_agent.py

The wiring, in order:

1. ``Harness.for_openai_agents(...)`` provisions the workspace and installs a
   ``TracingProcessor`` — one completed ``Runner.run`` trace fires
   ``hook.on_turn_end`` (one run == one evaluated agent turn).
2. ``harness.system_context(session_id)`` supplies a stable capability note.
3. ``as_openai_function_tools(harness.task_tools)`` registers read-only rules.

For the fully offline, credential-free version of this loop, see
``examples/misc/offline_self_heal.py``.
"""

from __future__ import annotations

import asyncio
import sys

from pandaprobe_harness import Harness, HarnessConfig
from pandaprobe_harness.agent_tools.native import as_openai_function_tools

try:
    from agents import Agent, Runner
except ImportError:
    sys.exit("missing extra — install with: pip install 'pandaprobe-harness[openai-agents]'")

SESSION_ID = "s-openai-agents-demo"
BASE_INSTRUCTIONS = "You are a payments support agent. Use your tools carefully."


async def main() -> None:
    harness = Harness.for_openai_agents(
        session_id=SESSION_ID, config=HarnessConfig.from_env()
    )

    tools = as_openai_function_tools(harness.task_tools)  # + your domain tools

    for user_input in (
        "Charge customer 42 the monthly fee.",
        "Now charge customer 43 as well.",
    ):
        agent = Agent(
            name="support-agent",
            instructions=harness.system_context(SESSION_ID) + "\n" + BASE_INSTRUCTIONS,
            tools=tools,
        )
        result = await Runner.run(agent, user_input)
        print(result.final_output)
        await harness.settle(SESSION_ID)


if __name__ == "__main__":
    asyncio.run(main())
