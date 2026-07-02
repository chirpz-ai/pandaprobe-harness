"""Claude Agent SDK + PandaProbe Harness — a documented integration sketch.

Requires the ``claude-agent-sdk`` extra and real credentials (an authenticated
``pandaprobe`` CLI plus ``ANTHROPIC_API_KEY``):

    pip install 'pandaprobe-harness[claude-agent-sdk]'
    python examples/claude_agent_sdk_agent.py

The wiring, in order:

1. ``Harness.for_claude_agent_sdk(...)`` provisions the workspace and patches
   ``ClaudeSDKClient.receive_response`` — one completed response stream fires
   ``hook.on_turn_end`` (one query/response == one evaluated agent turn).
2. ``as_anthropic_tools(harness.toolset)`` yields plain Anthropic tool dicts
   (``specs``) plus an async ``dispatch(name, args)`` — every ``tool_use``
   block naming a harness tool is routed through ``dispatch``. Below, the
   specs are registered as in-process MCP tools whose handlers ARE that route.
3. ``harness.system_context()`` (rules + pull protocol + mailbox banner) is
   prepended to the system prompt and re-read each turn.

For the fully offline, credential-free version of this loop, see
``examples/offline_self_heal.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys

from pandaprobe_harness import Harness
from pandaprobe_harness.agent_tools.native import as_anthropic_tools

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        create_sdk_mcp_server,
        tool,
    )
except ImportError:
    sys.exit("missing extra — install with: pip install 'pandaprobe-harness[claude-agent-sdk]'")

SESSION_ID = "s-claude-sdk-demo"
BASE_PROMPT = "You are a payments support agent. Use your tools carefully."


def _register(specs: list[dict], dispatch) -> list:
    """Register each Anthropic tool spec, routing tool_use through dispatch."""

    handlers = []
    for spec in specs:

        @tool(spec["name"], spec["description"], spec["input_schema"])
        async def _handler(args: dict, _name: str = spec["name"]) -> dict:
            result = await dispatch(_name, args)  # -> {"ok": ..., ...} envelope
            return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

        handlers.append(_handler)
    return handlers


async def main() -> None:
    # One factory call: workspace + hook + toolset + the receive_response patch.
    harness = Harness.for_claude_agent_sdk(session_id=SESSION_ID)

    # Anthropic tool dicts + the dispatcher that executes them.
    specs, dispatch = as_anthropic_tools(harness.toolset)
    server = create_sdk_mcp_server(name="harness", tools=_register(specs, dispatch))

    # System prompt = harness context (rules + protocol + mailbox banner) +
    # yours. Re-read per client/turn so a fresh notice surfaces as the banner.
    options = ClaudeAgentOptions(
        system_prompt=harness.system_context() + "\n" + BASE_PROMPT,
        mcp_servers={"harness": server},
        allowed_tools=[f"mcp__harness__{spec['name']}" for spec in specs],
    )

    async with ClaudeSDKClient(options=options) as client:
        for user_input in (
            "Charge customer 42 the monthly fee.",
            "Now charge customer 43 as well.",
        ):
            await client.query(user_input)
            async for message in client.receive_response():  # turn ends when drained
                print(message)
            # Optional: join the in-flight evaluation (bounded by
            # drain_timeout_s) so the loop observes each turn's outcome.
            await harness.refresh(SESSION_ID)


if __name__ == "__main__":
    asyncio.run(main())
