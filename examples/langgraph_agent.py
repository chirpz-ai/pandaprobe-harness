"""LangGraph + PandaProbe Harness — a documented integration sketch.

Requires the ``langgraph`` extra and real credentials (an authenticated
``pandaprobe`` CLI plus your model provider's API key):

    pip install 'pandaprobe-harness[langgraph]'
    python examples/langgraph_agent.py

The wiring, in order:

1. ``Harness.for_langgraph(...)`` provisions the workspace and registers the
   LangGraph adapter (turn detection via a LangChain callback).
2. ``harness.adapter.make_callback()`` returns the handler; pass it in each
   invoke's ``config`` so the hook fires on every root chain end (one turn).
3. ``harness.system_context()`` (rules + pull protocol + mailbox banner) is
   prepended to the system prompt — and re-read every turn, so a freshly
   posted notice surfaces as the banner on the very next turn.
4. ``as_langchain_tools(harness.toolset)`` hands the agent its own
   self-diagnostic tools (mailbox, trace inspection, rules, journal).

For the fully offline, credential-free version of this loop, see
``examples/offline_self_heal.py``.
"""

from __future__ import annotations

import asyncio
import sys

from pandaprobe_harness import Harness
from pandaprobe_harness.agent_tools.native import as_langchain_tools

try:
    from langgraph.prebuilt import create_react_agent
except ImportError:
    sys.exit("missing extra — install with: pip install 'pandaprobe-harness[langgraph]'")

SESSION_ID = "s-langgraph-demo"
BASE_PROMPT = "You are a payments support agent. Use your tools carefully."


async def main() -> None:
    # One factory call: workspace + hook + toolset + the LangGraph adapter.
    harness = Harness.for_langgraph(session_id=SESSION_ID)

    # Turn detection: this LangChain callback fires `hook.on_turn_end` on the
    # ROOT chain end — one `ainvoke` == one evaluated agent turn.
    handler = harness.adapter.make_callback()

    # The agent's self-diagnostic tools sit right next to your domain tools.
    tools = as_langchain_tools(harness.toolset)  # + your own tools

    # A callable prompt re-reads harness.system_context() EVERY turn: after the
    # hook posts a notice, the '⚠ HARNESS' mailbox banner appears here and the
    # standing pull protocol tells the agent to work the mailbox first.
    def prompt(state: dict) -> list:
        system = harness.system_context() + "\n" + BASE_PROMPT
        return [{"role": "system", "content": system}, *state["messages"]]

    graph = create_react_agent("openai:gpt-4o-mini", tools, prompt=prompt)

    for user_input in (
        "Charge customer 42 the monthly fee.",
        "Now charge customer 43 as well.",
    ):
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"callbacks": [handler], "configurable": {"thread_id": SESSION_ID}},
        )
        print(result["messages"][-1].content)
        # Optional: join the in-flight evaluation (bounded by drain_timeout_s)
        # so this simple loop observes each turn's outcome before continuing.
        await harness.refresh(SESSION_ID)


if __name__ == "__main__":
    asyncio.run(main())
