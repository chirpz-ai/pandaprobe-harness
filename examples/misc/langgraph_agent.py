"""LangGraph + PandaProbe Harness — a documented integration sketch.

Requires the ``langgraph`` extra and real credentials (an authenticated
``pandaprobe`` CLI plus your model provider's API key):

    pip install 'pandaprobe-harness[langgraph]'
    python examples/misc/langgraph_agent.py

The wiring, in order:

1. ``Harness.for_langgraph(...)`` provisions the workspace and registers the
   LangGraph adapter (turn detection via a LangChain callback).
2. ``harness.adapter.make_callback()`` returns the handler; pass it in each
   invoke's ``config`` so the hook fires on every root chain end (one turn).
3. ``harness.system_context(session_id)`` supplies bounded learned guidance.
4. ``as_langchain_tools(harness.task_tools)`` adds optional read-only rule tools.

For the fully offline, credential-free version of this loop, see
``examples/misc/offline_self_heal.py``.
"""

from __future__ import annotations

import asyncio
import os
import sys

from pandaprobe_harness import Harness, HarnessConfig
from pandaprobe_harness.agent_tools.native import as_langchain_tools

try:
    from langgraph.prebuilt import create_react_agent
except ImportError:
    sys.exit("missing extra — install with: pip install 'pandaprobe-harness[langgraph]'")

SESSION_ID = "s-langgraph-demo"
BASE_PROMPT = "You are a payments support agent. Use your tools carefully."


async def main() -> None:
    task_model = os.environ.get("TASK_MODEL")
    if not task_model:
        sys.exit("set TASK_MODEL and HARNESS_REPAIR_MODEL explicitly")
    harness = Harness.for_langgraph(
        session_id=SESSION_ID, config=HarnessConfig.from_env()
    )

    # Turn detection: this LangChain callback fires `hook.on_turn_end` on the
    # ROOT chain end — one `ainvoke` == one evaluated agent turn.
    handler = harness.adapter.make_callback()

    tools = as_langchain_tools(harness.task_tools)  # + your own domain tools

    def prompt(state: dict) -> list:
        system = harness.system_context(SESSION_ID) + "\n" + BASE_PROMPT
        return [{"role": "system", "content": system}, *state["messages"]]

    graph = create_react_agent(task_model, tools, prompt=prompt)

    for user_input in (
        "Charge customer 42 the monthly fee.",
        "Now charge customer 43 as well.",
    ):
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"callbacks": [handler], "configurable": {"thread_id": SESSION_ID}},
        )
        print(result["messages"][-1].content)
        await harness.settle(SESSION_ID)


if __name__ == "__main__":
    asyncio.run(main())
