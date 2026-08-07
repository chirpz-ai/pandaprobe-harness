"""CrewAI task ownership with PandaProbe-managed repair.

Requires the CrewAI extra, provider credentials, PandaProbe CLI auth, and an
explicit ``HARNESS_REPAIR_MODEL``. The task agent receives a capability note,
not rule bodies, administrative shell, or mailbox capabilities.
"""

from __future__ import annotations

import asyncio
import sys

from pandaprobe_harness import Harness, HarnessConfig

try:
    from crewai import Agent, Crew, Task
except ImportError:
    sys.exit("missing extra — install with: pip install 'pandaprobe-harness[crewai]'")

SESSION_ID = "s-crewai-demo"
BASE_BACKSTORY = "You are a payments support analyst. Use your domain tools carefully."


async def main() -> None:
    harness = Harness.for_crewai(
        session_id=SESSION_ID, config=HarnessConfig.from_env()
    )
    for request in (
        "Charge customer 42 the monthly fee.",
        "Now charge customer 43 as well.",
    ):
        analyst = Agent(
            role="payments support analyst",
            goal="Resolve the request while honoring learned guidance.",
            backstory=harness.system_context(SESSION_ID, task_hint=request)
            + "\n"
            + BASE_BACKSTORY,
            tools=[],  # add only developer-owned domain tools here
        )
        task = Task(description=request, expected_output="a short resolution summary")
        crew = Crew(agents=[analyst], tasks=[task])
        print(crew.kickoff())
        await harness.settle(SESSION_ID)


if __name__ == "__main__":
    asyncio.run(main())
