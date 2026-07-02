"""CrewAI + PandaProbe Harness — a documented integration sketch (sandboxed pattern).

Requires the ``crewai`` extra and real credentials (an authenticated
``pandaprobe`` CLI plus your model provider's API key):

    pip install 'pandaprobe-harness[crewai]'
    python examples/crewai_agent.py

The wiring, in order:

1. ``Harness.for_crewai(...)`` provisions the workspace and patches
   ``Crew.kickoff`` — a completed crew run fires ``hook.on_turn_end``
   (one kickoff == one evaluated agent turn).
2. Instead of native function tools, this example shows the SANDBOXED
   delivery channel: the agent gets one restricted shell tool
   (``harness.shell``, allow-listed binaries only, scoped credentials) and
   pulls its diagnostics through the companion CLI, e.g.
   ``pandaprobe-harness-agent harness_mailbox_list``. The companion CLI
   resolves the same workspace from the ``HARNESS_*`` environment.
3. The agent's backstory embeds ``harness.system_context()`` (rules + pull
   protocol + mailbox banner), rebuilt each turn so new notices surface.

For the fully offline, credential-free version of this loop, see
``examples/offline_self_heal.py``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys

from pandaprobe_harness import Harness

try:
    from crewai import Agent, Crew, Task
    from crewai.tools import tool
except ImportError:
    sys.exit("missing extra — install with: pip install 'pandaprobe-harness[crewai]'")

SESSION_ID = "s-crewai-demo"
BASE_BACKSTORY = "You are a payments support analyst. Use your tools carefully."


async def main() -> None:
    # One factory call: workspace + hook + toolset + the Crew.kickoff patch.
    harness = Harness.for_crewai(session_id=SESSION_ID)
    # The companion CLI (spawned inside the sandbox) reads HARNESS_* env vars;
    # point it at this harness's workspace.
    os.environ["HARNESS_ROOT"] = str(harness.config.harness_root)
    shell = harness.shell  # RestrictedShellTool: allow-list, no shell=True

    @tool("sandbox_shell")
    def sandbox_shell(command: str) -> str:
        """Run a restricted diagnostic command, e.g.
        'pandaprobe-harness-agent harness_mailbox_list' or
        'pandaprobe evals scores get <trace-id> --target trace'."""
        # CrewAI tools are sync and may run on the event-loop thread; execute
        # the async restricted shell on a private worker-thread event loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(asyncio.run, shell(command)).result()
        return result.stdout if result.ok else f"exit={result.exit_code}: {result.stderr}"

    for request in (
        "Charge customer 42 the monthly fee.",
        "Now charge customer 43 as well.",
    ):
        # Backstory rebuilt each turn = per-turn re-read of the harness system
        # context, so the '⚠ HARNESS' mailbox banner appears right after a
        # breach and the pull protocol drives the agent through the mailbox.
        analyst = Agent(
            role="payments support analyst",
            goal="Resolve the request while honoring your operating rules.",
            backstory=harness.system_context() + "\n" + BASE_BACKSTORY,
            tools=[sandbox_shell],
        )
        task = Task(description=request, expected_output="a short resolution summary")
        crew = Crew(agents=[analyst], tasks=[task])
        # kickoff is synchronous; running it inside asyncio.run(main()) lets
        # the patched kickoff schedule this turn's evaluation on this loop.
        print(crew.kickoff())
        # Optional: join the in-flight evaluation (bounded by drain_timeout_s)
        # so this simple loop observes each turn's outcome before continuing.
        await harness.refresh(SESSION_ID)


if __name__ == "__main__":
    asyncio.run(main())
