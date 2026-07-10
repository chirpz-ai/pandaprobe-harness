"""A generic dependency-free SingleTaskRunner for ``--dry-run`` smoke.

Lets every benchmark exercise the full run -> records -> report pipeline with
no external harness, server, or API — the deterministic acceptance gate. The
real integrations (AppWorld HTTP, Harbor subprocess, tau2 orchestrator) replace
this on non-dry-run paths.
"""

from __future__ import annotations

import time
from typing import Any

from ..agents.harness_wiring import HarnessWiring
from ..agents.loop import run_agent_loop
from ..providers.litellm_client import ChatClient
from ..providers.models import ResolvedModel
from .base import TaskOutcome

__all__ = ["MockTaskRunner"]

_NOOP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "A placeholder tool used only by the dry-run mock benchmark.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class MockTaskRunner:
    """Drives the shared loop against a canned task; no external dependencies."""

    def __init__(self, name: str, *, tasks: int = 4) -> None:
        self.name = name
        self._tasks = tasks

    def list_tasks(self, dataset: str) -> list[str]:
        return [f"{self.name}_{i}" for i in range(1, self._tasks + 1)]

    async def run_once(
        self,
        *,
        task_id: str,
        session_id: str,
        model: ResolvedModel,
        client: ChatClient,
        max_turns: int,
        wiring: HarnessWiring | None,
        preamble: str | None = None,
    ) -> TaskOutcome:
        start = time.monotonic()
        system_prompt = f"[mock:{self.name}] complete the task and reply when done."
        if preamble is not None:
            system_prompt = preamble + "\n\n" + system_prompt

        async def executor(name: str, args: dict[str, Any]) -> Any:
            return {"ok": True}

        result = await run_agent_loop(
            client=client, model=model, session_id=session_id,
            system_prompt=system_prompt, tools=[_NOOP_TOOL], tool_executor=executor,
            initial_messages=[{"role": "user", "content": f"do {task_id}"}],
            max_turns=max_turns, wiring=wiring, task_hint=f"do {task_id}",
        )
        # Deterministic pseudo-outcome: even-indexed tasks "pass".
        idx = int(task_id.rsplit("_", 1)[-1]) if task_id.rsplit("_", 1)[-1].isdigit() else 0
        passed = idx % 2 == 0
        return TaskOutcome(
            passed=passed,
            native_metrics={"mock": True, "stopped_reason": result.stopped_reason},
            turns=result.turns,
            wall_time_s=time.monotonic() - start,
            usage=result.usage,
            error=result.error,
        )

    async def aclose(self) -> None:
        pass
