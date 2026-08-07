"""Managed repair in one task session, fully offline and credential-free."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _offline_support import (
    GUIDANCE,
    DeveloperTaskAgent,
    OfflineCli,
    OfflineRepairCompletion,
)

from pandaprobe_harness import Harness, HarnessConfig


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pandaprobe-repair-") as directory:
        session_id = "offline-task"
        cli = OfflineCli()
        repair = OfflineRepairCompletion()
        harness = Harness.create(
            HarnessConfig(
                harness_root=Path(directory) / "workspace",
                repair_model="offline/fake-wrapper",
                gate_window=1,
                poll_interval_s=0,
                eval_retry_backoff_s=0,
                health_check=False,
            ),
            cli=cli,
            _repair_completion=repair,
        )
        task_agent = DeveloperTaskAgent()

        first = await task_agent.run_turn(
            harness.system_context(session_id, task_hint="payment"), harness.task_tools
        )
        for _ in range(2):
            cli.script_trace(
                session_id,
                task_completion=0.2,
                coherence=0.2,
                tool_correctness=0.1,
                argument_correctness=0.1,
            )
        harness.on_turn_end(
            {"session_id": session_id, "turn_index": 1, "end_state": first}
        )
        settlement = await harness.settle(session_id)
        assert settlement.repair is not None
        assert settlement.repair.status == "candidate_added"

        second_context = harness.system_context(session_id, task_hint="payment")
        second = await task_agent.run_turn(second_context, harness.task_tools)
        assert GUIDANCE not in second_context
        assert second["action"] == "check_status_then_charge"
        assert task_agent.actions == ["charge_twice", "check_status_then_charge"]

        print("offline managed repair: PASS")
        print(f"repair session: {settlement.repair.repair_session_id}")
        print(f"candidate: {settlement.repair.candidate_rule_ids[0]}")


if __name__ == "__main__":
    asyncio.run(main())
