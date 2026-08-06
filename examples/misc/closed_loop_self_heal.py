"""Managed candidate authoring plus replay validation, fully offline."""

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
from pandaprobe_harness.workspace.evalset import EvalCase


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pandaprobe-closed-loop-") as directory:
        session_id = "offline-closed-loop"
        replay_session = "offline-replay"
        cli = OfflineCli()
        repair = OfflineRepairCompletion()

        async def replay(case: EvalCase, guidance: str) -> str:
            assert case.replayable
            assert GUIDANCE in guidance
            return replay_session

        harness = Harness.create(
            HarnessConfig(
                harness_root=Path(directory) / "workspace",
                repair_model="offline/fake-wrapper",
                gate_window=1,
                capture_eval_cases=True,
                poll_interval_s=0,
                eval_retry_backoff_s=0,
                health_check=False,
            ),
            cli=cli,
            replay=replay,
            _repair_completion=repair,
        )
        task_agent = DeveloperTaskAgent()

        first = await task_agent.run_turn(harness.system_context(session_id))
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
        first_settlement = await harness.settle(session_id)
        assert first_settlement.repair is not None
        assert len(harness.rules.candidates()) == 1

        cli.script_trace(
            replay_session,
            task_completion=0.95,
            coherence=0.95,
            tool_correctness=0.95,
            argument_correctness=0.95,
        )
        second = await task_agent.run_turn(harness.system_context(session_id))
        cli.script_trace(
            session_id,
            task_completion=0.9,
            coherence=0.9,
            tool_correctness=0.9,
            argument_correctness=0.9,
        )
        harness.on_turn_end(
            {"session_id": session_id, "turn_index": 2, "end_state": second}
        )
        await harness.settle(session_id)
        await harness.drain_validation()

        (active,) = harness.rules.active()
        assert active.rule == GUIDANCE
        assert harness.rules.candidates() == []
        print("offline closed loop: PASS")
        print(f"promoted rule: {active.id}")


if __name__ == "__main__":
    asyncio.run(main())
