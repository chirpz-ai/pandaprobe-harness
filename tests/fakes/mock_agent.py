"""A scripted mock agent that drives the raw loop and self-heals.

Behaviour:

* Turns with no alert and not yet healed: emit an *identical repeated* tool call
  — the infinite-repetition failure the harness should catch.
* On receiving a SYSTEM ALERT: use the restricted shell tool to read
  ``latest_eval.json`` and query the CLI, then append a permanent mitigation
  rule to ``harness_rules.md`` (self-heal). Emit a ``diagnose`` turn.
* After healing: emit a distinct, corrected action.

Everything the agent does is recorded for assertions.
"""

from __future__ import annotations

from typing import Any

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, RestrictedShellTool
from pandaprobe_harness.adapters.raw_loop import RawLoopAdapter


class MockLLMAgent:
    def __init__(
        self,
        *,
        session_id: str,
        shell: RestrictedShellTool,
        filesystem: HarnessFilesystem,
        config: HarnessConfig,
    ) -> None:
        self._session_id = session_id
        self._shell = shell
        self._fs = filesystem
        self._config = config

        self.turn_index = 0
        self.healed = False
        self.actions: list[str] = []
        self.shell_commands: list[str] = []

    async def take_turn(self, alerts: list[str]) -> dict[str, Any]:
        self.turn_index += 1

        if alerts and not self.healed:
            await self._diagnose_and_heal()
            action = "diagnose"
        elif self.healed:
            action = "verified_payment_then_charge"  # corrected, non-repeating
        else:
            action = "charge_payment"  # repeated identical call (the failure)

        self.actions.append(action)
        return RawLoopAdapter.make_turn(
            self._session_id, self.turn_index, action=action
        )

    async def _diagnose_and_heal(self) -> None:
        # 1. Read the diagnostic dump from the workspace.
        await self._run_shell(f"cat {self._config.latest_eval_file}")
        # 2. Inspect what went wrong via the PandaProbe CLI.
        await self._run_shell("pandaprobe evals scores get trace-1")
        # 3. Record a permanent mitigation directive.
        self._fs.append_rule(
            "Never call the payment tool twice without first verifying the "
            "transaction status identifier.",
            source="self-heal",
        )
        self.healed = True

    async def _run_shell(self, command: str) -> None:
        self.shell_commands.append(command)
        await self._shell(command)
