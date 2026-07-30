"""Arm-B wiring: the only thing that differs between the two arms.

An arm-B trial passes a :class:`HarnessWiring` into the shared loop; arm A
passes ``None``. The wiring supplies (1) the per-turn system preamble (the skill
root plus any pending-notice banner, or a fixed override during replay), (2) the
harness tools as OpenAI function-tool JSON, (3) a dispatcher for ``harness_*``
tool calls, (4) the replayable ``end_state`` that turns a breaching session into
an eval case, and (5) the **per-turn barrier**.

The barrier is what makes healing in-session. v1 hooked the harness once per
task-trial, so a session's trajectory was scored exactly once and the trend
machinery never had enough samples to fire — and any lesson arrived after the
task was already over. Now every turn ends with ``on_turn_end`` + ``settle``, so
the next turn's preamble already reflects whatever the harness just found.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandaprobe_harness import Harness, SettleResult

__all__ = ["HarnessWiring", "specs_to_openai"]

logger = logging.getLogger("pandabench.harness")


def specs_to_openai(specs: Any) -> list[dict[str, Any]]:
    """Convert harness ``ToolSpec`` objects to OpenAI function-tool JSON.

    We roll our own instead of ``as_openai_function_tools`` so the study needs
    no ``[openai-agents]`` extra; dispatch goes through ``harness.toolset.call``.
    """

    tools: list[dict[str, Any]] = []
    for spec in specs:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
        )
    return tools


class HarnessWiring:
    """Bundles the arm-B integration surface for one task-trial."""

    def __init__(
        self,
        *,
        harness: Harness,
        benchmark: str,
        task_id: str,
        capture: bool,
        replay_descriptor: dict[str, Any],
        preamble_override: str | None = None,
        session_id: str | None = None,
        flush: Callable[[], None] | None = None,
        settle_each_turn: bool = True,
    ) -> None:
        self.harness = harness
        self.benchmark = benchmark
        self.task_id = task_id
        self.capture = capture
        self.replay_descriptor = replay_descriptor
        self.preamble_override = preamble_override
        self.session_id = session_id
        self._flush = flush
        # A replay must not recurse into the barrier: it re-runs a task purely to
        # be scored, and hooking its turns would spawn evals inside an eval.
        self._settle_each_turn = settle_each_turn and session_id is not None
        # Cache the tool JSON once; the spec set is stable for a harness.
        self._tools = specs_to_openai(harness.toolset.specs())
        self.turns_settled = 0

    def system_preamble(self) -> str:
        """The preamble to prepend to the benchmark system prompt this turn.

        During replay we inject the harness-supplied rules string verbatim (the
        candidate under evaluation is already rendered into it); otherwise we
        recompute ``system_context`` so the References index and the pending-notice
        banner reflect whatever the last turn's barrier just produced.
        """

        if self.preamble_override is not None:
            return self.preamble_override
        return self.harness.system_context()

    def harness_tools(self) -> list[dict[str, Any]]:
        return self._tools

    def is_harness_tool(self, name: str) -> bool:
        return name.startswith("harness_")

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Route a ``harness_*`` tool call to the harness toolset."""

        return await self.harness.toolset.call(name, args)

    async def settle_turn(self, turn_index: int) -> SettleResult | None:
        """End one turn and wait for the harness to finish diagnosing it.

        Returns the ``SettleResult`` (or ``None`` if the barrier is off or errored)
        so the caller can read this turn's report. That return value matters: the
        barrier *consumes* the turn's pending evaluation, so a caller that settled
        again to get the report would get an empty one — which is how v1 ended up
        with null score telemetry on every recorded row.

        Ordering matters too: traces are flushed *before* ``on_turn_end`` so the
        trace listing that follows can already see this turn's trace. Never raises
        — a harness hiccup must degrade the study's telemetry, not fail the trial.
        """

        if not self._settle_each_turn or self.session_id is None:
            return None
        try:
            if self._flush is not None:
                self._flush()  # push buffered traces so this turn is scorable
            self.harness.on_turn_end(
                {
                    "session_id": self.session_id,
                    "turn_index": turn_index,
                    "end_state": self.end_state(),
                }
            )
            result = await self.harness.settle(self.session_id)
            self.turns_settled += 1
            if result.timed_out:
                logger.info(
                    "harness barrier timed out on %s turn %d; work continues detached",
                    self.session_id,
                    turn_index,
                )
            return result
        except Exception as exc:  # noqa: BLE001 - never fail a trial on harness trouble
            logger.warning(
                "harness barrier failed on %s turn %d: %s", self.session_id, turn_index, exc
            )
            return None

    def end_state(self) -> dict[str, Any]:
        """The ``on_turn_end`` payload's ``end_state``.

        Serves two consumers: it is stashed verbatim as ``EvalCase.replay_input``
        when a breach is captured, and it identifies the task to the outcome
        verifier. It therefore carries the task id in **both** phases — capture is
        gated by ``capture_eval_cases`` (off outside the learning phase), so the
        frozen eval phase still gets a verified outcome without capturing cases.
        """

        state: dict[str, Any] = {"benchmark": self.benchmark, "task_id": self.task_id}
        if self.capture:
            state["replay"] = self.replay_descriptor
        return state
