"""Arm-B task-agent wiring for package-owned managed repair.

An arm-B trial passes a :class:`HarnessWiring` into the shared loop; arm A
passes ``None``. The wiring supplies a stable capability-only preamble,
read-only task tools, replay metadata, and the per-turn settlement barrier. The
package-owned repair agent receives notices and administrative capabilities;
the benchmark task agent never does.

The barrier is what makes healing in-session. v1 hooked the harness once per
task-trial, so a session's trajectory was scored exactly once and the trend
machinery never had enough samples to fire — and any lesson arrived after the
task was already over. Now every turn ends with ``on_turn_end`` + ``settle``, so
the next turn can discover whatever the harness just found through its tools.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from pandaprobe_harness import ReplayContext, RuleScopeHint

if TYPE_CHECKING:
    from pandaprobe_harness import Harness, SettleResult

__all__ = ["AgentWiring", "HarnessWiring", "ReplayRuleWiring", "specs_to_openai"]

logger = logging.getLogger("pandabench.harness")


class AgentWiring(Protocol):
    """Agent-facing surface shared by live learning and frozen evaluation."""

    @property
    def settles_turns(self) -> bool: ...

    def system_preamble(self) -> str: ...

    def harness_tools(self) -> list[dict[str, Any]]: ...

    def is_harness_tool(self, name: str) -> bool: ...

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...

    async def settle_turn(self, turn_index: int) -> Any: ...


def specs_to_openai(specs: Any) -> list[dict[str, Any]]:
    """Convert harness ``ToolSpec`` objects to OpenAI function-tool JSON.

    We roll our own instead of ``as_openai_function_tools`` so the study needs
    no ``[openai-agents]`` extra; dispatch goes through ``harness.task_tools``.
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
        session_id: str,
        flush: Callable[[], None] | None = None,
        settle_each_turn: bool = True,
        rule_scope_hints: tuple[RuleScopeHint, ...] = (),
    ) -> None:
        self.harness = harness
        self.benchmark = benchmark
        self.task_id = task_id
        self.capture = capture
        self.replay_descriptor = replay_descriptor
        self.session_id = session_id
        self._flush = flush
        # A replay must not recurse into the barrier: it re-runs a task purely to
        # be scored, and hooking its turns would spawn evals inside an eval.
        self._settle_each_turn = settle_each_turn
        self._rule_scope_hints = rule_scope_hints
        # Cache the tool JSON once; the spec set is stable for a harness.
        self._tools = specs_to_openai(harness.task_tools.specs())
        self.turns_settled = 0
        # The most recent (turn_index, result), so settling a turn twice returns
        # the first answer instead of re-evaluating. See settle_turn.
        self._settled: tuple[int, SettleResult | None] | None = None

    @property
    def settles_turns(self) -> bool:
        """Whether callers should invoke the live per-turn evaluation barrier."""

        return self._settle_each_turn

    def system_preamble(self) -> str:
        """Stable capability note; rule content remains behind read-only tools."""

        return self.harness.system_context(self.session_id)

    def set_rule_scope_hints(self, hints: tuple[RuleScopeHint, ...]) -> None:
        """Attach safe semantic metadata discovered during task initialization."""

        self._rule_scope_hints = hints

    def harness_tools(self) -> list[dict[str, Any]]:
        return self._tools

    def is_harness_tool(self, name: str) -> bool:
        # Capability enforcement lives in TaskToolset.call, so hallucinated
        # administrative names are routed there and rejected safely.
        return name.startswith("harness_")

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Route a ``harness_*`` call through the task capability boundary."""

        return await self.harness.task_tools.call(name, args)

    async def settle_turn(self, turn_index: int) -> SettleResult | None:
        """End one turn and wait for the harness to finish diagnosing it.

        Returns the ``SettleResult`` (or ``None`` if the barrier is off or errored)
        so the caller can read this turn's report. That return value matters: the
        barrier *consumes* the turn's pending evaluation, so a caller that settled
        again to get the report would get an empty one — which is how v1 ended up
        with null score telemetry on every recorded row.

        **Idempotent per turn.** Settling the same index twice returns the first
        answer rather than firing a second evaluation. Two callers legitimately
        settle turns — the agent loop for turns that continue, the runner for the
        trial's last one — and their hand-off is easy to get wrong (at the turn cap
        the loop's last turn *is* the trial's last). A repeat would find no new
        traces, so its report would carry none of the tier scores, and the caller
        recording telemetry from it would write that emptiness over the real
        diagnosis. It would also spend a second platform eval run for nothing.

        Ordering matters too: traces are flushed *before* ``on_turn_end`` so the
        trace listing that follows can already see this turn's trace. Never raises
        — a harness hiccup must degrade the study's telemetry, not fail the trial.
        """

        if not self._settle_each_turn:
            return None
        if self._settled is not None and self._settled[0] == turn_index:
            return self._settled[1]
        # Claimed before awaiting, so a failed settle still counts as attempted
        # rather than inviting a duplicate from the next caller.
        self._settled = (turn_index, None)
        try:
            if self._flush is not None:
                self._flush()  # push buffered traces so this turn is scorable
            self.harness.on_turn_end(
                {
                    "session_id": self.session_id,
                    "turn_index": turn_index,
                    "end_state": self.end_state(),
                    "rule_scope_hints": [
                        hint.to_json() for hint in self._rule_scope_hints
                    ],
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
            self._settled = (turn_index, result)
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
        verifier. The live wiring therefore always carries the task id; capture is
        separately gated by ``capture_eval_cases``. Frozen eval does not call this
        method because it has no live harness or outcome verifier.
        """

        state: dict[str, Any] = {"benchmark": self.benchmark, "task_id": self.task_id}
        if self.capture:
            state["replay"] = self.replay_descriptor
        return state


class ReplayRuleWiring:
    """Read-only on-demand rule access for a detached validation replay."""

    def __init__(self, context: ReplayContext) -> None:
        self._context = context
        self._tools = specs_to_openai(context.task_tools.specs())

    @property
    def settles_turns(self) -> bool:
        return False

    def system_preamble(self) -> str:
        return str(self._context)

    def harness_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    def is_harness_tool(self, name: str) -> bool:
        return name.startswith("harness_")

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return await self._context.task_tools.call(name, args)

    async def settle_turn(self, turn_index: int) -> None:
        del turn_index
