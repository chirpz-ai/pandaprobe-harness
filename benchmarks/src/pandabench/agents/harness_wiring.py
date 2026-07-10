"""Arm-B wiring: the only thing that differs between the two arms.

An arm-B trial passes a :class:`HarnessWiring` into the shared loop; arm A
passes ``None``. The wiring supplies (1) the per-turn system preamble
(``system_context`` with task-conditioned rule retrieval, or a fixed override
during replay), (2) the 14 harness tools as OpenAI function-tool JSON, (3) a
dispatcher for ``harness_*`` tool calls, and (4) the replayable ``end_state``
the runner hands to ``on_turn_end`` so breaching sessions become eval cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandaprobe_harness import Harness

__all__ = ["HarnessWiring", "specs_to_openai"]


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
    ) -> None:
        self.harness = harness
        self.benchmark = benchmark
        self.task_id = task_id
        self.capture = capture
        self.replay_descriptor = replay_descriptor
        self.preamble_override = preamble_override
        # Cache the tool JSON once; the spec set is stable for a harness.
        self._tools = specs_to_openai(harness.toolset.specs())

    def system_preamble(self, task_hint: str) -> str:
        """The preamble to prepend to the benchmark system prompt this turn.

        During replay we inject the harness-supplied rules string verbatim (the
        candidate under evaluation is already rendered into it); otherwise we
        recompute ``system_context`` so retrieval reflects current rules/notices.
        """

        if self.preamble_override is not None:
            return self.preamble_override
        return self.harness.system_context(task_hint=task_hint)

    def harness_tools(self) -> list[dict[str, Any]]:
        return self._tools

    def is_harness_tool(self, name: str) -> bool:
        return name.startswith("harness_")

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Route a ``harness_*`` tool call to the harness toolset."""

        return await self.harness.toolset.call(name, args)

    def end_state(self) -> dict[str, Any]:
        """The ``on_turn_end`` payload's ``end_state``.

        Non-empty only when capturing (learning phase): a non-empty end_state is
        REQUIRED for eval-case capture — the harness stashes this whole dict as
        ``EvalCase.replay_input``. In the frozen eval phase it is ``{}``.
        """

        if not self.capture:
            return {}
        return {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "replay": self.replay_descriptor,
        }
