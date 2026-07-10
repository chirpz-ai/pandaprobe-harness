"""Harbor custom agent for Terminal-Bench 2.x.

Harbor instantiates this class from ``-a pandabench.adapters.harbor_agent:PandaBenchAgent``
and runs it IN-PROCESS on the host; the agent drives the sandboxed container
purely through ``environment.exec``. That lets us reuse the shared pandabench
loop + harness verbatim — the bash tool is just ``environment.exec``.

Per-run config arrives via Harbor's ``--agent-kwarg`` (typed) and ``--agent-env``:
  --ak arm=harness --ak seed=1 --ak model_key=claude-sonnet-4-6 \
  --ak backend=vertex_ai --ak capture=true --ak harness_root=/abs/path
The harness workspace (``harness_root``) is shared across attempts of a
(model x arm x seed) run so learning accumulates; run Harbor with ``-n 1`` for
the arm-B learning pass to keep workspace writes serial.

VERIFICATION: implemented against harbor 0.18.0's ``harbor.agents.base.BaseAgent``
(4 abstract methods) and ``BaseEnvironment.exec -> ExecResult``. A live run needs
Docker + Harbor's env to also contain ``pandabench`` (see IMPLEMENTATION_NOTES).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Harbor is only importable inside Harbor's own environment at run time; the rest
# of the pandabench suite never imports this module, so a top-level import is safe.
from harbor.agents.base import BaseAgent

from ..agents.harness_wiring import HarnessWiring
from ..agents.loop import run_agent_loop
from ..harness_glue import (
    build_harness,
    build_harness_config,
    make_session_id,
    project_name_for,
)
from ..providers.litellm_client import LiteLLMClient
from ..providers.models import load_registry
from ..providers.tracing import PandaTracer

logger = logging.getLogger("pandabench.harbor")

_CONFIGS = Path(__file__).resolve().parents[3] / "configs"

TB_SYSTEM = """\
You are an expert software engineer working directly in a Linux terminal to \
complete the task below. You have one tool, `bash`, which runs a shell command \
in the task's terminal and returns its stdout, stderr, and exit code. Work step \
by step: inspect the environment, make changes, and verify. When the task is \
fully complete, stop calling tools and briefly state what you did.
"""

_BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the task terminal. Returns stdout/stderr/exit code.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command."}},
            "required": ["command"],
        },
    },
}


class PandaBenchAgent(BaseAgent):  # type: ignore[misc]
    """Runs the shared pandabench loop against a Harbor sandbox."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        *args: Any,
        arm: str = "baseline",
        seed: int = 0,
        model_key: str | None = None,
        backend: str | None = None,
        capture: bool = False,
        harness_root: str | None = None,
        max_turns: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, *args, model_name=model_name, logger=logger, **kwargs)
        self._arm = arm
        self._seed = seed
        self._capture = capture
        self._max_turns = max_turns
        registry = load_registry(_CONFIGS / "models.yaml")
        self._model = registry.resolve(
            model_key or model_name or "gemini-2.5-flash", backend=backend
        )
        tracer = PandaTracer.from_env() if arm == "harness" else PandaTracer.disabled()
        self._client = LiteLLMClient(tracer=tracer)
        self._harness = None
        if arm == "harness" and harness_root:
            import os

            os.environ["PANDAPROBE_PROJECT_NAME"] = project_name_for("terminal_bench", arm)
            phase = "learning" if capture else "eval"
            cfg = build_harness_config(
                harness_root=Path(harness_root), phase=phase, study=_load_study(),
                benchmark="terminal_bench",
            )
            self._harness = build_harness(cfg=cfg)

    @staticmethod
    def name() -> str:
        return "pandabench"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        session_id = self.session_id or make_session_id(
            benchmark="terminal_bench", task_id="tb", arm=self._arm,
            model_key=self._model.key, seed=self._seed, trial=0,
        )

        async def executor(name: str, args: dict[str, Any]) -> Any:
            if name != "bash":
                return {"error": f"unknown tool {name!r}"}
            result = await environment.exec(str(args.get("command", "")))
            return {
                "stdout": getattr(result, "stdout", "") or "",
                "stderr": getattr(result, "stderr", "") or "",
                "return_code": getattr(result, "return_code", None),
            }

        wiring: HarnessWiring | None = None
        if self._harness is not None:
            descriptor = {"benchmark": "terminal_bench", "task_id": session_id,
                          "arm": self._arm, "model_key": self._model.key, "seed": self._seed}
            wiring = HarnessWiring(
                harness=self._harness, benchmark="terminal_bench", task_id=session_id,
                capture=self._capture, replay_descriptor=descriptor,
            )

        result = await run_agent_loop(
            client=self._client, model=self._model, session_id=session_id,
            system_prompt=TB_SYSTEM, tools=[_BASH_TOOL], tool_executor=executor,
            initial_messages=[{"role": "user", "content": instruction}],
            max_turns=self._max_turns, wiring=wiring, task_hint=instruction,
        )

        if self._harness is not None and wiring is not None:
            self._harness.on_turn_end(
                {"session_id": session_id, "turn_index": max(result.turns, 1),
                 "end_state": wiring.end_state()}
            )
            await self._harness.refresh(session_id)
            await self._harness.drain_validation()

        # Report usage/cost back to Harbor (best-effort; fields are optional).
        try:
            context.n_input_tokens = result.usage.input_tokens
            context.n_output_tokens = result.usage.output_tokens
            context.cost_usd = result.usage.cost_usd
            context.metadata = {"arm": self._arm, "seed": self._seed,
                                "stopped_reason": result.stopped_reason}
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not populate Harbor context: %s", exc)


def _load_study() -> Any:
    from ..config import load_study

    return load_study(_CONFIGS / "study.yaml")
