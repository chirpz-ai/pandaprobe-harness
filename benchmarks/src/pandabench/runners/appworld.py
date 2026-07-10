"""AppWorld runner — we own the loop; the environment lives behind HTTP.

One task-trial = one harness session. The agent's single tool is ``execute``
(runs Python where AppWorld's ``apis`` object is preloaded); the benchmark-native
verdict comes from ``/evaluate`` (goal-completion + database-state tests,
including no-collateral-damage). See ``appworld_env`` for the isolation rationale.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..agents.harness_wiring import HarnessWiring
from ..agents.loop import run_agent_loop
from ..providers.litellm_client import ChatClient
from ..providers.models import ResolvedModel
from .appworld_env import AppWorldEnv, AppWorldServer, make_env
from .base import TaskOutcome

logger = logging.getLogger("pandabench.appworld")

__all__ = ["AppWorldRunner", "build_appworld_runner"]

APPWORLD_SYSTEM = """\
You are a capable AI assistant completing a task on behalf of your supervisor by \
writing and running Python code against a set of digital-service APIs.

Environment:
- You have ONE tool, `execute`, which runs Python in a persistent shell where an \
`apis` object is preloaded. Print results you need to see; only stdout is returned.
- Discover APIs with `apis.api_docs.show_app_descriptions()`, \
`apis.api_docs.show_api_descriptions(app_name=...)`, and \
`apis.api_docs.show_api_doc(app_name=..., api_name=...)`.
- Authenticate where needed using the supervisor's credentials (retrievable via \
the `supervisor` app), and call APIs with the exact parameters they document.

Rules:
- Take the task step by step: inspect the relevant API docs, then act.
- Make ONLY the changes the task requires — never delete or modify unrelated data.
- When the task is fully complete, stop calling tools and reply with a short \
confirmation of what you did.

Available apps and APIs:
{api_docs}
"""

_EXECUTE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Execute Python code in the AppWorld shell (the `apis` object is "
            "preloaded). Returns captured stdout, or a traceback string on error."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to run."}},
            "required": ["code"],
        },
    },
}


class AppWorldRunner:
    """Drives AppWorld tasks through the shared agent loop."""

    name = "appworld"

    def __init__(
        self,
        env: AppWorldEnv,
        *,
        server: AppWorldServer | None = None,
        experiment_name: str = "pandabench",
    ) -> None:
        self._env = env
        self._server = server
        self._experiment = experiment_name

    def list_tasks(self, dataset: str) -> list[str]:
        return self._env.list_task_ids(dataset or "dev")

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
        try:
            info = await asyncio.to_thread(
                self._env.initialize, task_id, experiment_name=self._experiment
            )
            api_docs = await asyncio.to_thread(self._env.api_docs)
        except Exception as exc:  # noqa: BLE001 - env failure is a trial error, not a crash
            logger.warning("appworld init failed for %s: %s", task_id, exc)
            return _errored(str(exc), time.monotonic() - start)

        system_prompt = APPWORLD_SYSTEM.format(api_docs=api_docs)
        if preamble is not None:  # replay: inject the harness rules string directly
            system_prompt = preamble + "\n\n" + system_prompt

        user_msg = _task_message(info)

        async def executor(name: str, args: dict[str, Any]) -> Any:
            if name != "execute":
                return {"error": f"unknown tool {name!r}"}
            return await asyncio.to_thread(self._env.execute, task_id, str(args.get("code", "")))

        result = await run_agent_loop(
            client=client, model=model, session_id=session_id,
            system_prompt=system_prompt, tools=[_EXECUTE_TOOL], tool_executor=executor,
            initial_messages=[{"role": "user", "content": user_msg}],
            max_turns=max_turns, wiring=wiring, task_hint=info.instruction,
        )

        try:
            verdict = await asyncio.to_thread(self._env.evaluate, task_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("appworld evaluate failed for %s: %s", task_id, exc)
            return TaskOutcome(
                passed=False, native_metrics={"eval_error": str(exc)}, turns=result.turns,
                wall_time_s=time.monotonic() - start, usage=result.usage,
                error=result.error or f"evaluate: {exc}",
            )
        finally:
            await asyncio.to_thread(self._env.close, task_id)

        pass_ratio = verdict.num_passes / verdict.num_tests if verdict.num_tests else 0.0
        return TaskOutcome(
            passed=verdict.success,
            native_metrics={
                "success": verdict.success,
                "num_tests": verdict.num_tests,
                "num_passes": verdict.num_passes,
                "pass_ratio": pass_ratio,
                "difficulty": verdict.difficulty,
                "stopped_reason": result.stopped_reason,
            },
            turns=result.turns,
            wall_time_s=time.monotonic() - start,
            usage=result.usage,
            error=result.error,
        )

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.stop()


def _task_message(info: Any) -> str:
    sup = info.supervisor or {}
    name = f"{sup.get('first_name', '')} {sup.get('last_name', '')}".strip() or "your supervisor"
    lines = [f"Task from {name}:", "", info.instruction]
    if info.datetime:
        lines += ["", f"Current date/time: {info.datetime}"]
    if sup:
        lines += ["", f"Supervisor details: {sup}"]
    return "\n".join(lines)


def _errored(msg: str, elapsed: float) -> TaskOutcome:
    from ..providers.litellm_client import Usage

    return TaskOutcome(
        passed=False, native_metrics={"init_error": msg}, turns=0,
        wall_time_s=elapsed, usage=Usage(), error=msg,
    )


def build_appworld_runner(*, dry_run: bool) -> AppWorldRunner:
    env, server, _root = make_env(dry_run=dry_run)
    return AppWorldRunner(env, server=server)
