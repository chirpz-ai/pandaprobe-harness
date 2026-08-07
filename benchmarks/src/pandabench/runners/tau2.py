"""tau2-bench runner — we drive tau2's Orchestrator per (task x trial).

One episode = one harness session. tau2's own ``run_task`` hardcodes the
``LLMAgent(tools, domain_policy, llm, llm_args)`` constructor and so cannot be
handed harness config, which is why we build the pieces and drive
``Orchestrator`` ourselves with :class:`~pandabench.adapters.tau2_agent.PandaBenchTau2Agent`.

Two things about tau2 shape this module:

* ``Orchestrator.run()`` is **synchronous and blocking**, and ``run_once`` is
  awaited from a live event loop — so the episode runs in a worker thread and the
  agent submits its coroutines back to our loop (see the adapter's ``_await``).
* ``Orchestrator.run()`` does **not** grade: it returns a ``SimulationRun`` with
  ``reward_info=None``. Grading is a separate ``evaluate_simulation`` call.

The user simulator stays on tau2's stock ``generate()`` path with a fixed model
(``roles.user_simulator``), so it is identical across arms and cannot bias the
comparison. Its spend is therefore excluded from the trial's usage.

tau2's data tree is not shipped: ``TAU2_DATA_DIR`` must point at a clone's
``data/`` **before the first tau2 import** (tau2 reads it at import time).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from pandaprobe_harness import RuleScopeHint

from ..agents.harness_wiring import AgentWiring
from ..providers.litellm_client import ChatClient, Usage
from ..providers.models import ModelRegistry, ResolvedModel, load_registry
from .base import SingleTaskRunner, TaskOutcome
from .mock import MockTaskRunner

logger = logging.getLogger("pandabench.tau2")

__all__ = ["Tau2Runner", "build_tau2_runner"]

_SUPPORTED_DOMAINS = ("airline", "retail", "telecom")
_DOMAIN_SCOPE_DESCRIPTIONS = {
    "airline": "Airline booking, reservation, passenger, and flight-change workflows.",
    "retail": "Retail order, return, exchange, refund, and account workflows.",
    "telecom": "Telecom account, service, device, billing, and plan workflows.",
}

_CONFIGS_HINT = (
    "tau2's data tree is not shipped. Clone it and export TAU2_DATA_DIR:\n"
    "  git clone --branch v0.2.0 https://github.com/sierra-research/tau2-bench.git\n"
    "  export TAU2_DATA_DIR=<clone>/data\n"
    "Install tau2 itself with:  uv sync --extra tau2"
)


def _require_tau2() -> None:
    """Fail with instructions rather than an opaque ImportError/KeyError."""

    if not os.environ.get("TAU2_DATA_DIR"):
        raise RuntimeError(f"TAU2_DATA_DIR is not set.\n{_CONFIGS_HINT}")
    try:
        import tau2  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RuntimeError(f"tau2 is not installed ({exc}).\n{_CONFIGS_HINT}") from exc


class Tau2Runner(SingleTaskRunner):
    """Drives tau2 episodes through our agent adapter and harness wiring."""

    name = "tau2"

    def __init__(self, *, domain: str = "retail") -> None:
        # session id -> tau2's own reward, so repeated tasks remain isolated.
        # The three benchmark domains use DB/communicate, env-assertion, and/or
        # action criteria, all handled deterministically by EvaluationType.ALL.
        self._outcomes: dict[str, float] = {}
        self._models: ModelRegistry | None = None
        self._domain = "retail"
        self._workflow_hints: dict[str, tuple[RuleScopeHint, ...]] = {}
        self.configure_dataset(domain)

    def _registry(self) -> ModelRegistry:
        """The model registry, loaded once (the user-simulator role lives here)."""

        if self._models is None:
            from ..cli import CONFIGS

            self._models = load_registry(CONFIGS / "models.yaml")
        return self._models

    # -- SingleTaskRunner -----------------------------------------------------

    def configure_dataset(self, dataset: str) -> None:
        domain = dataset.strip() or self._domain
        if domain not in _SUPPORTED_DOMAINS:
            choices = ", ".join(_SUPPORTED_DOMAINS)
            raise ValueError(f"unsupported tau2 domain {domain!r}; choose one of: {choices}")
        if domain != self._domain:
            # Task ids overlap across domains, so cached verifier outcomes cannot
            # survive a domain switch on a reused runner instance.
            self._outcomes.clear()
            self._workflow_hints.clear()
        self._domain = domain

    def list_tasks(self, dataset: str) -> list[str]:
        self.configure_dataset(dataset)
        _require_tau2()
        from tau2.run import load_tasks

        tasks = load_tasks(self._domain)
        for task in tasks:
            workflow = _safe_task_workflow(task)
            if workflow is not None:
                self._workflow_hints[str(task.id)] = (
                    RuleScopeHint(
                        key=self._domain,
                        description=_DOMAIN_SCOPE_DESCRIPTIONS[self._domain],
                        applicability="topical",
                        recommended=True,
                    ),
                    RuleScopeHint(
                        key=workflow,
                        description=f"{workflow.replace('-', ' ').title()} workflows.",
                        applicability="task",
                        recommended=False,
                    ),
                )
        return [str(task.id) for task in tasks]

    def rule_scope_hints(self, task_id: str) -> tuple[RuleScopeHint, ...]:
        return self._workflow_hints.get(
            task_id,
            (
                RuleScopeHint(
                    key=self._domain,
                    description=_DOMAIN_SCOPE_DESCRIPTIONS[self._domain],
                    applicability="topical",
                    recommended=True,
                ),
            ),
        )

    def outcome_for(self, task_id: str, session_id: str) -> float | None:
        """tau2's own reward for ``session_id``, if this process has graded it.

        Unlike a judged proxy this is ground truth, so when present it is what
        decides rule promotion.
        """

        del task_id
        return self._outcomes.get(session_id)

    async def run_once(
        self,
        *,
        task_id: str,
        session_id: str,
        model: ResolvedModel,
        client: ChatClient,
        max_turns: int,
        wiring: AgentWiring | None,
    ) -> TaskOutcome:
        start = time.monotonic()
        try:
            _require_tau2()
            pieces = self._build(
                task_id=task_id, session_id=session_id, model=model, client=client,
                max_turns=max_turns, wiring=wiring,
            )
        except Exception as exc:  # noqa: BLE001 - setup failure is a trial error
            logger.warning("tau2 setup failed for %s: %s", task_id, exc)
            return _errored(str(exc), time.monotonic() - start)

        orchestrator, task = pieces
        try:
            # Blocking, and it drives the agent — which submits its coroutines
            # back to this loop, so it must not run *on* this loop.
            simulation = await asyncio.to_thread(orchestrator.run)
        except Exception as exc:  # noqa: BLE001 - one bad episode is not a crash
            logger.warning("tau2 episode failed for %s: %s", task_id, exc)
            return _errored(str(exc), time.monotonic() - start)

        reward, native = await self._grade(simulation, task)
        if reward is not None:
            self._outcomes[session_id] = reward

        from tau2.metrics.agent_metrics import is_successful

        return TaskOutcome(
            passed=bool(reward is not None and is_successful(reward)),
            native_metrics=native,
            turns=_agent_turns(simulation),
            wall_time_s=time.monotonic() - start,
            usage=_agent_usage(simulation),
        )

    async def aclose(self) -> None:
        # Episodes are independent and there is no environment server, so unlike
        # AppWorld there is nothing to tear down and no lock to serialize.
        return None

    # -- internals ------------------------------------------------------------

    def _build(
        self,
        *,
        task_id: str,
        session_id: str,
        model: ResolvedModel,
        client: ChatClient,
        max_turns: int,
        wiring: AgentWiring | None,
    ) -> tuple[Any, Any]:
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.registry import registry
        from tau2.run import load_tasks
        from tau2.user.user_simulator import UserSimulator

        from ..adapters.tau2_agent import PandaBenchTau2Agent

        tasks = {str(t.id): t for t in load_tasks(self._domain)}
        try:
            task = tasks[str(task_id)]
        except KeyError:
            raise KeyError(
                f"unknown tau2 task {task_id!r} in domain {self._domain!r}"
            ) from None

        environment = registry.get_env_constructor(self._domain)()
        policy = environment.get_policy()

        agent = PandaBenchTau2Agent(
            environment.get_tools(),
            policy,
            client=client,
            model=model,
            session_id=session_id,
            wiring=wiring,
            loop=asyncio.get_running_loop(),
        )

        # retail raises for get_user_tools; tau2's own runner tolerates it the
        # same way and passes None.
        try:
            user_tools = environment.get_user_tools()
        except Exception:  # noqa: BLE001 - "User tools not available" is normal
            user_tools = None

        registry_cfg = self._registry()
        simulator_model = registry_cfg.resolve(registry_cfg.role("user_simulator"))
        user = UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=simulator_model.litellm_model,
            llm_args={"temperature": 1.0},
        )

        orchestrator = Orchestrator(
            domain=self._domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            # NOTE: a tau2 "step" is one message hop (agent->env, env->agent,
            # agent->user, user->agent), NOT an agent turn — 100 steps is roughly
            # 25-35 agent turns.
            max_steps=max_turns,
        )
        return orchestrator, task

    async def _grade(self, simulation: Any, task: Any) -> tuple[float | None, dict[str, Any]]:
        """Score the episode. ``Orchestrator.run()`` leaves ``reward_info`` unset."""

        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        native: dict[str, Any] = {
            "domain": self._domain,
            "termination_reason": str(getattr(simulation, "termination_reason", "")),
            "step_count": len(getattr(simulation, "messages", []) or []),
            "user_cost": getattr(simulation, "user_cost", None),
            "agent_cost": getattr(simulation, "agent_cost", None),
        }
        try:
            reward_info = await asyncio.to_thread(
                evaluate_simulation,
                simulation=simulation,
                task=task,
                # ALL, never ALL_WITH_NL_ASSERTIONS — the latter calls an LLM.
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=self._domain,
            )
        except Exception as exc:  # noqa: BLE001 - ungraded is not a crash
            logger.warning("tau2 evaluation failed: %s", exc)
            native["eval_error"] = str(exc)
            return None, native

        simulation.reward_info = reward_info
        native["reward"] = reward_info.reward
        breakdown = getattr(reward_info, "reward_breakdown", None) or {}
        native["reward_breakdown"] = {str(k): v for k, v in breakdown.items()}
        return float(reward_info.reward), native


def _agent_turns(simulation: Any) -> int:
    """Agent turns, i.e. assistant messages — not tau2's step count."""

    from tau2.data_model.message import AssistantMessage

    messages = getattr(simulation, "messages", None) or []
    return sum(1 for m in messages if isinstance(m, AssistantMessage))


def _safe_task_workflow(task: Any) -> str | None:
    """Read only explicit bounded tau2 workflow/category metadata when present."""

    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    for key in ("workflow", "category", "task_family"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:48]
    return None


def _agent_usage(simulation: Any) -> Usage:
    """Token/cost totals for the AGENT only.

    The user simulator runs on a fixed model that is identical across arms, so
    charging its spend to the trial would inflate both arms equally and obscure
    the harness's own overhead.
    """

    from tau2.data_model.message import AssistantMessage

    messages = getattr(simulation, "messages", None) or []
    total = Usage()
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        usage = getattr(message, "usage", None) or {}
        total = total + Usage(
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            cost_usd=float(getattr(message, "cost", 0.0) or 0.0),
        )
    return total


def _errored(message: str, wall_time_s: float) -> TaskOutcome:
    return TaskOutcome(
        passed=False,
        native_metrics={"error": message},
        turns=0,
        wall_time_s=wall_time_s,
        usage=Usage(),
        error=message,
    )


def build_tau2_runner(*, dry_run: bool) -> SingleTaskRunner:
    if dry_run:
        return MockTaskRunner("tau2")
    return Tau2Runner()
