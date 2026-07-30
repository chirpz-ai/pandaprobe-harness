"""The one-call facade: workspace + hook + toolset, with a zero-adapter path.

``Harness.create()`` provisions the diagnostic workspace, wires the hook,
mailbox, journal, rules, and toolset together, and (optionally) runs the
startup health check. Any custom agent loop integrates in a handful of lines
— no adapter required::

    harness = Harness.create()
    system_prompt = harness.system_context() + my_prompt
    tools = my_tools + list(harness.toolset.specs())

    async with harness.turn(session_id):
        await my_agent_step(...)

    # or:  result = await harness.run_turn(session_id, my_agent_step, ...)
    # or:  decorated = harness.turn(session_id)(my_agent_step)

Per-framework factories (``for_langgraph``, ``for_crewai``, …) additionally
wire the framework's turn detector and preserve its session resolution by
passing ``adapter.parse_turn`` as the hook's parser.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, ParamSpec, TypeVar

from .agent_tools.toolset import HarnessToolset
from .cli.client import CliClient
from .cli.subprocess_client import SubprocessCliClient
from .config import HarnessConfig
from .evaluation.evaluator import MetricEvaluator
from .evaluation.history import ScoreHistoryStore
from .evaluation.metrics import EvalReport
from .filesystem.layout import HarnessFilesystem
from .hook.core import PandaHarnessHook, SettleResult
from .hook.tiers import VerifierFn
from .sandbox.policy import ShellPolicy
from .sandbox.shell import RestrictedShellTool
from .validation.regression import RegressionReport, run_regression
from .validation.validator import ValidationVerdict
from .workspace.evalset import EvalSet, ReplayFn
from .workspace.journal import Journal
from .workspace.mailbox import Mailbox
from .workspace.rules import RulesStore

__all__ = ["Harness"]

logger = logging.getLogger("pandaprobe_harness.harness")

P = ParamSpec("P")
R = TypeVar("R")


class _TurnScope:
    """Async context manager *and* decorator delimiting one agent turn.

    With ``settle=True`` the scope also **waits** for the turn's self-heal cycle
    before returning, so a rule the agent learns from this turn is in force for
    the next one. Off by default: it adds per-turn latency, and every existing
    caller's semantics are fire-and-forget.
    """

    def __init__(self, harness: Harness, session_id: str, *, settle: bool = False) -> None:
        self._harness = harness
        self._session_id = session_id
        self._settle = settle

    async def __aenter__(self) -> _TurnScope:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        # A failed turn is still an evaluable turn — fire on exceptional exit too.
        await self._finish()
        return False

    def __call__(self, fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return await fn(*args, **kwargs)
            finally:
                await self._finish()

        return wrapper

    async def _finish(self) -> None:
        self._harness._notify_turn(self._session_id)
        if self._settle:
            await self._harness.settle(self._session_id)


class Harness:
    """The assembled self-healing envelope around an agent."""

    def __init__(
        self,
        *,
        config: HarnessConfig,
        cli: CliClient,
        filesystem: HarnessFilesystem,
        mailbox: Mailbox,
        journal: Journal,
        rules: RulesStore,
        history: ScoreHistoryStore,
        hook: PandaHarnessHook,
        toolset: HarnessToolset,
        shell: RestrictedShellTool,
        evalset: EvalSet,
        evaluator: MetricEvaluator,
        replay: ReplayFn | None = None,
        adapter: Any | None = None,
    ) -> None:
        self._config = config
        self._cli = cli
        self._filesystem = filesystem
        self._mailbox = mailbox
        self._journal = journal
        self._rules = rules
        self._history = history
        self._hook = hook
        self._toolset = toolset
        self._shell = shell
        self._evalset = evalset
        self._evaluator = evaluator
        self._replay = replay
        self._adapter = adapter
        self._turn_counts: dict[str, int] = {}
        self._background: set[asyncio.Task[Any]] = set()

    # -- construction -----------------------------------------------------------

    @classmethod
    def create(
        cls,
        config: HarnessConfig | None = None,
        *,
        cli: CliClient | None = None,
        replay: ReplayFn | None = None,
        verifier: VerifierFn | None = None,
    ) -> Harness:
        """Provision the workspace and assemble the full harness (no adapter).

        ``replay`` is the optional developer-supplied replay function used by
        candidate-rule validation and :meth:`run_regression`; without it the
        harness falls back to forward-trial validation and regression runs
        degrade to skips.

        ``verifier`` is an optional outcome oracle — ``(session_id, end_state) ->
        float | bool`` — for developers who already know what success means (a
        golden dataset, a benchmark evaluator, a business rule). When supplied it
        becomes an authoritative ``outcome_correct`` signal that drives breaches
        and rule promotion. Prefer a *continuous* score over a pass/fail flag: a
        binary that is almost always 0 discriminates no better than the degenerate
        session metrics v2 exists to replace.
        """

        return cls._build(
            config=config, cli=cli, adapter=None, replay=replay, verifier=verifier
        )

    @classmethod
    def _build(
        cls,
        *,
        config: HarnessConfig | None,
        cli: CliClient | None,
        adapter: Any | None,
        replay: ReplayFn | None = None,
        verifier: VerifierFn | None = None,
    ) -> Harness:
        cfg = config or HarnessConfig.from_env()
        client = cli or SubprocessCliClient(cfg.cli_binary, default_timeout=cfg.cli_timeout_s)

        filesystem = HarnessFilesystem(cfg)
        filesystem.provision()
        mailbox = Mailbox(cfg)
        mailbox.provision()
        journal = Journal(cfg)
        rules = RulesStore(cfg, journal=journal)
        rules.sync_markdown()
        history = ScoreHistoryStore(cfg)
        evalset = EvalSet(cfg, journal=journal)
        evalset.provision()
        evaluator = MetricEvaluator(client, cfg)

        hook = PandaHarnessHook(
            client,
            config=cfg,
            mailbox=mailbox,
            journal=journal,
            rules=rules,
            filesystem=filesystem,
            evaluator=evaluator,
            history=history,
            evalset=evalset,
            replay=replay,
            verifier=verifier,
            parser=adapter.parse_turn if adapter is not None else None,
        )
        if adapter is not None:
            adapter.register(hook)

        toolset = HarnessToolset(
            config=cfg,
            cli=client,
            mailbox=mailbox,
            journal=journal,
            rules=rules,
        )
        shell = RestrictedShellTool(ShellPolicy(workdir=cfg.harness_root))

        harness = cls(
            config=cfg,
            cli=client,
            filesystem=filesystem,
            mailbox=mailbox,
            journal=journal,
            rules=rules,
            history=history,
            hook=hook,
            toolset=toolset,
            shell=shell,
            evalset=evalset,
            evaluator=evaluator,
            replay=replay,
            adapter=adapter,
        )
        if cfg.health_check:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass  # no loop yet: the check runs lazily before the first eval
            else:
                task = loop.create_task(hook.check_health())
                harness._background.add(task)
                task.add_done_callback(harness._background.discard)
        return harness

    # -- per-framework factories ---------------------------------------------------

    @classmethod
    def for_langgraph(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
        replay: ReplayFn | None = None,
    ) -> Harness:
        from .adapters.langgraph import LangGraphAdapter

        return cls._build(
            config=config,
            cli=cli,
            adapter=LangGraphAdapter(session_id=session_id),
            replay=replay,
        )

    @classmethod
    def for_langchain(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
        replay: ReplayFn | None = None,
    ) -> Harness:
        from .adapters.langchain import LangChainAdapter

        return cls._build(
            config=config,
            cli=cli,
            adapter=LangChainAdapter(session_id=session_id),
            replay=replay,
        )

    @classmethod
    def for_deepagents(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
        replay: ReplayFn | None = None,
    ) -> Harness:
        from .adapters.deepagents import DeepAgentsAdapter

        return cls._build(
            config=config,
            cli=cli,
            adapter=DeepAgentsAdapter(session_id=session_id),
            replay=replay,
        )

    @classmethod
    def for_crewai(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
        replay: ReplayFn | None = None,
    ) -> Harness:
        from .adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(session_id=session_id)
        harness = cls._build(config=config, cli=cli, adapter=adapter, replay=replay)
        adapter.instrument()  # logs and degrades gracefully when the extra is absent
        return harness

    @classmethod
    def for_claude_agent_sdk(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
        replay: ReplayFn | None = None,
    ) -> Harness:
        from .adapters.claude_agent_sdk import ClaudeAgentSDKAdapter

        adapter = ClaudeAgentSDKAdapter(session_id=session_id)
        harness = cls._build(config=config, cli=cli, adapter=adapter, replay=replay)
        adapter.instrument()
        return harness

    @classmethod
    def for_openai_agents(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
        replay: ReplayFn | None = None,
    ) -> Harness:
        from .adapters.openai_agents import OpenAIAgentsAdapter

        adapter = OpenAIAgentsAdapter(session_id=session_id)
        harness = cls._build(config=config, cli=cli, adapter=adapter, replay=replay)
        adapter.instrument()
        return harness

    # -- surface ---------------------------------------------------------------

    @property
    def config(self) -> HarnessConfig:
        return self._config

    @property
    def cli(self) -> CliClient:
        return self._cli

    @property
    def filesystem(self) -> HarnessFilesystem:
        return self._filesystem

    @property
    def hook(self) -> PandaHarnessHook:
        return self._hook

    @property
    def mailbox(self) -> Mailbox:
        return self._mailbox

    @property
    def journal(self) -> Journal:
        return self._journal

    @property
    def rules(self) -> RulesStore:
        return self._rules

    @property
    def toolset(self) -> HarnessToolset:
        return self._toolset

    @property
    def evalset(self) -> EvalSet:
        """The replayable regression eval-set (failure/win scenarios)."""

        return self._evalset

    @property
    def shell(self) -> RestrictedShellTool:
        return self._shell

    @property
    def adapter(self) -> Any:
        """The framework adapter wired by a ``for_*`` factory (else ``None``)."""

        return self._adapter

    def system_context(self) -> str:
        """The block to prepend to the agent's system prompt.

        The skill root — self-heal protocol, tool list, and a generated index of
        the agent's rule files — plus a banner when notices are pending. It carries
        **no rule text**: the agent reads the files it judges relevant with
        ``harness_rules_read``, so the workspace stays its own.
        """

        return self._hook.startup_context()

    def on_turn_end(self, raw_turn: object) -> None:
        self._hook.on_turn_end(raw_turn)

    async def refresh(self, session_id: str) -> EvalReport | None:
        return await self._hook.refresh(session_id)

    async def refresh_all(self) -> None:
        await self._hook.refresh_all()

    async def settle(
        self, session_id: str, *, timeout: float | None = None
    ) -> SettleResult:
        """Wait for this turn's self-heal cycle: evaluation, notice, validation.

        Call once per turn to get **in-session** healing — the agent then starts
        its next turn already able to see the notice its last turn produced and
        any rule promoted from it. Without this barrier the agent can run several
        turns ahead of the (slower) evaluation, which is how a session ends before
        its own diagnosis arrives.

        Bounded by ``barrier_timeout_s``; on expiry the work continues detached
        and ``SettleResult.timed_out`` is set.
        """

        return await self._hook.settle(session_id, timeout=timeout)

    async def check_health(self) -> bool:
        return await self._hook.check_health()

    async def validate_candidates(self) -> list[ValidationVerdict]:
        """Run one candidate-evaluation round now (empty when validation is off)."""

        return await self._hook.validate_candidates()

    async def drain_validation(self) -> None:
        """Await in-flight candidate-validation tasks (bounded)."""

        await self._hook.drain_validation()

    async def run_regression(self, *, sample: int | None = None) -> RegressionReport:
        """Replay the eval-set against the current rule set and report drift.

        Requires the ``replay`` function passed at construction to actually
        re-run cases; without one the report is all skips (with one clear
        warning logged), never an exception.
        """

        return await run_regression(
            config=self._config,
            rules=self._rules,
            evalset=self._evalset,
            evaluator=self._evaluator,
            journal=self._journal,
            replay=self._replay,
            sample=sample,
        )

    # -- zero-adapter turn helpers ---------------------------------------------

    def turn(self, session_id: str, *, settle: bool = False) -> _TurnScope:
        """Delimit one agent turn: ``async with`` context manager or decorator.

        Pass ``settle=True`` to also await the turn's self-heal cycle on exit
        (see :meth:`settle`).
        """

        return _TurnScope(self, session_id, settle=settle)

    async def run_turn(
        self,
        session_id: str,
        fn: Callable[P, Awaitable[R]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Run one arbitrary agent step, firing turn-end on completion.

        To also await the turn's self-heal cycle, use
        ``async with harness.turn(session_id, settle=True):`` instead — this
        signature forwards ``**kwargs`` verbatim to ``fn``, so it cannot claim a
        keyword of its own without risking a collision.
        """

        try:
            return await fn(*args, **kwargs)
        finally:
            self._notify_turn(session_id)

    #: Bound the facade's per-session turn counter (memory in long-lived procs).
    _MAX_TRACKED_SESSIONS = 4096

    def _notify_turn(self, session_id: str) -> None:
        count = self._turn_counts.get(session_id, 0) + 1
        self._turn_counts[session_id] = count
        if len(self._turn_counts) > self._MAX_TRACKED_SESSIONS:
            self._turn_counts.pop(next(iter(self._turn_counts)), None)
        self._hook.on_turn_end(
            {"session_id": session_id, "turn_index": count, "end_state": {}}
        )
