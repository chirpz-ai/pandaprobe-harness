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
from .evaluation.history import ScoreHistoryStore
from .evaluation.metrics import EvalReport
from .filesystem.layout import HarnessFilesystem
from .hook.core import PandaHarnessHook
from .sandbox.policy import ShellPolicy
from .sandbox.shell import RestrictedShellTool
from .workspace.journal import Journal
from .workspace.mailbox import Mailbox
from .workspace.rules import RulesStore

__all__ = ["Harness"]

logger = logging.getLogger("pandaprobe_harness.harness")

P = ParamSpec("P")
R = TypeVar("R")


class _TurnScope:
    """Async context manager *and* decorator delimiting one agent turn."""

    def __init__(self, harness: Harness, session_id: str) -> None:
        self._harness = harness
        self._session_id = session_id

    async def __aenter__(self) -> _TurnScope:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        # A failed turn is still an evaluable turn — fire on exceptional exit too.
        self._harness._notify_turn(self._session_id)
        return False

    def __call__(self, fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return await fn(*args, **kwargs)
            finally:
                self._harness._notify_turn(self._session_id)

        return wrapper


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
        self._adapter = adapter
        self._turn_counts: dict[str, int] = {}
        self._background: set[asyncio.Task[Any]] = set()

    # -- construction -----------------------------------------------------------

    @classmethod
    def create(
        cls, config: HarnessConfig | None = None, *, cli: CliClient | None = None
    ) -> Harness:
        """Provision the workspace and assemble the full harness (no adapter)."""

        return cls._build(config=config, cli=cli, adapter=None)

    @classmethod
    def _build(
        cls,
        *,
        config: HarnessConfig | None,
        cli: CliClient | None,
        adapter: Any | None,
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

        hook = PandaHarnessHook(
            client,
            config=cfg,
            mailbox=mailbox,
            journal=journal,
            rules=rules,
            filesystem=filesystem,
            history=history,
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
            history=history,
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
    ) -> Harness:
        from .adapters.langgraph import LangGraphAdapter

        return cls._build(config=config, cli=cli, adapter=LangGraphAdapter(session_id=session_id))

    @classmethod
    def for_langchain(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
    ) -> Harness:
        from .adapters.langchain import LangChainAdapter

        return cls._build(config=config, cli=cli, adapter=LangChainAdapter(session_id=session_id))

    @classmethod
    def for_deepagents(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
    ) -> Harness:
        from .adapters.deepagents import DeepAgentsAdapter

        return cls._build(config=config, cli=cli, adapter=DeepAgentsAdapter(session_id=session_id))

    @classmethod
    def for_crewai(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
    ) -> Harness:
        from .adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(session_id=session_id)
        harness = cls._build(config=config, cli=cli, adapter=adapter)
        adapter.instrument()  # logs and degrades gracefully when the extra is absent
        return harness

    @classmethod
    def for_claude_agent_sdk(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
    ) -> Harness:
        from .adapters.claude_agent_sdk import ClaudeAgentSDKAdapter

        adapter = ClaudeAgentSDKAdapter(session_id=session_id)
        harness = cls._build(config=config, cli=cli, adapter=adapter)
        adapter.instrument()
        return harness

    @classmethod
    def for_openai_agents(
        cls,
        *,
        session_id: str | None = None,
        config: HarnessConfig | None = None,
        cli: CliClient | None = None,
    ) -> Harness:
        from .adapters.openai_agents import OpenAIAgentsAdapter

        adapter = OpenAIAgentsAdapter(session_id=session_id)
        harness = cls._build(config=config, cli=cli, adapter=adapter)
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
    def shell(self) -> RestrictedShellTool:
        return self._shell

    @property
    def adapter(self) -> Any:
        """The framework adapter wired by a ``for_*`` factory (else ``None``)."""

        return self._adapter

    def system_context(self) -> str:
        """Rules + pull protocol + mailbox banner, for the agent's system prompt."""

        return self._hook.startup_context()

    def on_turn_end(self, raw_turn: object) -> None:
        self._hook.on_turn_end(raw_turn)

    async def refresh(self, session_id: str) -> EvalReport | None:
        return await self._hook.refresh(session_id)

    async def refresh_all(self) -> None:
        await self._hook.refresh_all()

    async def check_health(self) -> bool:
        return await self._hook.check_health()

    # -- zero-adapter turn helpers ---------------------------------------------

    def turn(self, session_id: str) -> _TurnScope:
        """Delimit one agent turn: ``async with`` context manager or decorator."""

        return _TurnScope(self, session_id)

    async def run_turn(
        self,
        session_id: str,
        fn: Callable[P, Awaitable[R]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Run one arbitrary agent step, firing turn-end on completion."""

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
