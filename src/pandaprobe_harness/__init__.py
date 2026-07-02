"""PandaProbe Harness — autonomous, pull-based self-healing for LLM agents.

Public API::

    from pandaprobe_harness import (
        Harness,
        HarnessConfig,
        PandaHarnessHook,
        HarnessToolset,
        Mailbox,
        Journal,
        RulesStore,
        RestrictedShellTool,
        SubprocessCliClient,
    )
"""

from __future__ import annotations

from .adapters.protocol import FrameworkAdapter
from .adapters.raw_loop import RawLoopAdapter
from .agent_tools.spec import ToolSpec
from .agent_tools.toolset import HarnessToolset
from .cli.client import CliClient, CliResult
from .cli.subprocess_client import SubprocessCliClient
from .config import HarnessConfig
from .evaluation.evaluator import MetricEvaluator
from .evaluation.history import EwmaState, ScoreHistoryStore
from .evaluation.history_source import HistorySource
from .evaluation.metrics import EvalReport, Metric, MetricScore
from .evaluation.trends import TrendDetector, TrendVerdict
from .filesystem.layout import HarnessFilesystem
from .harness import Harness
from .hook.context import compose_system_preamble
from .hook.core import PandaHarnessHook
from .hook.turn import TurnContext, parse_turn_payload
from .monitors.client import MonitorClient, MonitorResponse
from .sandbox.policy import ShellPolicy
from .sandbox.shell import RestrictedShellTool, ShellResult
from .workspace.journal import Journal
from .workspace.mailbox import (
    DiagnosticNotice,
    Mailbox,
    MailboxStatus,
    NoticeMetric,
    Resolution,
)
from .workspace.rules import Rule, RulesCapError, RulesStore
from .workspace.sanitize import sanitize_text

__version__ = "0.5.0"

__all__ = [
    "CliClient",
    "CliResult",
    "DiagnosticNotice",
    "EvalReport",
    "EwmaState",
    "FrameworkAdapter",
    "Harness",
    "HarnessConfig",
    "HarnessFilesystem",
    "HarnessToolset",
    "HistorySource",
    "Journal",
    "Mailbox",
    "MailboxStatus",
    "Metric",
    "MetricEvaluator",
    "MetricScore",
    "MonitorClient",
    "MonitorResponse",
    "NoticeMetric",
    "PandaHarnessHook",
    "RawLoopAdapter",
    "Resolution",
    "RestrictedShellTool",
    "Rule",
    "RulesCapError",
    "RulesStore",
    "ScoreHistoryStore",
    "ShellPolicy",
    "ShellResult",
    "SubprocessCliClient",
    "ToolSpec",
    "TrendDetector",
    "TrendVerdict",
    "TurnContext",
    "__version__",
    "compose_system_preamble",
    "parse_turn_payload",
    "sanitize_text",
]
