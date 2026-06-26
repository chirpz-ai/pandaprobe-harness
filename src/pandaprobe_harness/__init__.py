"""PandaProbe Harness — a meta-cognitive diagnostic sandbox for agents.

Public API::

    from pandaprobe_harness import (
        HarnessConfig,
        PandaHarnessHook,
        MetricEvaluator,
        HarnessFilesystem,
        RawLoopAdapter,
        RestrictedShellTool,
        SubprocessCliClient,
        MonitorClient,
    )
"""

from __future__ import annotations

from .adapters.protocol import FrameworkAdapter
from .adapters.raw_loop import RawLoopAdapter
from .cli.client import CliClient, CliResult
from .cli.subprocess_client import SubprocessCliClient
from .config import HarnessConfig
from .evaluation.evaluator import MetricEvaluator
from .evaluation.history import EwmaState, ScoreHistoryStore
from .evaluation.metrics import EvalReport, Metric, MetricScore
from .evaluation.trends import TrendDetector, TrendVerdict
from .filesystem.layout import HarnessFilesystem
from .hook.alert import build_system_alert, build_trend_alert
from .hook.context import compose_system_preamble
from .hook.core import PandaHarnessHook
from .hook.turn import TurnContext
from .monitors.client import MonitorClient, MonitorResponse
from .sandbox.policy import ShellPolicy
from .sandbox.shell import RestrictedShellTool, ShellResult

__version__ = "0.4.0"

__all__ = [
    "HarnessConfig",
    "PandaHarnessHook",
    "TurnContext",
    "MetricEvaluator",
    "EvalReport",
    "Metric",
    "MetricScore",
    "ScoreHistoryStore",
    "EwmaState",
    "TrendDetector",
    "TrendVerdict",
    "HarnessFilesystem",
    "FrameworkAdapter",
    "RawLoopAdapter",
    "RestrictedShellTool",
    "ShellResult",
    "ShellPolicy",
    "CliClient",
    "CliResult",
    "SubprocessCliClient",
    "MonitorClient",
    "MonitorResponse",
    "build_system_alert",
    "build_trend_alert",
    "compose_system_preamble",
    "__version__",
]
