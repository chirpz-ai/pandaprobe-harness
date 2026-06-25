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
    )
"""

from __future__ import annotations

from .adapters.protocol import FrameworkAdapter
from .adapters.raw_loop import RawLoopAdapter
from .cli.client import CliClient, CliResult
from .cli.subprocess_client import SubprocessCliClient
from .config import HarnessConfig
from .evaluation.evaluator import MetricEvaluator
from .evaluation.metrics import EvalReport, Metric, MetricScore
from .filesystem.layout import HarnessFilesystem
from .hook.core import PandaHarnessHook
from .hook.turn import TurnContext
from .sandbox.policy import ShellPolicy
from .sandbox.shell import RestrictedShellTool, ShellResult

__version__ = "0.3.0"

__all__ = [
    "HarnessConfig",
    "PandaHarnessHook",
    "TurnContext",
    "MetricEvaluator",
    "EvalReport",
    "Metric",
    "MetricScore",
    "HarnessFilesystem",
    "FrameworkAdapter",
    "RawLoopAdapter",
    "RestrictedShellTool",
    "ShellResult",
    "ShellPolicy",
    "CliClient",
    "CliResult",
    "SubprocessCliClient",
    "__version__",
]
