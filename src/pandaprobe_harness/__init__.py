"""PandaProbe Harness — task evaluation with package-owned managed repair.

Public API::

    from pandaprobe_harness import (
        Harness,
        HarnessConfig,
        PandaHarnessHook,
        TaskToolset,
        RepairResult,
        SubprocessCliClient,
    )
"""

from __future__ import annotations

from ._version import __version__
from .adapters.protocol import FrameworkAdapter
from .adapters.raw_loop import RawLoopAdapter
from .agent_tools.spec import ToolDispatcher, ToolSpec
from .agent_tools.toolset import TaskToolset
from .calibration import (
    CalibrationReport,
    LabeledStats,
    MetricCalibration,
    ThresholdPoint,
    calibrate,
)
from .cli.client import CliClient, CliResult
from .cli.subprocess_client import SubprocessCliClient
from .config import HarnessConfig
from .evaluation.evaluator import MetricEvaluator
from .evaluation.history import GateState, ScoreHistoryStore
from .evaluation.history_source import HistorySource
from .evaluation.metrics import EvalReport, Metric, MetricScore
from .evaluation.traces import TraceLocator, TraceRef
from .evaluation.trajectory import GateVerdict, TrajectoryGate
from .filesystem.layout import HarnessFilesystem
from .harness import Harness
from .hook.context import compose_system_preamble
from .hook.core import PandaHarnessHook, SettleResult
from .hook.tiers import TierRunner, VerifierFn
from .hook.turn import RuleScopeHint, TurnContext, parse_turn_payload
from .monitors.client import MonitorClient, MonitorResponse
from .repair.agent import ManagedRepairAgent
from .repair.models import RepairAssignment, RepairResult, RepairStatus, RepairUsage
from .sandbox.policy import ShellPolicy
from .sandbox.shell import RestrictedShellTool, ShellResult
from .validation.regression import CaseResult, RegressionReport, run_regression
from .validation.validator import (
    ForwardTrialValidator,
    ReplayValidator,
    RuleValidator,
    ValidationEngine,
    ValidationVerdict,
)
from .workspace.evalset import CaseKind, EvalCase, EvalSet, ReplayContext, ReplayFn
from .workspace.journal import Journal
from .workspace.mailbox import DiagnosticNotice, Mailbox, MailboxStatus, NoticeMetric, Resolution
from .workspace.rules import (
    Rule,
    RulesCapError,
    RulesStore,
    RuleStatus,
    TrialState,
    derive_notice_tags,
)
from .workspace.sanitize import sanitize_text

__all__ = [
    "CalibrationReport",
    "CaseKind",
    "CaseResult",
    "CliClient",
    "CliResult",
    "DiagnosticNotice",
    "EvalCase",
    "EvalReport",
    "EvalSet",
    "ForwardTrialValidator",
    "FrameworkAdapter",
    "GateState",
    "GateVerdict",
    "Harness",
    "HarnessConfig",
    "HarnessFilesystem",
    "HistorySource",
    "Journal",
    "LabeledStats",
    "Mailbox",
    "MailboxStatus",
    "Metric",
    "MetricCalibration",
    "MetricEvaluator",
    "MetricScore",
    "MonitorClient",
    "MonitorResponse",
    "ManagedRepairAgent",
    "NoticeMetric",
    "PandaHarnessHook",
    "RawLoopAdapter",
    "RegressionReport",
    "ReplayFn",
    "ReplayContext",
    "ReplayValidator",
    "RepairAssignment",
    "RepairResult",
    "RepairStatus",
    "RepairUsage",
    "Resolution",
    "RestrictedShellTool",
    "Rule",
    "RuleScopeHint",
    "RuleStatus",
    "RuleValidator",
    "RulesCapError",
    "RulesStore",
    "ScoreHistoryStore",
    "SettleResult",
    "ShellPolicy",
    "ShellResult",
    "SubprocessCliClient",
    "ThresholdPoint",
    "TierRunner",
    "TaskToolset",
    "ToolDispatcher",
    "ToolSpec",
    "TraceLocator",
    "TraceRef",
    "TrajectoryGate",
    "TrialState",
    "TurnContext",
    "VerifierFn",
    "ValidationEngine",
    "ValidationVerdict",
    "__version__",
    "calibrate",
    "compose_system_preamble",
    "derive_notice_tags",
    "parse_turn_payload",
    "run_regression",
    "sanitize_text",
]
