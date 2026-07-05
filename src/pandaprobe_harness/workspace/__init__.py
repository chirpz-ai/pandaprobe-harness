"""The diagnostic workspace substrate: mailbox, journal, and structured rules.

These three stores are the pull model's backbone: the hook *writes* here, the
agent *reads and rewrites* here through its harness toolset, and nothing is
ever pushed into the agent's input queue.
"""

from __future__ import annotations

from .evalset import CaseKind, EvalCase, EvalSet, ReplayFn
from .journal import Journal
from .mailbox import (
    DiagnosticNotice,
    Mailbox,
    MailboxStatus,
    NoticeMetric,
    Resolution,
    Severity,
)
from .rules import (
    Rule,
    RulesCapError,
    RulesStore,
    RuleStatus,
    TrialState,
    derive_notice_tags,
)
from .sanitize import sanitize_text

__all__ = [
    "CaseKind",
    "DiagnosticNotice",
    "EvalCase",
    "EvalSet",
    "Journal",
    "Mailbox",
    "MailboxStatus",
    "NoticeMetric",
    "ReplayFn",
    "Resolution",
    "Rule",
    "RuleStatus",
    "RulesCapError",
    "RulesStore",
    "Severity",
    "TrialState",
    "derive_notice_tags",
    "sanitize_text",
]
