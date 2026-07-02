"""The diagnostic workspace substrate: mailbox, journal, and structured rules.

These three stores are the pull model's backbone: the hook *writes* here, the
agent *reads and rewrites* here through its harness toolset, and nothing is
ever pushed into the agent's input queue.
"""

from __future__ import annotations

from .journal import Journal
from .mailbox import (
    DiagnosticNotice,
    Mailbox,
    MailboxStatus,
    NoticeMetric,
    Resolution,
    Severity,
)
from .rules import Rule, RulesCapError, RulesStore
from .sanitize import sanitize_text

__all__ = [
    "DiagnosticNotice",
    "Journal",
    "Mailbox",
    "MailboxStatus",
    "NoticeMetric",
    "Resolution",
    "Rule",
    "RulesCapError",
    "RulesStore",
    "Severity",
    "sanitize_text",
]
