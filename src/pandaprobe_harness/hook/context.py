"""The skill root + mailbox banner → the agent's startup/system context.

This is the *only* framework-facing "push" left in the pull model, and it is a
passive one: a block the developer prepends to the agent's system prompt (every
framework already loads one). It carries exactly two things:

1. the **skill root** — the self-heal protocol, the tool list, and a generated
   References index of the rule files that exist, and
2. a compact mailbox banner when diagnostic notices are pending.

It deliberately carries **no rule text**. v1 rendered every active rule into the
prompt; v2 does not, because the workspace belongs to the agent: it reads
``rules/global.md`` and any scoped file it judges relevant, on demand, via
``harness_rules_read``. The root names those files so the pull is one tool call
away.

No eval-derived free text enters this preamble either — the banner is counts plus
a severity enum. For frameworks that rebuild the system prompt each turn, the
banner is the trigger; for static-prompt frameworks, the protocol's "work through
pending notices" instruction is.
"""

from __future__ import annotations

import logging

from ..workspace.mailbox import Mailbox
from ..workspace.rules import RulesStore

__all__ = ["compose_system_preamble"]

logger = logging.getLogger("pandaprobe_harness.hook")

_HEADER = "===================== PANDAPROBE HARNESS ==========================="
_FOOTER = "===================================================================="


def compose_system_preamble(rules: RulesStore, mailbox: Mailbox) -> str:
    """Return the harness system-context block (skill root + pending-notice banner).

    Takes no task hint: v1 used one to pre-select which rules to inline, and there
    is no rule text here to select any more. The agent conditions its own retrieval
    on the task through ``harness_rules_search`` / ``harness_rules_read``.

    Degrades gracefully: any workspace read failure yields a smaller block, never
    an exception into the host loop.
    """

    banner = ""
    try:
        status = mailbox.status()
        if status.pending_count > 0:
            severity = status.max_severity or "breach"
            banner = (
                f"\n⚠ HARNESS: {status.pending_count} pending diagnostic notice(s) "
                f"(max severity: {severity}). Before continuing, use your harness "
                "tools to read the mailbox, inspect the flagged trace, record a "
                "mitigation rule, and acknowledge each notice.\n"
            )
    except Exception:  # noqa: BLE001 - context assembly must never raise
        logger.debug("failed to read mailbox status for context", exc_info=True)

    try:
        root = rules.render_root().strip()
    except Exception:  # noqa: BLE001 - context assembly must never raise
        logger.debug("failed to render the skill root for context", exc_info=True)
        root = ""

    return f"{_HEADER}\n{root}\n{banner}{_FOOTER}\n"
