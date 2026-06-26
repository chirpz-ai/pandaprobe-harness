"""Rules → startup-context composition (closes the self-healing loop).

The agent appends learned mitigations to ``harness_rules.md`` during self-heal,
but those rules only take effect if they are read back into the agent's context
on subsequent runs. ``compose_system_preamble`` returns the living rules wrapped
in a clearly-delimited block that a developer prepends to their agent's system
prompt at startup (each adapter also exposes it via ``startup_context``).
"""

from __future__ import annotations

from ..filesystem.layout import HarnessFilesystem

__all__ = ["compose_system_preamble"]

_HEADER = "===================== PANDAPROBE HARNESS RULES ====================="
_FOOTER = "===================================================================="
_INTRO = (
    "The following are the living, self-authored operating rules for this agent, "
    "learned from prior failures. Treat them as binding constraints and apply "
    "them before acting."
)


def compose_system_preamble(filesystem: HarnessFilesystem) -> str:
    """Return the harness rules as a system-prompt preamble block.

    Returns an empty string if the rules file does not exist yet (nothing to
    inject), so callers can unconditionally prepend the result.
    """

    try:
        rules = filesystem.read_rules().strip()
    except FileNotFoundError:
        return ""
    if not rules:
        return ""
    return f"{_HEADER}\n{_INTRO}\n\n{rules}\n{_FOOTER}\n"
