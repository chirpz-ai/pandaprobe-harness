"""Stable capability-only preamble for the developer's task agent."""

from __future__ import annotations

__all__ = ["compose_system_preamble"]

_HEADER = "===================== PANDAPROBE HARNESS ==========================="
_FOOTER = "===================================================================="
_CAPABILITY = (
    "Optional learned guidance is available through PandaProbe's read-only tools. "
    "Call harness_rules_list to load harness_guide.md and its scope "
    "references; use harness_rules_read, harness_rules_search, or "
    "harness_rule_status only when useful for the current task."
)


def compose_system_preamble(
    *_args: object,
    task_hint: str | None = None,
) -> str:
    """Return a constant preamble without reading or expanding the rule store.

    The unused positional arguments and ``task_hint`` preserve the unreleased
    integration call shape while making the no-implicit-retrieval boundary
    explicit and testable.
    """

    del task_hint
    return f"{_HEADER}\n{_CAPABILITY}\n{_FOOTER}\n"
