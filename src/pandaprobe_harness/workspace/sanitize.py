"""Sanitization for eval-derived free text (prompt-injection trust boundary).

Text that originates outside the harness — platform eval ``reason`` strings,
trace content echoed into summaries, agent-authored rules — crosses into the
agent's system context via the mailbox and ``harness_rules.md``. Before it
does, it is passed through :func:`sanitize_text`, which:

1. strips control characters (keeping ``\\n``/``\\t``) and ANSI escape
   sequences;
2. collapses long runs of banner characters (``=``, ``-``, ``#``) so injected
   text cannot forge the harness preamble delimiters;
3. neutralizes the harness's own trusted marker phrases by inserting a
   separator between their words;
4. caps the length.

This is deliberately conservative: the goal is that untrusted text can never
masquerade as harness-authored framing, not to make it unreadable.
"""

from __future__ import annotations

import re

__all__ = ["sanitize_text"]

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BANNER_RUN_RE = re.compile(r"([=\-#])\1{7,}")
# Trusted framing phrases the harness itself emits; untrusted text must not
# be able to reproduce them verbatim.
_MARKER_RES = (
    re.compile(r"(pandaprobe)\s+(harness)", re.IGNORECASE),
    re.compile(r"(system)\s+(alert)", re.IGNORECASE),
    re.compile(r"(harness)(:)", re.IGNORECASE),
)
_TRUNCATION_SUFFIX = "…[truncated]"


def sanitize_text(text: str | None, *, max_len: int = 2000) -> str:
    """Neutralize and length-cap untrusted text bound for agent context."""

    if not text:
        return ""
    cleaned = _ANSI_RE.sub("", text)
    cleaned = _CONTROL_RE.sub("", cleaned)
    cleaned = _BANNER_RUN_RE.sub(lambda m: m.group(1) * 7, cleaned)
    for marker in _MARKER_RES:
        cleaned = marker.sub(r"\1·\2", cleaned)
    if max_len > 0 and len(cleaned) > max_len:
        cleaned = cleaned[: max(0, max_len - len(_TRUNCATION_SUFFIX))] + _TRUNCATION_SUFFIX
    return cleaned
