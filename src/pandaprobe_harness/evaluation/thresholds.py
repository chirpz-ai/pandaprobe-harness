"""Isolated breach-decision policy.

Kept separate from the evaluator so the comparison semantics can be unit-tested
in isolation and tuned without touching orchestration code.
"""

from __future__ import annotations

__all__ = ["is_breach"]


def is_breach(value: float | None, threshold: float) -> bool:
    """Return True when ``value`` is a concrete score strictly below ``threshold``.

    A ``None`` value (pending, unresolved, or degraded eval) is *not* a breach —
    the harness never alerts on the absence of a score.
    """

    return value is not None and value < threshold
