"""Reliability metrics and paired statistics (stdlib + numpy/scipy, no network).

``pass@1`` = fraction of tasks whose FIRST trial passed. ``pass^k`` = fraction
of tasks whose ALL k trials passed. Arm comparisons are paired per task/seed:
McNemar's test on the discordant pass/fail pairs plus a bootstrap CI on the
pass-rate delta. At ~30-40 eval tasks these detect only large deltas, so results
are framed as directional (the report states the power caveat).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "McNemarResult",
    "PairedDelta",
    "bootstrap_ci",
    "mcnemar",
    "paired_delta",
    "pass_at_1",
    "pass_hat_k",
]


def pass_at_1(first_trial_passes: Sequence[bool]) -> float:
    """Fraction of tasks whose first trial passed."""

    if not first_trial_passes:
        return 0.0
    return sum(1 for p in first_trial_passes if p) / len(first_trial_passes)


def pass_hat_k(trials_by_task: Sequence[Sequence[bool]]) -> float:
    """Fraction of tasks whose ALL trials passed (pass^k)."""

    tasks = [t for t in trials_by_task if t]
    if not tasks:
        return 0.0
    return sum(1 for trials in tasks if all(trials)) / len(tasks)


def bootstrap_ci(
    values: Sequence[float], *, n: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """b = A-fail/B-pass (improvements); c = A-pass/B-fail (regressions)."""

    b: int
    c: int
    statistic: float
    p_value: float
    underpowered: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "b_improved": self.b, "c_regressed": self.c,
            "statistic": self.statistic, "p_value": self.p_value,
            "underpowered": self.underpowered,
        }


def mcnemar(pairs: Sequence[tuple[bool, bool]], *, exact_threshold: int = 25) -> McNemarResult:
    """McNemar's test on paired (arm_A_pass, arm_B_pass) outcomes.

    Exact binomial when the discordant count is small; otherwise chi-square with
    continuity correction. ``underpowered`` flags a discordant count too small to
    detect anything but large effects.
    """

    b = sum(1 for a, x in pairs if not a and x)  # A fail, B pass -> improvement
    c = sum(1 for a, x in pairs if a and not x)  # A pass, B fail -> regression
    disc = b + c
    if disc == 0:
        return McNemarResult(b, c, 0.0, 1.0, underpowered=True)
    if disc < exact_threshold:
        k = min(b, c)
        p = float(min(1.0, 2.0 * stats.binom.cdf(k, disc, 0.5)))
        statistic = float(min(b, c))
    else:
        statistic = (abs(b - c) - 1) ** 2 / disc
        p = float(stats.chi2.sf(statistic, df=1))
    return McNemarResult(b, c, statistic, p, underpowered=disc < exact_threshold)


@dataclass(frozen=True, slots=True)
class PairedDelta:
    """Paired pass-rate delta (arm B - arm A) with a bootstrap CI + McNemar."""

    n_pairs: int
    rate_a: float
    rate_b: float
    delta: float
    ci_low: float
    ci_high: float
    mcnemar: McNemarResult

    def to_dict(self) -> dict[str, object]:
        return {
            "n_pairs": self.n_pairs, "rate_a": self.rate_a, "rate_b": self.rate_b,
            "delta": self.delta, "ci_low": self.ci_low, "ci_high": self.ci_high,
            **self.mcnemar.to_dict(),
        }


def paired_delta(
    pairs: Sequence[tuple[bool, bool]], *, n: int = 10_000, seed: int = 0
) -> PairedDelta:
    """Compare two arms over paired pass/fail outcomes on the same tasks."""

    if not pairs:
        return PairedDelta(0, 0.0, 0.0, 0.0, 0.0, 0.0, mcnemar([]))
    a = [1.0 if x else 0.0 for x, _ in pairs]
    b = [1.0 if y else 0.0 for _, y in pairs]
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    rate_a = sum(a) / len(a)
    rate_b = sum(b) / len(b)
    lo, hi = bootstrap_ci(diffs, n=n, seed=seed)
    return PairedDelta(len(pairs), rate_a, rate_b, rate_b - rate_a, lo, hi, mcnemar(pairs))
