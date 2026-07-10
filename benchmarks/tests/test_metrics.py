"""Offline tests for the metrics math: pass@1, pass^k, McNemar, paired delta."""

from __future__ import annotations

import pytest

from pandabench.metrics import bootstrap_ci, mcnemar, paired_delta, pass_at_1, pass_hat_k


def test_pass_at_1():
    assert pass_at_1([True, False, True]) == pytest.approx(2 / 3)
    assert pass_at_1([]) == 0.0


def test_pass_hat_k():
    # task0 all-pass, task1 partial, task2 all-fail -> 1/3.
    assert pass_hat_k([[True, True], [True, False], [False, False]]) == pytest.approx(1 / 3)
    assert pass_hat_k([]) == 0.0


def test_mcnemar_exact_small_discordant():
    pairs = [(False, True)] * 5 + [(True, False)] * 1  # 5 improvements, 1 regression
    res = mcnemar(pairs)
    assert res.b == 5 and res.c == 1
    assert res.underpowered is True  # disc=6 < 25
    assert 0.0 <= res.p_value <= 1.0


def test_mcnemar_all_concordant_is_underpowered():
    res = mcnemar([(True, True), (False, False)])
    assert res.b == 0 and res.c == 0
    assert res.p_value == 1.0
    assert res.underpowered is True


def test_mcnemar_large_uses_chi_square():
    pairs = [(False, True)] * 20 + [(True, False)] * 10  # disc=30 >= 25
    res = mcnemar(pairs)
    assert res.underpowered is False
    assert res.statistic > 0


def test_paired_delta():
    pairs = [(False, True), (False, True), (True, True), (False, False)]
    delta = paired_delta(pairs)
    assert delta.n_pairs == 4
    assert delta.rate_a == pytest.approx(0.25)
    assert delta.rate_b == pytest.approx(0.75)
    assert delta.delta == pytest.approx(0.5)
    assert delta.ci_low <= delta.delta <= delta.ci_high


def test_bootstrap_ci_deterministic():
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    a = bootstrap_ci(values, seed=7)
    b = bootstrap_ci(values, seed=7)
    assert a == b
    assert a[0] <= a[1]
