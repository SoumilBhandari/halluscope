"""Unit tests for the Semantic Entropy Probe's pure evaluation logic (no model)."""

import numpy as np

from se_probe import evaluate_sep


def test_evaluate_sep_returns_all_metrics():
    rng = np.random.default_rng(0)
    n, d = 80, 16
    H = rng.random((n, d))
    SE = rng.random(n)
    labels = rng.integers(0, 2, n)
    res = evaluate_sep(H, SE, labels)
    for key in ("sep_auroc", "direct_auroc", "se_auroc", "n_test", "se_threshold"):
        assert key in res
    for key in ("sep_auroc", "direct_auroc", "se_auroc"):
        v = res[key]
        assert (0.0 <= v <= 1.0) or v != v  # a valid AUROC or NaN
    assert res["n_test"] == n - int(n * 0.6)


def test_direct_probe_learns_separable_labels():
    # Plant a strong signal in feature 0; the direct correctness probe should beat chance.
    rng = np.random.default_rng(1)
    n, d = 240, 8
    labels = rng.integers(0, 2, n)
    H = rng.random((n, d))
    H[:, 0] += labels * 6.0
    SE = rng.random(n)
    res = evaluate_sep(H, SE, labels)
    assert res["direct_auroc"] > 0.8


def test_sep_recovers_se_signal_when_se_predicts_correctness():
    # Construct a case where SE itself separates the labels and is recoverable
    # from H: the SE probe should then beat chance.
    rng = np.random.default_rng(2)
    n, d = 240, 8
    labels = rng.integers(0, 2, n)
    SE = labels + rng.normal(0, 0.1, n)  # SE tracks correctness
    H = rng.random((n, d))
    H[:, 0] += SE * 3.0  # SE is linearly recoverable from H
    res = evaluate_sep(H, SE, labels)
    assert res["se_auroc"] > 0.8
    assert res["sep_auroc"] > 0.7
