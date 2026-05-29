"""Unit tests for the eval harness helpers (no model)."""

import math

from eval import _fmt, _fmt_ci, auroc, bootstrap_ci


def test_auroc_perfect_separation():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_auroc_fully_inverted():
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_auroc_single_class_is_nan_not_crash():
    result = auroc([0.1, 0.2, 0.3], [0, 0, 0])
    assert math.isnan(result)


def test_fmt_formats_floats():
    assert _fmt(0.8534).strip() == "0.8534"


def test_fmt_handles_nan():
    assert _fmt(float("nan")).strip() == "N/A"


def test_bootstrap_ci_brackets_a_valid_interval():
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 6
    labels = [0, 0, 0, 1, 1, 1] * 6
    lo, hi = bootstrap_ci(scores, labels, n_boot=200)
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_empty_is_nan():
    lo, hi = bootstrap_ci([], [])
    assert math.isnan(lo) and math.isnan(hi)


def test_fmt_ci_handles_none_and_values():
    assert _fmt_ci(None, None).strip() == "-"
    assert "0.400" in _fmt_ci(0.4, 0.6)
