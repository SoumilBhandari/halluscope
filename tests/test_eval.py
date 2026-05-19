"""Unit tests for the eval harness helpers (no model)."""

import math

from eval import _fmt, auroc


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
