"""Unit tests for probe inference math, layer selection, and JSON round-trip."""

import numpy as np

from probe import _best_layer, load_probe, predict_proba, save_probe


def test_predict_proba_sigmoid_math():
    probe = {
        "mean": np.array([0.0, 0.0]), "scale": np.array([1.0, 1.0]),
        "coef": np.array([1.0, -1.0]), "intercept": 0.0, "layer": 4,
    }
    # x=[1,0] -> logit = 1 -> sigmoid(1)
    p = predict_proba(np.array([[1.0, 0.0]]), probe)
    assert np.isclose(p[0], 1.0 / (1.0 + np.exp(-1.0)))


def test_predict_proba_is_monotonic_and_bounded():
    probe = {"mean": np.zeros(1), "scale": np.ones(1), "coef": np.array([2.0]),
             "intercept": 0.0, "layer": 1}
    probs = predict_proba(np.array([[-2.0], [0.0], [2.0]]), probe)
    assert probs[0] < probs[1] < probs[2]
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_predict_proba_applies_the_scaler():
    # x = mean -> standardized to 0 -> logit 0 -> probability 0.5
    probe = {"mean": np.array([10.0]), "scale": np.array([2.0]),
             "coef": np.array([1.0]), "intercept": 0.0, "layer": 1}
    assert np.isclose(predict_proba(np.array([[10.0]]), probe)[0], 0.5)


def test_best_layer_picks_max_auroc():
    assert _best_layer({4: 0.7, 8: 0.9, 12: 0.8}, n_layers=24) == 8


def test_best_layer_breaks_ties_toward_middle():
    assert _best_layer({2: 0.9, 12: 0.9, 22: 0.9}, n_layers=24) == 12


def test_probe_json_round_trip(tmp_path):
    probe = {
        "layer": 7, "intercept": 0.25,
        "mean": [1.0, 2.0], "scale": [0.5, 0.5], "coef": [0.1, -0.3],
        "model": "test-model",
    }
    path = tmp_path / "probe.json"
    save_probe(probe, str(path))
    loaded = load_probe(str(path))
    assert loaded["layer"] == 7
    assert loaded["model"] == "test-model"
    assert np.allclose(loaded["coef"], [0.1, -0.3])
    assert isinstance(loaded["mean"], np.ndarray)
