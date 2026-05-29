"""
Method 3 — hidden-state linear probe (the "aha" of the repo).

A logistic regression trained on the model's internal activations at a single
layer and position. One forward pass per answer at inference time.

Why this works: mid-to-late transformer layers encode factual confidence in
directions that a linear classifier can find. The model internally "knows"
when it's confabulating — this probe reads that signal.

Honest caveat: strong in-distribution (HaluEval train -> HaluEval test) but
drops on TruthfulQA (OOD). The probe learns the *style* of HaluEval
hallucinations, not universal lying.

The trained probe is saved as plain JSON (scaler statistics + logistic-
regression weights), so loading a probe never executes pickled code.

Usage:
    python probe.py --train                     # train on HaluEval, save probe.json
    python probe.py --sweep                     # find the best layer first
    python probe.py --score "q" "a"             # score a single q/a pair
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model import DEFAULT_MODEL, default_device, load_model


def extract_hidden(question, answer, model, tokenizer, device, layer=16):
    """
    One forward pass; returns the hidden state at `layer`, last token position.
    Shape: (D,) as a float32 numpy array.
    """
    prompt = question + " " + answer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer][0, -1, :].cpu().float().numpy()


def build_layer_features(records, model, tokenizer, device, layers, batch_size=16, verbose=True):
    """
    Batched extraction of last-token hidden states at several layers, with a
    single forward pass per batch. Returns ({layer: (N, D) float32}, y).

    The layer sweep uses this so it costs one forward pass per example, not one
    per candidate layer. The last *non-pad* token is selected per sequence, so
    the result is independent of the tokenizer's padding side.
    """
    feats = {layer: [] for layer in layers}
    labels = []
    for start in range(0, len(records), batch_size):
        if verbose and start % (batch_size * 8) == 0:
            print(f"  extracting [{start}/{len(records)}]", flush=True)
        batch = records[start : start + batch_size]
        prompts = [r["question"] + " " + r["answer"] for r in batch]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"]
        last_idx = mask.shape[1] - 1 - mask.flip(1).long().argmax(dim=1)  # last real token
        rows = torch.arange(mask.shape[0], device=mask.device)
        for layer in layers:
            feats[layer].append(out.hidden_states[layer][rows, last_idx].cpu().float())
        labels.extend(r["label"] for r in batch)
    return {layer: torch.cat(v).numpy() for layer, v in feats.items()}, np.array(labels)


def build_features(records, model, tokenizer, device, layer=16, batch_size=16, verbose=True):
    """Last-token hidden states at a single layer. Returns (X, y)."""
    feats, y = build_layer_features(records, model, tokenizer, device, [layer], batch_size, verbose)
    return feats[layer], y


def train_probe(records, model, tokenizer, device, layer=16, meta=None):
    """
    Fit StandardScaler + LogisticRegression on layer-`layer` features.
    Returns a probe dict (plain data — JSON-serializable).
    """
    print(f"Extracting features (layer {layer}) for {len(records)} records...")
    X, y = build_features(records, model, tokenizer, device, layer)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print("Fitting logistic regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(Xs, y)
    print(f"Train accuracy: {clf.score(Xs, y):.4f}")

    probe = {
        "layer": int(layer),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }
    if meta:
        probe.update(meta)
    return probe


def predict_proba(X, probe):
    """P(hallucinated) for a feature matrix X of shape (N, D). Pure numpy."""
    X = np.asarray(X, dtype=np.float64)
    z = (X - probe["mean"]) / probe["scale"]
    logits = z @ probe["coef"] + probe["intercept"]
    return 1.0 / (1.0 + np.exp(-logits))


def score(question, answer, model, tokenizer, device, probe):
    """
    Hallucination probability in [0, 1] (higher = more likely hallucinated).
    Shared scorer interface (probe bound via functools.partial in eval.py/app.py).
    """
    h = extract_hidden(question, answer, model, tokenizer, device, probe["layer"])
    return float(predict_proba(h[None, :], probe)[0])


def save_probe(probe, path="probe.json"):
    with open(path, "w") as f:
        json.dump(probe, f)
    print(f"Probe saved to {path}")


def load_probe(path="probe.json"):
    """Load a probe dict; array fields come back as numpy arrays."""
    with open(path) as f:
        probe = json.load(f)
    for key in ("mean", "scale", "coef"):
        probe[key] = np.asarray(probe[key], dtype=np.float64)
    return probe


def _best_layer(results, n_layers):
    """Highest-AUROC layer; ties break toward mid-network, where the
    factual-confidence signal concentrates."""
    best_auc = max(results.values())
    tied = [layer for layer, auc in results.items() if auc >= best_auc - 1e-9]
    mid = n_layers / 2
    return min(tied, key=lambda layer: abs(layer - mid))


def sweep_layers(train_records, val_records, model, tokenizer, device, layers=None):
    """
    Train a probe at each candidate layer and report validation AUROC. Features
    for all candidate layers are extracted in a single pass over the data, so
    the sweep costs one forward pass per example, not one per layer.
    Returns dict {layer: auroc}.
    """
    from sklearn.metrics import roc_auc_score

    n_layers = model.config.num_hidden_layers
    if layers is None:
        step = max(1, n_layers // 8)
        layers = list(range(step, n_layers, step))

    print(f"Extracting features at layers {layers} (train: {len(train_records)} records)...")
    X_train, y_train = build_layer_features(train_records, model, tokenizer, device, layers)
    print(f"Extracting features at layers {layers} (val: {len(val_records)} records)...")
    X_val, y_val = build_layer_features(val_records, model, tokenizer, device, layers)

    results = {}
    for layer in layers:
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(scaler.fit_transform(X_train[layer]), y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(scaler.transform(X_val[layer]))[:, 1])
        print(f"  layer {layer}: val AUROC {auc:.4f}")
        results[layer] = auc

    best = _best_layer(results, n_layers)
    spread = max(results.values()) - min(results.values())
    if spread < 0.02:
        print(
            f"\nWARNING: val AUROC varies by only {spread:.4f} across layers; "
            "the validation set is likely too small for the sweep to mean anything."
        )
    print(f"\nBest layer: {best} (val AUROC {results[best]:.4f})")
    return results


def _load_train_val(dataset, seed=42):
    """
    (train_records, val_records) for the chosen training dataset.

    For cross-dataset experiments: train on one dataset, then evaluate on the
    other with eval.py to measure OOD transfer. truthfulqa_mc gets a simple
    random train/val split, which is fine because the real test set is a
    *different* dataset.
    """
    if dataset == "halueval":
        from data import load_halueval_splits

        train, val, _ = load_halueval_splits()
        return train, val
    if dataset == "truthfulqa_mc":
        import random

        from data import load_truthfulqa_mc

        records = load_truthfulqa_mc()
        random.Random(seed).shuffle(records)
        n_val = max(1, len(records) // 6)
        return records[n_val:], records[:n_val]
    raise ValueError(f"unknown train dataset: {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--score", nargs=2, metavar=("QUESTION", "ANSWER"))
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--train_dataset", choices=["halueval", "truthfulqa_mc"], default="halueval",
                        help="dataset to train/sweep the probe on (evaluate on the other for OOD)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe", default="probe.json")
    parser.add_argument("--max_n", type=int, default=None, help="cap training records (debug)")
    args = parser.parse_args()

    device = args.device or default_device()

    if args.sweep or args.train:
        print(f"Loading model {args.model}...")
        model, tokenizer = load_model(args.model, device)
        train_records, val_records = _load_train_val(args.train_dataset)
        if args.max_n:
            train_records = train_records[: args.max_n]
            val_records = val_records[: max(2, args.max_n // 5)]

    if args.sweep:
        results = sweep_layers(train_records, val_records, model, tokenizer, device)
        best = _best_layer(results, model.config.num_hidden_layers)
        print(f"\nRun with --train --layer {best} to train the probe at the best layer.")

    elif args.train:
        from sklearn.metrics import roc_auc_score

        probe = train_probe(
            train_records, model, tokenizer, device, args.layer,
            meta={"model": args.model, "train_dataset": args.train_dataset},
        )
        X_val, y_val = build_features(val_records, model, tokenizer, device, args.layer, verbose=False)
        try:
            print(f"Validation AUROC: {roc_auc_score(y_val, predict_proba(X_val, probe)):.4f}")
        except ValueError:
            print("Validation AUROC: n/a (degenerate val set)")
        save_probe(probe, args.probe)

    elif args.score:
        question, answer = args.score
        if not os.path.exists(args.probe):
            print(f"Probe {args.probe} not found. Run --train first.")
        else:
            model, tokenizer = load_model(args.model, device)
            probe = load_probe(args.probe)
            print(f"Hallucination score: {score(question, answer, model, tokenizer, device, probe):.4f}")
    else:
        parser.print_help()
