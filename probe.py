"""
Method 3 — hidden-state linear probe (the "aha" of the repo).

A logistic regression trained on the model's internal activations at a single
layer and position. One forward pass per answer at inference time.

Why this works: mid-to-late transformer layers encode factual confidence in
directions that a linear classifier can find. The model internally "knows"
when it's confabulating — this probe reads that signal.

Honest caveat: strong in-distribution (HaluEval train → HaluEval test,
AUROC ~0.85–0.96) but drops to ~0.7–0.8 on TruthfulQA (OOD). The probe
learns the *style* of HaluEval hallucinations, not universal lying.

Usage:
    python probe.py --train                     # train on HaluEval, save probe.pkl
    python probe.py --sweep                     # find best layer first
    python probe.py --score "q" "a"             # score a single q/a pair
"""

import argparse
import os
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model import DEFAULT_MODEL, load_model


def extract_hidden(question, answer, model, tokenizer, device, layer=16):
    """
    One forward pass; returns the hidden state at `layer`, last token position.
    Shape: (D,) as float32 numpy array.
    """
    prompt = question + " " + answer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of L+1 tensors each (1, T, D)
    h = outputs.hidden_states[layer][0, -1, :]  # (D,)
    return h.cpu().float().numpy()


def build_features(records, model, tokenizer, device, layer=16, verbose=True):
    """Extract hidden-state features for a list of records."""
    X, y = [], []
    for i, rec in enumerate(records):
        if verbose and i % 100 == 0:
            print(f"  extracting [{i}/{len(records)}]", flush=True)
        h = extract_hidden(rec["question"], rec["answer"], model, tokenizer, device, layer)
        X.append(h)
        y.append(rec["label"])
    return np.stack(X), np.array(y)


def train_probe(records, model, tokenizer, device, layer=16):
    """
    Fit StandardScaler + LogisticRegression on hidden states extracted from records.
    Returns: (scaler, clf)
    """
    print(f"Extracting features (layer {layer}) for {len(records)} records...")
    X, y = build_features(records, model, tokenizer, device, layer)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Fitting logistic regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_scaled, y)
    train_acc = clf.score(X_scaled, y)
    print(f"Train accuracy: {train_acc:.4f}")
    return scaler, clf


def save_probe(scaler, clf, layer, path="probe.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf, "layer": layer}, f)
    print(f"Probe saved to {path}")


def load_probe(path="probe.pkl"):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["scaler"], d["clf"], d["layer"]


def score(question, answer, model, tokenizer, device, scaler, clf, layer=16):
    """
    Returns hallucination probability in [0, 1] (higher = more likely hallucinated).
    Shared scorer interface.
    """
    h = extract_hidden(question, answer, model, tokenizer, device, layer)
    h_scaled = scaler.transform(h.reshape(1, -1))
    return float(clf.predict_proba(h_scaled)[0, 1])


def sweep_layers(train_records, val_records, model, tokenizer, device, layers=None):
    """
    Train a probe at each layer and report validation AUROC.
    Returns dict: {layer: auroc}
    """
    from sklearn.metrics import roc_auc_score

    if layers is None:
        n_layers = model.config.num_hidden_layers
        step = max(1, n_layers // 8)
        layers = list(range(step, n_layers, step))

    results = {}
    for layer in layers:
        print(f"\nLayer {layer}:")
        X_train, y_train = build_features(train_records, model, tokenizer, device, layer, verbose=False)
        X_val, y_val = build_features(val_records, model, tokenizer, device, layer, verbose=False)
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(scaler.fit_transform(X_train), y_train)
        val_probs = clf.predict_proba(scaler.transform(X_val))[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        print(f"  val AUROC: {auc:.4f}")
        results[layer] = auc

    best = max(results, key=results.get)
    print(f"\nBest layer: {best} (AUROC {results[best]:.4f})")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--score", nargs=2, metavar=("QUESTION", "ANSWER"))
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe", default="probe.pkl")
    parser.add_argument("--max_n", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from data import load_halueval_split

    if args.sweep or args.train:
        print(f"Loading model {args.model}...")
        model, tokenizer = load_model(args.model, device)
        train_records, val_records = load_halueval_split(max_n=args.max_n)

    if args.sweep:
        results = sweep_layers(train_records, val_records, model, tokenizer, device)
        best_layer = max(results, key=results.get)
        print(f"\nRun with --train --layer {best_layer} to train the probe at the best layer.")

    elif args.train:
        scaler, clf = train_probe(train_records, model, tokenizer, device, args.layer)
        save_probe(scaler, clf, args.layer, args.probe)

    elif args.score:
        question, answer = args.score
        if not os.path.exists(args.probe):
            print(f"Probe {args.probe} not found. Run --train first.")
        else:
            model, tokenizer = load_model(args.model, device)
            scaler, clf, layer = load_probe(args.probe)
            s = score(question, answer, model, tokenizer, device, scaler, clf, layer)
            print(f"Hallucination score: {s:.4f}")
    else:
        parser.print_help()
