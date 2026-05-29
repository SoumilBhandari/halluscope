"""
Method 4 — Semantic Entropy Probe (Kossen et al. 2024).

Semantic entropy (Method 2) is a strong uncertainty signal but expensive:
M samples + O(M^2) NLI calls per question. The Semantic Entropy Probe trains a
linear probe to *predict* semantic entropy from the hidden state of a single
greedy generation — SE-like signal at probe cost (one forward pass, no sampling
at inference).

Build pipeline, per TruthfulQA question:
  1. greedy-generate an answer and grade it (correct / hallucinated);
  2. record the layer-L last-token hidden state of (question + answer);
  3. compute semantic entropy by sampling — the (expensive) training target.
Binarize SE at its median, then fit logistic regression: hidden state -> high-SE.
Held-out questions are scored by the probe's P(high-SE) and, like every other
method, evaluated by AUROC against the correctness label. For a fair side-by-
side, the same split also reports raw-SE AUROC (the Method-2 signal) and a
direct probe trained to predict correctness from the same hidden states.

Usage:
    python se_probe.py --build --n 250 --layer 20 --out results.json
"""

import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import semantic_entropy as se_mod
from data import grade_truthfulqa, load_truthfulqa
from model import DEFAULT_MODEL, default_device, generate, load_model
from probe import extract_hidden


def build_examples(questions, model, tokenizer, device, nli_model, nli_tokenizer,
                   layer=20, M=10, temperature=1.0, seed=0, verbose=True):
    """
    Per question: the greedy answer's hidden state, its semantic entropy, and
    its correctness label. Returns (H [n, D], SE [n], labels [n]).
    """
    H, SE, labels = [], [], []
    for i, q in enumerate(questions):
        answer = generate(
            q["question"], model, tokenizer, device,
            max_new_tokens=100, temperature=0, return_logprobs=False,
        )["text"].strip()
        H.append(extract_hidden(q["question"], answer, model, tokenizer, device, layer))
        samples = se_mod.sample_answers(
            q["question"], model, tokenizer, device, M=M, temperature=temperature, seed=seed
        )
        clusters = se_mod.cluster_answers(samples, q["question"], nli_model, nli_tokenizer, device)
        SE.append(se_mod.cluster_entropy(clusters))
        labels.append(grade_truthfulqa(answer, q["correct_answers"], q["incorrect_answers"]))
        if verbose and (i % 10 == 0 or i == len(questions) - 1):
            print(f"  [{i + 1}/{len(questions)}] SE={SE[-1]:.3f} label={labels[-1]}", flush=True)
    return np.stack(H), np.array(SE), np.array(labels)


def evaluate_sep(H, SE, labels, train_frac=0.6, seed=0):
    """
    Pure evaluation (no model). On a train/test split:
      - sep_auroc:    probe predicts high-SE from hidden states -> AUROC vs correctness
      - direct_auroc: probe predicts correctness directly from hidden states
      - se_auroc:     raw semantic entropy -> AUROC vs correctness (Method-2 signal)
    All three are measured on the same held-out split for a fair comparison.
    """
    H = np.asarray(H, dtype=np.float64)
    SE = np.asarray(SE, dtype=np.float64)
    labels = np.asarray(labels)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    cut = int(len(labels) * train_frac)
    tr, te = idx[:cut], idx[cut:]

    def probe_auroc(train_targets):
        if len(set(train_targets)) < 2 or len(set(labels[te])) < 2:
            return float("nan")
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(scaler.fit_transform(H[tr]), train_targets)
        scores = clf.predict_proba(scaler.transform(H[te]))[:, 1]
        return float(roc_auc_score(labels[te], scores))

    threshold = float(np.median(SE[tr]))
    sep_auroc = probe_auroc((SE[tr] > threshold).astype(int))
    direct_auroc = probe_auroc(labels[tr])
    try:
        se_auroc = float(roc_auc_score(labels[te], SE[te]))
    except ValueError:
        se_auroc = float("nan")

    return {
        "sep_auroc": sep_auroc,
        "direct_auroc": direct_auroc,
        "se_auroc": se_auroc,
        "n_test": int(len(te)),
        "se_threshold": threshold,
    }


def _merge_out(path, rows, model_name):
    merged = {}
    if os.path.exists(path):
        with open(path) as f:
            merged = {(r["method"], r["dataset"]): r for r in json.load(f).get("rows", [])}
    for r in rows:
        merged[(r["method"], r["dataset"])] = r
    payload = {"model": model_name, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "rows": list(merged.values())}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="build the SEP dataset and evaluate")
    parser.add_argument("--n", type=int, default=250, help="number of TruthfulQA questions")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="merge results into this JSON (like eval.py)")
    args = parser.parse_args()

    if not args.build:
        parser.print_help()
        raise SystemExit

    device = args.device or default_device()
    print(f"Loading model {args.model} on {device}...")
    model, tokenizer = load_model(args.model, device)
    print("Loading NLI model...")
    nli_model, nli_tokenizer = se_mod.load_nli_model(device)

    questions = load_truthfulqa()[: args.n]
    print(f"Building SEP examples for {len(questions)} TruthfulQA questions "
          f"(layer {args.layer}, M={args.M}, temp={args.temperature})...")
    t0 = time.time()
    H, SE, labels = build_examples(
        questions, model, tokenizer, device, nli_model, nli_tokenizer,
        layer=args.layer, M=args.M, temperature=args.temperature,
    )
    res = evaluate_sep(H, SE, labels)
    elapsed = time.time() - t0

    print(f"\nSemantic Entropy Probe   AUROC: {res['sep_auroc']:.4f}  (n_test={res['n_test']})")
    print(f"Raw semantic entropy     AUROC: {res['se_auroc']:.4f}  (same split)")
    print(f"Direct correctness probe AUROC: {res['direct_auroc']:.4f}  (same split, for comparison)")
    print(f"({elapsed:.0f}s, layer {args.layer}, M={args.M}, temp={args.temperature})")

    if args.out:
        def nz(x):
            return None if (x is None or x != x) else x

        rows = [
            {"method": "se_probe", "dataset": "truthfulqa", "n": res["n_test"],
             "auroc": nz(res["sep_auroc"]), "ci_lo": None, "ci_hi": None,
             "notes": f"predicts SE from layer {args.layer}; one forward pass"},
            {"method": "semantic", "dataset": "truthfulqa", "n": res["n_test"],
             "auroc": nz(res["se_auroc"]), "ci_lo": None, "ci_hi": None,
             "notes": f"M={args.M} samples, temp={args.temperature}"},
        ]
        _merge_out(args.out, rows, args.model)
        print(f"Results merged into {args.out}")
