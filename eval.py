"""
AUROC evaluation harness — the credibility core of HalluScope.

Runs a detection method against HaluEval or TruthfulQA and reports AUROC with a
bootstrap 95% confidence interval. The table printed by main() is what goes
into the README.

Datasets:
  - halueval       fixed (question, answer, label) pairs.
  - truthfulqa_mc  TruthfulQA multiple-choice as fixed pairs — clean
                   ground-truth labels, no ROUGE grading noise.
  - truthfulqa     generative: the model answers, the answer is graded by the
                   (noisier) ROUGE oracle, then scored. Required for semantic
                   entropy, which needs the model to generate the answer.

Semantic entropy only runs on the generative `truthfulqa` set; on the fixed
datasets eval.py reports it as N/A on purpose.

Usage:
    python eval.py --method baseline --dataset halueval
    python eval.py --method probe --dataset truthfulqa_mc --probe probe.json
    python eval.py --method semantic --dataset truthfulqa
    python eval.py --all --dataset truthfulqa_mc --out results.json
"""

import argparse
import functools
import json
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import baseline as baseline_mod
import semantic_entropy as se_mod
from data import grade_truthfulqa, load_halueval_splits, load_truthfulqa, load_truthfulqa_mc
from model import DEFAULT_MODEL, default_device, generate, load_model

# Datasets whose records are fixed (question, answer, label) triples.
FIXED_DATASETS = ("halueval", "truthfulqa_mc")


def auroc(scores, labels):
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return float("nan")


def bootstrap_ci(scores, labels, n_boot=1000, seed=0, alpha=0.05):
    """95% bootstrap CI for AUROC. Returns (lo, hi), or (nan, nan) if undefined."""
    if not scores:
        return float("nan"), float("nan")
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            aucs.append(roc_auc_score(labels[idx], scores[idx]))
        except ValueError:
            continue
    if not aucs:
        return float("nan"), float("nan")
    return (
        float(np.percentile(aucs, 100 * alpha / 2)),
        float(np.percentile(aucs, 100 * (1 - alpha / 2))),
    )


def evaluate_records(score_fn, records, model, tokenizer, device, verbose=True):
    """Score fixed (question, answer, label) records. Returns (auroc, scores, labels)."""
    scores, labels = [], []
    for i, rec in enumerate(records):
        if verbose and i % 50 == 0:
            print(f"  [{i}/{len(records)}]", flush=True)
        scores.append(score_fn(rec["question"], rec["answer"], model, tokenizer, device))
        labels.append(rec["label"])
    return auroc(scores, labels), scores, labels


def generate_truthfulqa_answers(questions, model, tokenizer, device, verbose=True):
    """
    Generate a greedy answer per question and grade it. Done once and shared
    across methods, so `--all` does not regenerate the same answers three times.
    Returns a list of {"question", "answer", "label"}.
    """
    graded = []
    for i, q in enumerate(questions):
        if verbose and i % 10 == 0:
            print(f"  generating [{i}/{len(questions)}]", flush=True)
        result = generate(
            q["question"], model, tokenizer, device,
            max_new_tokens=100, temperature=0, return_logprobs=False,
        )
        answer = result["text"].strip()
        label = grade_truthfulqa(answer, q["correct_answers"], q["incorrect_answers"])
        graded.append({"question": q["question"], "answer": answer, "label": label})
    return graded


def evaluate_truthfulqa(score_fn, graded, model, tokenizer, device, verbose=True):
    """Score pre-generated, pre-graded TruthfulQA answers. Returns (auroc, scores, labels)."""
    scores, labels = [], []
    for i, g in enumerate(graded):
        if verbose and i % 10 == 0:
            print(f"  [{i}/{len(graded)}]", flush=True)
        scores.append(score_fn(g["question"], g["answer"], model, tokenizer, device))
        labels.append(g["label"])
    return auroc(scores, labels), scores, labels


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and v == v else "  N/A "


def _fmt_ci(lo, hi):
    if isinstance(lo, float) and lo == lo and isinstance(hi, float) and hi == hi:
        return f"{lo:.3f}-{hi:.3f}"
    return "      -      "


def print_table(rows):
    header = f"{'Method':<20} {'Dataset':<14} {'AUROC':>7} {'95% CI':>14} {'N':>6}  Notes"
    print("\n" + header)
    print("-" * max(len(header), 80))
    for r in rows:
        ci = _fmt_ci(r.get("ci_lo"), r.get("ci_hi"))
        print(f"{r['method']:<20} {r['dataset']:<14} {_fmt(r['auroc']):>7} {ci:>14} {r['n']:>6}  {r['notes']}")
    print()


def build_score_fn(method, model, tokenizer, device, probe_path):
    """Returns (score_fn, notes), or (None, reason) if the method can't be built."""
    if method == "baseline":
        return baseline_mod.score, "neg mean token log-prob"

    if method == "probe":
        import probe as probe_mod
        if not os.path.exists(probe_path):
            return None, f"probe not found ({probe_path}) — run probe.py --train"
        probe = probe_mod.load_probe(probe_path)
        return functools.partial(probe_mod.score, probe=probe), f"layer {probe['layer']}, last token"

    if method == "semantic":
        nli_model, nli_tokenizer = se_mod.load_nli_model(device)
        score_fn = functools.partial(se_mod.score, nli_model=nli_model, nli_tokenizer=nli_tokenizer)
        return score_fn, "M=10 samples, DeBERTa NLI"

    return None, f"unknown method {method}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "probe", "semantic", "all"], default="baseline")
    parser.add_argument("--all", action="store_true", help="run every method applicable to the dataset")
    parser.add_argument(
        "--dataset", choices=["halueval", "truthfulqa", "truthfulqa_mc"], default="halueval"
    )
    parser.add_argument("--probe", default="probe.json", help="probe file for --method probe")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_n", type=int, default=None, help="cap the number of eval examples")
    parser.add_argument("--out", default=None, help="optional path to write/merge results JSON")
    args = parser.parse_args()

    device = args.device or default_device()
    print(f"Loading model {args.model} on {device}...")
    model, tokenizer = load_model(args.model, device)

    methods = ["baseline", "probe", "semantic"] if (args.all or args.method == "all") else [args.method]

    graded = records = None
    if args.dataset == "halueval":
        _, _, records = load_halueval_splits()
        if args.max_n:
            records = records[: args.max_n]
        n_examples = len(records)
        print(f"HaluEval test set: {n_examples} records")
    elif args.dataset == "truthfulqa_mc":
        records = load_truthfulqa_mc()
        if args.max_n:
            records = records[: args.max_n]
        n_examples = len(records)
        print(f"TruthfulQA-MC: {n_examples} records")
    else:  # generative truthfulqa
        questions = load_truthfulqa()
        if args.max_n:
            questions = questions[: args.max_n]
        n_examples = len(questions)
        print(f"TruthfulQA: {n_examples} questions")
        print("Generating model answers (once, shared across methods)...")
        graded = generate_truthfulqa_answers(questions, model, tokenizer, device)

    table_rows = []
    for method in methods:
        # Semantic entropy needs model-generated answers; it can't score the
        # fixed answers the non-generative datasets supply.
        if method == "semantic" and args.dataset in FIXED_DATASETS:
            print(
                f"\nSkipping semantic entropy on {args.dataset}: it measures the model's "
                "uncertainty about a question, not a pre-written answer. Use --dataset truthfulqa."
            )
            table_rows.append({
                "method": "semantic", "dataset": args.dataset, "n": 0,
                "auroc": float("nan"), "ci_lo": None, "ci_hi": None,
                "notes": "N/A - SE needs generated answers; use truthfulqa",
            })
            continue

        print(f"\nEvaluating: {method} on {args.dataset}")
        score_fn, notes = build_score_fn(method, model, tokenizer, device, args.probe)
        if score_fn is None:
            print(f"  {notes}")
            table_rows.append({
                "method": method, "dataset": args.dataset, "n": 0,
                "auroc": float("nan"), "ci_lo": None, "ci_hi": None, "notes": notes,
            })
            continue

        t0 = time.time()
        if args.dataset in FIXED_DATASETS:
            auc, scores, labels = evaluate_records(score_fn, records, model, tokenizer, device)
        else:
            auc, scores, labels = evaluate_truthfulqa(score_fn, graded, model, tokenizer, device)
        ci_lo, ci_hi = bootstrap_ci(scores, labels)
        elapsed = time.time() - t0
        print(f"  AUROC: {auc:.4f}  [95% CI {ci_lo:.3f}-{ci_hi:.3f}]  ({elapsed:.0f}s)")
        table_rows.append({
            "method": method, "dataset": args.dataset, "n": n_examples,
            "auroc": auc, "ci_lo": ci_lo, "ci_hi": ci_hi, "notes": notes,
        })

    print_table(table_rows)

    if args.out:
        def _clean(r):
            def nz(x):
                return None if (x is None or x != x) else x
            return {**r, "auroc": nz(r["auroc"]), "ci_lo": nz(r.get("ci_lo")), "ci_hi": nz(r.get("ci_hi"))}

        rows = [_clean(r) for r in table_rows]
        merged = {}
        if os.path.exists(args.out):
            with open(args.out) as f:
                merged = {(r["method"], r["dataset"]): r for r in json.load(f).get("rows", [])}
        for r in rows:
            merged[(r["method"], r["dataset"])] = r
        payload = {
            "model": args.model,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": list(merged.values()),
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
