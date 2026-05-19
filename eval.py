"""
AUROC evaluation harness — the credibility core of HalluScope.

Runs a detection method against HaluEval or TruthfulQA and reports AUROC.
The table printed by main() is what goes into the README.

Note on method/dataset pairings:
  - HaluEval supplies a fixed (question, answer) pair per record, so the
    baseline and the probe can score it directly.
  - Semantic entropy measures the *model's* uncertainty by sampling fresh
    answers; it cannot judge a pre-written answer. It is therefore only
    meaningful on TruthfulQA, where the model generates the answer being
    graded. Requesting `semantic` on `halueval` is reported as N/A.

Usage:
    python eval.py --method baseline --dataset halueval
    python eval.py --method probe --dataset halueval --probe probe.json
    python eval.py --method semantic --dataset truthfulqa
    python eval.py --all --dataset truthfulqa --out results.json
"""

import argparse
import functools
import json
import os
import time

from sklearn.metrics import roc_auc_score

import baseline as baseline_mod
import semantic_entropy as se_mod
from data import grade_truthfulqa, load_halueval_splits, load_truthfulqa
from model import DEFAULT_MODEL, default_device, generate, load_model


def auroc(scores, labels):
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return float("nan")


def evaluate_halueval(score_fn, records, model, tokenizer, device, verbose=True):
    """Score each (question, answer) record. Returns (auroc, scores, labels)."""
    scores, labels = [], []
    for i, rec in enumerate(records):
        if verbose and i % 20 == 0:
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


def print_table(rows):
    header = f"{'Method':<20} {'Dataset':<12} {'AUROC':>7} {'N':>6}  Notes"
    print("\n" + header)
    print("-" * max(len(header), 64))
    for r in rows:
        print(f"{r['method']:<20} {r['dataset']:<12} {_fmt(r['auroc']):>7} {r['n']:>6}  {r['notes']}")
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
    parser.add_argument("--dataset", choices=["halueval", "truthfulqa"], default="halueval")
    parser.add_argument("--probe", default="probe.json", help="probe file for --method probe")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_n", type=int, default=None, help="cap the number of eval examples")
    parser.add_argument("--out", default=None, help="optional path to write results JSON")
    args = parser.parse_args()

    device = args.device or default_device()
    print(f"Loading model {args.model} on {device}...")
    model, tokenizer = load_model(args.model, device)

    methods = ["baseline", "probe", "semantic"] if (args.all or args.method == "all") else [args.method]

    if args.dataset == "halueval":
        _, _, test_records = load_halueval_splits()
        if args.max_n:
            test_records = test_records[: args.max_n]
        n_examples = len(test_records)
        print(f"HaluEval test set: {n_examples} records")
        questions = None
    else:
        questions = load_truthfulqa()
        if args.max_n:
            questions = questions[: args.max_n]
        n_examples = len(questions)
        print(f"TruthfulQA: {n_examples} questions")
        print("Generating model answers (once, shared across methods)...")
        graded = generate_truthfulqa_answers(questions, model, tokenizer, device)
        test_records = None

    table_rows = []
    for method in methods:
        # Semantic entropy needs model-generated answers; it cannot score the
        # fixed answers HaluEval provides.
        if method == "semantic" and args.dataset == "halueval":
            print(
                "\nSkipping semantic entropy on HaluEval: it measures the model's "
                "uncertainty about a question, not a pre-written answer. "
                "Use --dataset truthfulqa."
            )
            table_rows.append({
                "method": "semantic", "dataset": "halueval", "n": 0,
                "auroc": float("nan"), "notes": "N/A - SE needs generated answers; use truthfulqa",
            })
            continue

        print(f"\nEvaluating: {method} on {args.dataset}")
        score_fn, notes = build_score_fn(method, model, tokenizer, device, args.probe)
        if score_fn is None:
            print(f"  {notes}")
            table_rows.append({
                "method": method, "dataset": args.dataset, "n": 0,
                "auroc": float("nan"), "notes": notes,
            })
            continue

        t0 = time.time()
        if args.dataset == "halueval":
            auc, _, _ = evaluate_halueval(score_fn, test_records, model, tokenizer, device)
        else:
            auc, _, _ = evaluate_truthfulqa(score_fn, graded, model, tokenizer, device)
        elapsed = time.time() - t0
        print(f"  AUROC: {auc:.4f}  ({elapsed:.0f}s)")
        table_rows.append({
            "method": method, "dataset": args.dataset, "n": n_examples,
            "auroc": auc, "notes": notes,
        })

    print_table(table_rows)

    if args.out:
        payload = {
            "model": args.model,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": [{**r, "auroc": None if r["auroc"] != r["auroc"] else r["auroc"]} for r in table_rows],
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
