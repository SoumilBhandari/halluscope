"""
AUROC evaluation harness — the credibility core of HalluScope.

Runs any detection method against HaluEval or TruthfulQA and reports AUROC.
The final table printed by main() is what goes into the README.

Usage:
    python eval.py --method baseline --dataset halueval
    python eval.py --method probe --dataset halueval --probe probe.pkl
    python eval.py --method semantic --dataset halueval
    python eval.py --all --dataset halueval   # run all three, print full table
"""

import argparse
import os
import time

import torch
from sklearn.metrics import roc_auc_score

import baseline as baseline_mod
import semantic_entropy as se_mod
from data import grade_truthfulqa, load_halueval_split, load_truthfulqa
from model import DEFAULT_MODEL, default_device, load_model


def auroc(scores, labels):
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return float("nan")


def evaluate_method(score_fn, records, model, tokenizer, device, max_n=None, verbose=True):
    """
    Calls score_fn(question, answer, model, tokenizer, device) for each record.
    Returns (auroc_float, scores_list, labels_list).
    """
    if max_n is not None:
        records = records[:max_n]

    scores, labels = [], []
    for i, rec in enumerate(records):
        if verbose and i % 20 == 0:
            print(f"  [{i}/{len(records)}]", flush=True)
        s = score_fn(rec["question"], rec["answer"], model, tokenizer, device)
        scores.append(s)
        labels.append(rec["label"])

    return auroc(scores, labels), scores, labels


def evaluate_on_truthfulqa(score_fn, model, tokenizer, device, max_n=None, verbose=True):
    """
    Generate answers with the model, grade them, then score with score_fn.
    Returns (auroc_float, scores_list, labels_list).
    """
    from model import generate

    questions = load_truthfulqa()
    if max_n is not None:
        questions = questions[:max_n]

    scores, labels = [], []
    for i, q in enumerate(questions):
        if verbose and i % 10 == 0:
            print(f"  [{i}/{len(questions)}]", flush=True)

        prompt = q["question"]
        result = generate(
            prompt, model, tokenizer, device,
            max_new_tokens=100, temperature=0, return_hidden_states=False,
        )
        model_answer = result["text"]
        label = grade_truthfulqa(model_answer, q["correct_answers"], q["incorrect_answers"])
        s = score_fn(prompt, model_answer, model, tokenizer, device)
        scores.append(s)
        labels.append(label)

    return auroc(scores, labels), scores, labels


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not (v != v) else "  N/A "


def print_table(rows):
    header = f"{'Method':<20} {'Dataset':<12} {'AUROC':>6}  Notes"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['method']:<20} {r['dataset']:<12} {_fmt(r['auroc']):>6}  {r['notes']}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline", "probe", "semantic", "all"], default="baseline")
    parser.add_argument("--dataset", choices=["halueval", "truthfulqa"], default="halueval")
    parser.add_argument("--probe", default="probe.pkl", help="path to saved probe (for --method probe)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_n", type=int, default=None, help="cap number of eval examples")
    parser.add_argument("--train_frac", type=float, default=0.8)
    args = parser.parse_args()

    device = args.device or default_device()
    print(f"Loading model {args.model} on {device}...")
    model, tokenizer = load_model(args.model, device)

    methods = ["baseline", "probe", "semantic"] if args.method == "all" else [args.method]
    table_rows = []

    # Prepare dataset
    if args.dataset == "halueval":
        _, test_records = load_halueval_split(train_frac=args.train_frac, max_n=args.max_n)
        print(f"HaluEval test set: {len(test_records)} records")
    else:
        test_records = None  # handled inside evaluate_on_truthfulqa

    for method in methods:
        print(f"\nEvaluating: {method} on {args.dataset}")
        t0 = time.time()

        if method == "baseline":
            score_fn = baseline_mod.score
            notes = "neg mean logprob"
        elif method == "probe":
            import probe as probe_mod
            if not os.path.exists(args.probe):
                print(f"  Probe file {args.probe} not found — run probe.py --train first.")
                table_rows.append({"method": "probe", "dataset": args.dataset, "auroc": float("nan"), "notes": "probe not trained"})
                continue
            scaler, clf, layer = probe_mod.load_probe(args.probe)
            import functools
            score_fn = functools.partial(probe_mod.score, scaler=scaler, clf=clf, layer=layer)
            notes = f"layer {layer}, last token"
        elif method == "semantic":
            nli_model, nli_tokenizer = se_mod.load_nli_model(device)
            import functools
            score_fn = functools.partial(se_mod.score, nli_model=nli_model, nli_tokenizer=nli_tokenizer)
            notes = "M=10 samples, DeBERTa NLI"

        if args.dataset == "halueval":
            auc, _, _ = evaluate_method(score_fn, test_records, model, tokenizer, device, verbose=True)
        else:
            auc, _, _ = evaluate_on_truthfulqa(score_fn, model, tokenizer, device, max_n=args.max_n)

        elapsed = time.time() - t0
        print(f"  AUROC: {auc:.4f}  ({elapsed:.0f}s)")
        table_rows.append({"method": method, "dataset": args.dataset, "auroc": auc, "notes": notes})

    print_table(table_rows)


if __name__ == "__main__":
    main()
