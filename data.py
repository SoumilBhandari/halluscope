"""
Dataset loaders for HaluEval and TruthfulQA.

HaluEval  — contrastive QA pairs (question, right_answer, hallucinated_answer).
            Each row yields two labeled records: label=0 (correct), label=1
            (hallucinated). Split question-disjoint into train/val/test;
            train builds the probe, val tunes the layer, test is held out.

TruthfulQA — 817 adversarial questions with correct + incorrect answer lists.
             The model generates an answer; grade_truthfulqa() labels it 0/1
             by ROUGE-1 F1 comparison against the answer lists. Eval-only —
             the out-of-distribution test for the probe.
"""

import random

from datasets import load_dataset
from rouge_score import rouge_scorer as rouge_lib


def load_halueval_splits(train_frac=0.7, val_frac=0.1, seed=42):
    """
    Returns (train, val, test) record lists from HaluEval QA.

    Each record is {"question": str, "answer": str, "label": int}
    (label 0 = correct, 1 = hallucinated).

    The split is computed on the *full* shuffled dataset and is determined
    entirely by `seed`. It never depends on how many records a caller later
    consumes, so train/val/test stay disjoint no matter what — this fixes the
    silent leakage that arose when training and evaluation passed different
    caps and recomputed the split independently.

    The split is question-disjoint: each HaluEval row (one question with a
    correct and a hallucinated answer) goes wholly into one split before being
    expanded into its two labeled records.
    """
    ds = load_dataset("pminervini/HaluEval", "qa")
    rows = list(ds[list(ds.keys())[0]])

    rng = random.Random(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    def to_records(rs):
        out = []
        for row in rs:
            q = row["question"]
            out.append({"question": q, "answer": row["right_answer"], "label": 0})
            out.append({"question": q, "answer": row["hallucinated_answer"], "label": 1})
        return out

    return to_records(train_rows), to_records(val_rows), to_records(test_rows)


def load_truthfulqa():
    """
    Returns list of {"question": str, "correct_answers": list[str],
                     "incorrect_answers": list[str]} dicts.
    """
    ds = load_dataset("truthful_qa", "generation")
    split = "validation" if "validation" in ds else list(ds.keys())[0]
    records = []
    for row in ds[split]:
        records.append(
            {
                "question": row["question"],
                "correct_answers": row["correct_answers"],
                "incorrect_answers": row["incorrect_answers"],
            }
        )
    return records


def load_truthfulqa_mc():
    """
    TruthfulQA multiple-choice (mc1) as fixed (question, answer, label) records
    with clean ground-truth labels — no ROUGE/generation grading, so the labels
    carry no oracle noise. label 0 = a correct choice, 1 = an incorrect
    (hallucinated) choice. Each question contributes one correct + several
    incorrect choices.
    """
    ds = load_dataset("truthful_qa", "multiple_choice")
    split = "validation" if "validation" in ds else list(ds.keys())[0]
    records = []
    for row in ds[split]:
        mc1 = row["mc1_targets"]
        for choice, correct in zip(mc1["choices"], mc1["labels"], strict=True):
            records.append(
                {"question": row["question"], "answer": choice, "label": 0 if correct == 1 else 1}
            )
    return records


_scorer = None


def grade_truthfulqa(model_answer, correct_answers, incorrect_answers):
    """
    Labels a model-generated answer as correct (0) or hallucinated (1).
    Uses ROUGE-1 F1: if the answer is closer to correct_answers than to
    incorrect_answers, label=0; otherwise label=1.
    """
    global _scorer
    if _scorer is None:
        _scorer = rouge_lib.RougeScorer(["rouge1"], use_stemmer=True)

    def max_rouge(answer, references):
        if not references:
            return 0.0
        return max(_scorer.score(ref, answer)["rouge1"].fmeasure for ref in references)

    correct_score = max_rouge(model_answer, correct_answers)
    incorrect_score = max_rouge(model_answer, incorrect_answers)
    return 0 if correct_score >= incorrect_score else 1


if __name__ == "__main__":
    print("Loading HaluEval...")
    train, val, test = load_halueval_splits()
    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)} records")
    print(f"  sample: {train[0]}")

    print("\nLoading TruthfulQA...")
    tqa = load_truthfulqa()
    print(f"  {len(tqa)} questions")
    q0 = tqa[0]
    print(f"  sample question: {q0['question']}")

    # Sanity-check the ROUGE oracle against its own ground-truth answers.
    good = grade_truthfulqa(q0["correct_answers"][0], q0["correct_answers"], q0["incorrect_answers"])
    bad = grade_truthfulqa(q0["incorrect_answers"][0], q0["correct_answers"], q0["incorrect_answers"])
    print(f"  grade oracle: correct -> {good} (expect 0), incorrect -> {bad} (expect 1)")
