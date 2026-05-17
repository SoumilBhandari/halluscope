"""
Dataset loaders for HaluEval and TruthfulQA.

HaluEval  — contrastive QA pairs (question, right_answer, hallucinated_answer).
            Each row yields two labeled records: label=0 (correct), label=1 (hallucinated).
            Used to train the probe and as the primary eval dataset.

TruthfulQA — 817 adversarial questions with correct + incorrect answer lists.
             The model generates an answer; grade_truthfulqa() labels it 0/1
             by ROUGE-1 F1 comparison against the ground-truth answer lists.
             Used only for eval (no train split) — the OOD test for the probe.
"""

import random

from datasets import load_dataset
from rouge_score import rouge_scorer as rouge_lib


def load_halueval(split="train", max_n=None, seed=42):
    """
    Returns list of {"question": str, "answer": str, "label": int} dicts.
    label=0 correct, label=1 hallucinated.
    Each HaluEval row produces two records.
    """
    ds = load_dataset("pminervini/HaluEval", "qa")

    # Map split names: HaluEval uses "data" for train
    split_map = {"train": "data", "validation": "data", "test": "data"}
    actual_split = split_map.get(split, split)

    if actual_split not in ds:
        actual_split = list(ds.keys())[0]

    rows = ds[actual_split]
    records = []
    for row in rows:
        q = row["question"]
        records.append({"question": q, "answer": row["right_answer"], "label": 0})
        records.append({"question": q, "answer": row["hallucinated_answer"], "label": 1})

    rng = random.Random(seed)
    rng.shuffle(records)

    if max_n is not None:
        records = records[:max_n]

    return records


def load_halueval_split(train_frac=0.8, max_n=None, seed=42):
    """Returns (train_records, test_records) from HaluEval with question-disjoint split."""
    ds = load_dataset("pminervini/HaluEval", "qa")
    actual_split = list(ds.keys())[0]
    rows = list(ds[actual_split])

    rng = random.Random(seed)
    rng.shuffle(rows)

    if max_n is not None:
        rows = rows[: max_n // 2]  # //2 because each row → 2 records

    n_train = int(len(rows) * train_frac)
    train_rows, test_rows = rows[:n_train], rows[n_train:]

    def rows_to_records(r):
        out = []
        for row in r:
            q = row["question"]
            out.append({"question": q, "answer": row["right_answer"], "label": 0})
            out.append({"question": q, "answer": row["hallucinated_answer"], "label": 1})
        return out

    return rows_to_records(train_rows), rows_to_records(test_rows)


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
    train, test = load_halueval_split(max_n=100)
    print(f"  train: {len(train)} records, test: {len(test)} records")
    print(f"  sample: {train[0]}")

    print("\nLoading TruthfulQA...")
    tqa = load_truthfulqa()
    print(f"  {len(tqa)} questions")
    print(f"  sample question: {tqa[0]['question']}")

    label = grade_truthfulqa(
        "Paris is the capital of France.",
        tqa[0]["correct_answers"],
        tqa[0]["incorrect_answers"],
    )
    print(f"  grade check (Paris/France): label={label}")
