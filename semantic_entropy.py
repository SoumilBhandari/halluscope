"""
Method 2 — semantic entropy (Farquhar et al., Nature 2024).

Samples M answers from the model, clusters them by bidirectional NLI entailment,
and computes entropy over cluster probabilities. Two answers in the same cluster
mean the model is saying "the same thing" semantically — high cluster entropy
means the model is genuinely uncertain about the *meaning*, not just the words.

This is the strongest single-pass detector (~AUROC 0.78–0.90) but requires
M+1 model passes (M samples + greedy display answer) and N*(N-1) NLI calls
per question. The compute cost is the price of catching confident hallucinations
that the logprob baseline misses.

Usage:
    python semantic_entropy.py --question "..." [--M 10]
"""

import argparse
import math

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model import DEFAULT_MODEL, load_model

NLI_MODEL_NAME = "microsoft/deberta-large-mnli"
ENTAIL_LABEL = 2  # DeBERTa MNLI label order: 0=contradiction, 1=neutral, 2=entailment


def load_nli_model(device):
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device).eval()
    return nli_model, nli_tokenizer


@torch.no_grad()
def _nli_label(premise, hypothesis, nli_model, nli_tokenizer, device):
    """Returns argmax label: 0=contradiction, 1=neutral, 2=entailment."""
    enc = nli_tokenizer(
        premise, hypothesis,
        return_tensors="pt", truncation=True, max_length=512,
    ).to(device)
    logits = nli_model(**enc).logits[0]
    return int(logits.argmax().item())


def entails(a, b, question, nli_model, nli_tokenizer, device):
    """True if a entails b (question prepended as context to both)."""
    premise = question + " " + a
    hypothesis = question + " " + b
    return _nli_label(premise, hypothesis, nli_model, nli_tokenizer, device) == ENTAIL_LABEL


def cluster_answers(answers, question, nli_model, nli_tokenizer, device):
    """
    Greedy clustering: answers i and j share a cluster iff
    entails(i→j) AND entails(j→i).
    Returns list of cluster ids (int), one per answer.
    """
    n = len(answers)
    cluster_id = list(range(n))  # start: each answer is its own cluster

    # Union-find helpers
    def find(x):
        while cluster_id[x] != x:
            cluster_id[x] = cluster_id[cluster_id[x]]
            x = cluster_id[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            cluster_id[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if (
                entails(answers[i], answers[j], question, nli_model, nli_tokenizer, device)
                and entails(answers[j], answers[i], question, nli_model, nli_tokenizer, device)
            ):
                union(i, j)

    return [find(i) for i in range(n)]


def _semantic_entropy(cluster_ids):
    """H = -sum_k p(C_k) log p(C_k) over unique clusters."""
    n = len(cluster_ids)
    counts = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1
    eps = 1e-10
    return -sum((cnt / n) * math.log(cnt / n + eps) for cnt in counts.values())


@torch.no_grad()
def sample_answers(question, model, tokenizer, device, M=10):
    """Sample M answers (diverse) from the model."""
    from model import generate
    results = generate(
        question, model, tokenizer, device,
        max_new_tokens=150,
        temperature=1.0, top_p=0.9, top_k=50,
        num_return_sequences=M,
        return_logprobs=False, return_hidden_states=False,
    )
    if isinstance(results, dict):
        results = [results]
    return [r["text"] for r in results]


def score(question, answer, model, tokenizer, device, nli_model=None, nli_tokenizer=None, M=10):
    """
    Hallucination score via semantic entropy. `answer` is ignored — the method
    samples fresh from the model to measure uncertainty. Kept in the signature
    for interface compatibility with eval.py.

    Higher SE = more uncertain = more likely hallucinated.
    """
    if nli_model is None or nli_tokenizer is None:
        raise ValueError("Pass nli_model and nli_tokenizer (loaded once, reused across calls).")

    answers = sample_answers(question, model, tokenizer, device, M=M)
    cluster_ids = cluster_answers(answers, question, nli_model, nli_tokenizer, device)
    return _semantic_entropy(cluster_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LLM ({args.model})...")
    model, tokenizer = load_model(args.model, device)
    print(f"Loading NLI model ({NLI_MODEL_NAME})...")
    nli_model, nli_tokenizer = load_nli_model(device)

    answers = sample_answers(args.question, model, tokenizer, device, M=args.M)
    print(f"\nSampled {len(answers)} answers:")
    for i, a in enumerate(answers):
        print(f"  [{i}] {a[:120]}")

    cluster_ids = cluster_answers(answers, args.question, nli_model, nli_tokenizer, device)
    se = _semantic_entropy(cluster_ids)

    print(f"\nCluster assignments: {cluster_ids}")
    n_clusters = len(set(cluster_ids))
    print(f"Unique clusters: {n_clusters} / {len(answers)}")
    print(f"Semantic entropy: {se:.4f}")
