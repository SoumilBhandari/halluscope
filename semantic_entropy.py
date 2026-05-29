"""
Method 2 — semantic entropy (Farquhar et al., Nature 2024).

Samples M answers from the model, clusters them by bidirectional NLI
entailment, and computes entropy over the cluster probabilities. Two answers
in the same cluster mean the model is saying "the same thing" semantically;
high cluster entropy means the model is genuinely uncertain about the
*meaning*, not just the wording.

This catches confident hallucinations the log-prob baseline misses, at the
cost of M+1 model passes (M samples + the greedy display answer) and N*(N-1)
NLI calls per question.

Usage:
    python semantic_entropy.py --question "..." [--M 10]
"""

import argparse
import math
from collections import Counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model import DEFAULT_MODEL, default_device, load_model

NLI_MODEL_NAME = "microsoft/deberta-large-mnli"
ENTAIL_LABEL = 2  # DeBERTa MNLI label order: 0=contradiction, 1=neutral, 2=entailment


def load_nli_model(device):
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device).eval()
    return nli_model, nli_tokenizer


@torch.no_grad()
def _nli_label(premise, hypothesis, nli_model, nli_tokenizer, device):
    """Returns the argmax label: 0=contradiction, 1=neutral, 2=entailment."""
    enc = nli_tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    return int(nli_model(**enc).logits[0].argmax().item())


def entails(a, b, question, nli_model, nli_tokenizer, device):
    """True if answer `a` entails answer `b` (question prepended as shared context)."""
    premise = question + " " + a
    hypothesis = question + " " + b
    return _nli_label(premise, hypothesis, nli_model, nli_tokenizer, device) == ENTAIL_LABEL


def cluster_by_equivalence(n, equivalent):
    """
    Union-find clustering of n items. `equivalent(i, j)` returns whether items
    i and j belong in the same cluster. Returns a list of cluster ids.

    Pure function — no model needed — so the clustering logic is unit-testable.
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if equivalent(i, j):
                union(i, j)

    return [find(i) for i in range(n)]


def cluster_answers(answers, question, nli_model, nli_tokenizer, device):
    """
    Cluster answers by bidirectional NLI entailment: answers i and j share a
    cluster iff each entails the other. Returns a cluster id per answer.
    """
    def equivalent(i, j):
        return (
            entails(answers[i], answers[j], question, nli_model, nli_tokenizer, device)
            and entails(answers[j], answers[i], question, nli_model, nli_tokenizer, device)
        )

    return cluster_by_equivalence(len(answers), equivalent)


def cluster_entropy(cluster_ids):
    """Shannon entropy over cluster probabilities: H = -sum_k p(C_k) log p(C_k)."""
    n = len(cluster_ids)
    if n == 0:
        return 0.0
    counts = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1
    return -sum((cnt / n) * math.log(cnt / n) for cnt in counts.values())


@torch.no_grad()
def sample_answers(question, model, tokenizer, device, M=10, temperature=1.0, seed=0):
    """Sample M diverse answers from the model. Seeded for reproducibility."""
    from model import generate

    if seed is not None:
        torch.manual_seed(seed)
    results = generate(
        question, model, tokenizer, device,
        max_new_tokens=150, temperature=temperature, top_p=0.9, top_k=50,
        num_return_sequences=M, return_logprobs=False, return_hidden_states=False,
    )
    if isinstance(results, dict):
        results = [results]
    return [r["text"] for r in results]


def score(question, answer, model, tokenizer, device, nli_model=None, nli_tokenizer=None,
          M=10, temperature=1.0, seed=0):
    """
    Hallucination score via semantic entropy. `answer` is ignored: the method
    samples fresh answers from the model to measure its uncertainty. The
    parameter is kept for interface compatibility with eval.py's scorers.

    Higher SE = more semantic disagreement across samples = more uncertain.
    Raising `temperature` increases sample diversity, which matters for
    heavily-aligned models that otherwise answer near-identically every time.
    """
    if nli_model is None or nli_tokenizer is None:
        raise ValueError("Pass nli_model and nli_tokenizer (loaded once, reused across calls).")

    answers = sample_answers(question, model, tokenizer, device, M=M, temperature=temperature, seed=seed)
    cluster_ids = cluster_answers(answers, question, nli_model, nli_tokenizer, device)
    return cluster_entropy(cluster_ids)


def diagnose(questions, model, tokenizer, device, nli_model, nli_tokenizer,
             M=10, temperature=1.0, seed=0):
    """
    Run SE over a list of questions and summarize sample diversity. If the
    model collapses most questions to 1-2 clusters, semantic entropy has almost
    nothing to measure — the likely cause of a chance-level AUROC.
    Returns (cluster_count_histogram, list_of_SE_values).
    """
    counts, ses = Counter(), []
    for i, q in enumerate(questions):
        answers = sample_answers(q, model, tokenizer, device, M=M, temperature=temperature, seed=seed)
        cluster_ids = cluster_answers(answers, q, nli_model, nli_tokenizer, device)
        k = len(set(cluster_ids))
        counts[k] += 1
        ses.append(cluster_entropy(cluster_ids))
        print(f"  [{i + 1}/{len(questions)}] clusters={k}/{M}  SE={ses[-1]:.3f}", flush=True)

    total = sum(counts.values())
    mean_clusters = sum(k * n for k, n in counts.items()) / total
    print(f"\nCluster-count distribution (clusters: #questions): {dict(sorted(counts.items()))}")
    print(f"Mean unique clusters: {mean_clusters:.2f} / {M}")
    print(f"Mean SE: {sum(ses) / len(ses):.3f}  (max possible {math.log(M):.3f})")
    return counts, ses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None, help="score a single question")
    parser.add_argument("--diagnose", type=int, default=0, metavar="N",
                        help="run SE over N TruthfulQA questions and report sample diversity")
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or default_device()

    if not args.question and not args.diagnose:
        parser.print_help()
        raise SystemExit("\nPass --question \"...\" or --diagnose N.")

    print(f"Loading LLM ({args.model})...")
    model, tokenizer = load_model(args.model, device)
    print(f"Loading NLI model ({NLI_MODEL_NAME})...")
    nli_model, nli_tokenizer = load_nli_model(device)

    if args.diagnose:
        from data import load_truthfulqa

        questions = [r["question"] for r in load_truthfulqa()[: args.diagnose]]
        print(f"\nDiagnosing sample diversity over {len(questions)} questions "
              f"(M={args.M}, temperature={args.temperature})...")
        diagnose(questions, model, tokenizer, device, nli_model, nli_tokenizer,
                 M=args.M, temperature=args.temperature)
    else:
        answers = sample_answers(args.question, model, tokenizer, device,
                                 M=args.M, temperature=args.temperature)
        print(f"\nSampled {len(answers)} answers:")
        for i, a in enumerate(answers):
            print(f"  [{i}] {a[:120]}")

        cluster_ids = cluster_answers(answers, args.question, nli_model, nli_tokenizer, device)
        print(f"\nCluster assignments: {cluster_ids}")
        print(f"Unique clusters: {len(set(cluster_ids))} / {len(answers)}")
        print(f"Semantic entropy: {cluster_entropy(cluster_ids):.4f}")
