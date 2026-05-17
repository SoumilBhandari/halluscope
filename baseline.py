"""
Method 1 — token log-probability baseline.

Scores an answer by how surprised the model is, token by token.
Intuition: hallucinated text is often generated with lower confidence.

Weakness (documented honestly): conflates *lexical* uncertainty with *factual*
uncertainty. A confidently wrong answer — the canonical hallucination — scores
low (looks "certain") and evades this detector. AUROC ~0.65–0.75 on HaluEval.
"""

import argparse
import os

import torch
import torch.nn.functional as F

from model import DEFAULT_MODEL, default_device, forward_hidden, load_model


@torch.no_grad()
def logprobs_for_answer(question, answer, model, tokenizer, device):
    """
    Returns per-token log-probs for the *answer* tokens only, conditioned on question.
    We tokenize question+answer, run a forward pass, then slice off the answer portion.
    """
    q_ids = tokenizer(question, return_tensors="pt")["input_ids"]
    qa_ids = tokenizer(question + " " + answer, return_tensors="pt")["input_ids"].to(device)

    outputs = model(qa_ids)
    logits = outputs.logits[0]  # (T, V)

    log_probs = F.log_softmax(logits, dim=-1)  # (T, V)

    # Shift: logits[t] predicts token[t+1]
    q_len = q_ids.shape[1]
    # Answer token indices in the full sequence
    ans_start = q_len  # first answer token position in qa_ids
    ans_ids = qa_ids[0, ans_start:]  # the actual answer token ids

    # log_probs[ans_start-1 : ans_start-1+len(ans_ids)] predict ans_ids
    pred_start = ans_start - 1
    pred_logp = log_probs[pred_start : pred_start + len(ans_ids), :]
    token_logp = pred_logp[range(len(ans_ids)), ans_ids]
    return token_logp


@torch.no_grad()
def predictive_entropy(question, answer, model, tokenizer, device):
    """
    Mean per-token predictive entropy: -sum_v p(v|context) log p(v|context).
    Averaged over answer tokens. Higher = more uncertain.
    """
    q_ids = tokenizer(question, return_tensors="pt")["input_ids"]
    qa_ids = tokenizer(question + " " + answer, return_tensors="pt")["input_ids"].to(device)

    outputs = model(qa_ids)
    logits = outputs.logits[0]  # (T, V)

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # (T,)

    q_len = q_ids.shape[1]
    ans_entropy = entropy[q_len - 1 :]  # align with shifted prediction
    return ans_entropy.mean().item()


def score(question, answer, model, tokenizer, device):
    """
    Hallucination score via negative mean log-prob (higher = more hallucinated).
    Normalized to [0, inf) — use AUROC, not a fixed threshold.
    """
    lp = logprobs_for_answer(question, answer, model, tokenizer, device)
    if lp.numel() == 0:
        return 0.0
    return (-lp.mean()).item()


def score_entropy(question, answer, model, tokenizer, device):
    """Alternative scorer using predictive entropy instead of mean log-prob."""
    return predictive_entropy(question, answer, model, tokenizer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or default_device()
    model, tokenizer = load_model(args.model, device)

    # Sanity check: true answer should score lower than false answer
    question = "What is the capital of France?"
    true_answer = "Paris is the capital of France."
    false_answer = "Berlin is the capital of France."

    s_true = score(question, true_answer, model, tokenizer, device)
    s_false = score(question, false_answer, model, tokenizer, device)

    print(f"Score (true answer):  {s_true:.4f}")
    print(f"Score (false answer): {s_false:.4f}")
    print(f"Sanity check passed:  {s_false > s_true}")
