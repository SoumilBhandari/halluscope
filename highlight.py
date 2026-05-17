"""
Claim decomposition and span-level hallucination highlighting.

Pipeline:
  1. Prompt the LLM to decompose its answer into atomic factual claims (JSON).
  2. Score each claim with the chosen detector.
  3. Map each claim back to its character span in the original answer.
  4. Return a list of annotated spans for rendering.

The visual — red→green colored spans — is the shareable LinkedIn hook.
"""

import difflib
import json
import re

import torch

from model import DEFAULT_MODEL, generate, load_model

DECOMPOSE_PROMPT = """Break the following answer into a JSON array of short, atomic factual claims. Each claim should be one sentence or less. Output only valid JSON.

Answer: {answer}

Claims:"""


def decompose_claims(answer, model, tokenizer, device, max_new_tokens=400):
    """
    Ask the model to decompose its answer into atomic claims.
    Returns list of claim strings. Falls back to sentence splitting if JSON parse fails.
    """
    prompt = DECOMPOSE_PROMPT.format(answer=answer)
    result = generate(
        prompt, model, tokenizer, device,
        max_new_tokens=max_new_tokens,
        temperature=0,
        return_logprobs=False,
        return_hidden_states=False,
    )
    raw = result["text"].strip()

    # Try to extract JSON array from the output
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            claims = json.loads(match.group())
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                return [c.strip() for c in claims if c.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s.strip() for s in sentences if s.strip()]


def find_span(claim, answer):
    """
    Find the best-matching character span of claim in answer using difflib.
    Returns (start, end) or None if no reasonable match found.
    """
    claim_lower = claim.lower()
    answer_lower = answer.lower()

    # Try exact substring first
    idx = answer_lower.find(claim_lower)
    if idx != -1:
        return (idx, idx + len(claim))

    # Sliding window fuzzy match: find window with highest similarity
    words = claim.split()
    if not words:
        return None

    window = len(claim)
    best_ratio, best_start = 0.0, 0

    for start in range(0, max(1, len(answer) - window + 1), max(1, window // 4)):
        end = min(start + window + len(words) * 2, len(answer))
        snippet = answer_lower[start:end]
        ratio = difflib.SequenceMatcher(None, claim_lower, snippet).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, start

    if best_ratio < 0.4:
        return None

    return (best_start, min(best_start + window, len(answer)))


def score_claims(question, claims, score_fn, model, tokenizer, device):
    """Score each claim independently. Returns list of floats."""
    return [score_fn(question, claim, model, tokenizer, device) for claim in claims]


def highlight(question, answer, score_fn, model, tokenizer, device):
    """
    Full pipeline: decompose → score → locate spans.
    Returns list of dicts:
      {"claim": str, "score": float, "span": (int, int) | None}
    score is in [0,1] where 1 = definitely hallucinated.
    """
    claims = decompose_claims(answer, model, tokenizer, device)
    scores = score_claims(question, claims, score_fn, model, tokenizer, device)

    # Normalize scores to [0,1] range (baseline returns unbounded values)
    if scores:
        lo, hi = min(scores), max(scores)
        if hi > lo:
            scores = [(s - lo) / (hi - lo) for s in scores]
        else:
            scores = [0.5] * len(scores)

    results = []
    for claim, sc in zip(claims, scores):
        span = find_span(claim, answer)
        results.append({"claim": claim, "score": sc, "span": span})

    return results


def render_html(answer, highlights):
    """
    Build an HTML string with colored spans.
    score=1 → red (hsl 0), score=0 → green (hsl 120).
    Overlapping / unmatched claims are appended as footnotes.
    """
    # Build a character-level score map
    scores_map = [None] * len(answer)
    for h in highlights:
        if h["span"] is not None:
            start, end = h["span"]
            for i in range(start, min(end, len(answer))):
                scores_map[i] = h["score"]

    # Build HTML by grouping consecutive characters with the same score
    parts = []
    i = 0
    while i < len(answer):
        sc = scores_map[i]
        j = i + 1
        while j < len(answer) and scores_map[j] == sc:
            j += 1
        chunk = answer[i:j]
        if sc is not None:
            hue = int((1 - sc) * 120)  # 0→red, 120→green
            lightness = 85 - int(sc * 20)  # slightly darker for high scores
            style = f"background-color: hsl({hue}, 80%, {lightness}%); border-radius: 2px; padding: 1px 2px;"
            parts.append(f'<span style="{style}" title="score: {sc:.2f}">{_esc(chunk)}</span>')
        else:
            parts.append(_esc(chunk))
        i = j

    return "<p>" + "".join(parts) + "</p>"


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    import argparse
    import baseline as baseline_mod

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model, device)

    results = highlight(args.question, args.answer, baseline_mod.score, model, tokenizer, device)
    print("\nClaims and hallucination scores:")
    for r in results:
        span_str = f"chars {r['span']}" if r["span"] else "no span"
        print(f"  [{r['score']:.3f}] {r['claim']}  ({span_str})")

    html = render_html(args.answer, results)
    print("\nHTML output (open in browser):")
    print(html)
