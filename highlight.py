"""
Claim decomposition and span-level hallucination highlighting.

Pipeline:
  1. Prompt the LLM to decompose its answer into atomic factual claims (JSON).
  2. Score each claim with the chosen detector.
  3. Map each claim back to its character span in the original answer.
  4. Return a list of annotated spans for rendering.

The visual — red->green colored spans — is the shareable hook of the demo.
"""

import difflib
import json
import re

from model import DEFAULT_MODEL, default_device, generate, load_model

DECOMPOSE_PROMPT = """Break the following answer into a JSON array of short, atomic factual claims. Each claim should be one sentence or less. Output only valid JSON.

Answer: {answer}

Claims:"""


def parse_claims(raw, fallback_text):
    """
    Extract a JSON array of claim strings from raw model output. Falls back to
    sentence-splitting `fallback_text` when no valid JSON array is present.
    Pure function — unit-testable without a model.
    """
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            claims = json.loads(match.group())
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                cleaned = [c.strip() for c in claims if c.strip()]
                if cleaned:
                    return cleaned
        except json.JSONDecodeError:
            pass
    sentences = re.split(r"(?<=[.!?])\s+", fallback_text.strip())
    return [s.strip() for s in sentences if s.strip()]


def decompose_claims(answer, model, tokenizer, device, max_new_tokens=400):
    """Ask the model to decompose its answer into atomic claims."""
    prompt = DECOMPOSE_PROMPT.format(answer=answer)
    result = generate(
        prompt, model, tokenizer, device,
        max_new_tokens=max_new_tokens, temperature=0,
        return_logprobs=False, return_hidden_states=False,
    )
    return parse_claims(result["text"].strip(), answer)


def find_span(claim, answer):
    """
    Find the best-matching character span of `claim` in `answer`.
    Returns (start, end), or None if no reasonable match is found.
    """
    claim_lower = claim.lower()
    answer_lower = answer.lower()

    idx = answer_lower.find(claim_lower)
    if idx != -1:
        return (idx, idx + len(claim))

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


def normalize_scores(scores):
    """
    Map raw detector scores into [0, 1] for coloring.

    Scores already within [0, 1] (the probe returns a probability) are used
    as-is, so a claim's color reflects its absolute hallucination probability.
    Unbounded scores (the baseline's negative log-prob) are min-max scaled
    across the claims of this answer — a relative signal, not an absolute one.
    """
    if not scores:
        return []
    if all(0.0 <= s <= 1.0 for s in scores):
        return list(scores)
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def score_claims(question, claims, score_fn, model, tokenizer, device):
    """Score each claim independently. Returns a list of floats."""
    return [score_fn(question, claim, model, tokenizer, device) for claim in claims]


def highlight(question, answer, score_fn, model, tokenizer, device):
    """
    Full pipeline: decompose -> score -> locate spans.
    Returns a list of {"claim": str, "score": float in [0,1], "span": (int,int)|None}.
    """
    claims = decompose_claims(answer, model, tokenizer, device)
    scores = normalize_scores(score_claims(question, claims, score_fn, model, tokenizer, device))
    return [
        {"claim": claim, "score": sc, "span": find_span(claim, answer)}
        for claim, sc in zip(claims, scores)
    ]


def render_html(answer, highlights):
    """
    Build an HTML string with colored spans.
    score=1 -> red (hue 0), score=0 -> green (hue 120).
    """
    scores_map = [None] * len(answer)
    for h in highlights:
        if h["span"] is not None:
            start, end = h["span"]
            for i in range(start, min(end, len(answer))):
                scores_map[i] = h["score"]

    parts = []
    i = 0
    while i < len(answer):
        sc = scores_map[i]
        j = i + 1
        while j < len(answer) and scores_map[j] == sc:
            j += 1
        chunk = answer[i:j]
        if sc is not None:
            hue = int((1 - sc) * 120)  # 0 -> red, 120 -> green
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

    device = args.device or default_device()
    model, tokenizer = load_model(args.model, device)

    results = highlight(args.question, args.answer, baseline_mod.score, model, tokenizer, device)
    print("\nClaims and hallucination scores:")
    for r in results:
        span_str = f"chars {r['span']}" if r["span"] else "no span"
        print(f"  [{r['score']:.3f}] {r['claim']}  ({span_str})")

    print("\nHTML output (open in browser):")
    print(render_html(args.answer, results))
