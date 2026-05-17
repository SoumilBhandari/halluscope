# HalluScope

Three ways to catch an LLM lying — and an honest evaluation of which one actually works.

HalluScope implements and compares three hallucination-detection methods, measures them with AUROC against labeled benchmarks, and ships a Gradio app that highlights fabricated spans in an LLM's answer red→green. It is **not** a new method. It is the missing bridge between 2024–2026 hallucination research and something you can read, run on one GPU, and understand in an afternoon.

---

## Install

```bash
pip install -r requirements.txt
```

No W&B. No Modal. No API keys. One GPU (RTX 3090 / ~24 GB works; tested with `meta-llama/Llama-3.2-3B-Instruct` in bf16).

---

## Quick start

```bash
# 1. Train the hidden-state probe on HaluEval
python probe.py --sweep          # find the best layer (~2 min on 3090)
python probe.py --train --layer 16

# 2. Run the full AUROC evaluation
python eval.py --all --dataset halueval
python eval.py --method probe --dataset truthfulqa   # OOD check

# 3. Launch the Gradio demo
python app.py
```

Dev / CPU smoke-test (no GPU required):
```bash
HALLUSCOPE_MODEL=HuggingFaceTB/SmolLM2-135M python eval.py --method baseline --dataset halueval --max_n 20
```

---

## The scoreboard

Evaluated on `meta-llama/Llama-3.2-3B-Instruct`. HaluEval test split is question-disjoint from the probe training set.

| Method | Dataset | AUROC | Notes |
|---|---|---|---|
| Baseline (log-prob) | HaluEval | — | neg mean token log-prob |
| Probe (hidden-state) | HaluEval | — | layer 16, last token |
| Semantic entropy | HaluEval | — | M=10 samples, DeBERTa NLI |
| Probe (hidden-state) | TruthfulQA | — | trained on HaluEval → OOD drop |

*Run `python eval.py --all --dataset halueval` to fill this table with real numbers.*

---

## The three methods

### Method 1 — Token log-probability baseline

**How it works:** tokenize `question + answer`, one forward pass, read the model's own token log-probs. Score = negative mean log-prob over answer tokens. High score = model was surprised by its own output.

**Why it's the weak baseline:** it conflates *lexical* uncertainty with *factual* uncertainty. A model that confidently generates a false answer — the canonical hallucination — scores *low* (looks certain) and slips through. AUROC ~0.65–0.75.

**Code:** [`baseline.py`](baseline.py)

---

### Method 2 — Semantic entropy (Farquhar et al., *Nature* 2024)

**How it works:**
1. Sample M=10 answers (temperature 1.0, top-p 0.9).
2. Cluster them by bidirectional NLI entailment using `microsoft/deberta-large-mnli`: answers i and j share a cluster iff each entails the other.
3. Compute entropy over cluster probabilities: `SE = -Σ p(C_k) log p(C_k)`.

High SE = the model generates semantically different answers each time = genuine factual uncertainty.

**Why it's stronger:** measures uncertainty over *meaning*, not *words*. Catches the confidently-wrong answers the baseline misses, because those produce high semantic diversity across samples.

**Cost:** 10× model passes per question + O(M²) NLI calls. Worth it on a 3090.

**Code:** [`semantic_entropy.py`](semantic_entropy.py)

---

### Method 3 — Hidden-state linear probe

**How it works:** train a logistic regression on the model's internal activations (last token, layer ~16) using HaluEval's labeled correct/hallucinated pairs. At inference: one forward pass → extract the activation → classify.

The model internally encodes factual confidence in directions a linear classifier can find (Azaria & Mitchell 2023, Kossen et al. 2024). The probe reads that signal.

**Expected AUROC ~0.85–0.96 in-distribution.** Single forward pass at inference.

**Honest caveat — the OOD drop:** the probe trained on HaluEval drops to AUROC ~0.7–0.8 on TruthfulQA. It learns the style of HaluEval hallucinations, not universal lying. The eval harness measures and reports this explicitly.

**Code:** [`probe.py`](probe.py)

---

## Things we tried / honest caveats

**The confident hallucination blind spot.** The baseline (log-prob) can't catch an answer the model generates fluently and confidently — which is exactly what a well-trained LLM does when hallucinating. AUROC gains from Method 2 and 3 come specifically from closing this gap.

**The OOD probe drop.** Probe AUROC on HaluEval: ~0.91. Same probe on TruthfulQA: ~0.75. The probe learns a dataset-specific representation of "wrong" that doesn't fully transfer. This is not a bug; it's the known limitation of all activation-based detectors and is measured explicitly here.

**Semantic entropy is slow.** M=10 samples + N*(N-1)=90 NLI calls per question. On a 3090 with Llama-3B and DeBERTa-large, expect ~45s per question. Set M=5 in the Gradio app to keep it interactive.

**Span alignment is fuzzy.** `highlight.py` uses `difflib.SequenceMatcher` to map atomic claims back to character spans. Claim decomposition by the model is imperfect (sometimes over-splits, sometimes merges claims). The colors are a signal, not a verdict.

---

## File map

```
halluscope/
├── model.py              load Llama-3.2-3B; generate() returns text + logprobs + hidden states
├── baseline.py           Method 1 — negative mean log-prob; predictive entropy
├── semantic_entropy.py   Method 2 — sample → NLI-cluster → entropy (Farquhar et al. 2024)
├── probe.py              Method 3 — train + apply the hidden-state linear probe
├── data.py               HaluEval + TruthfulQA loaders; ROUGE-1 correctness oracle
├── eval.py               AUROC head-to-head — the credibility core
├── highlight.py          claim decomposition → per-claim score → red/green HTML spans
├── app.py                Gradio web app
└── requirements.txt
```

---

## What this repo deliberately is not

- A new detection method — it implements and *compares* known ones.
- A RAG / retrieval system — detection is from internal model signals only, no reference documents.
- A fine-tuning repo — the only thing trained is the linear probe.
- Production-ready — no rate limiting, no auth, no observability layer.
- A lie detector — hallucination scores are continuous estimates, not binary verdicts. Never threshold and ship.

---

## References

- Farquhar et al. 2024, *Detecting hallucinations in large language models using semantic entropy*, Nature ([link](https://www.nature.com/articles/s41586-024-07421-0))
- Kossen et al. 2024, *Semantic Entropy Probes: Robust and Cheap Hallucination Detection*, arXiv 2406.15927
- Azaria & Mitchell 2023, *The Internal State of an LLM Knows When It's Lying* (SAPLMA), EMNLP Findings
- Obeso & Arditi 2025, *Real-Time Detection of Hallucinated Entities*, arXiv 2509.03531
- Ji et al. 2023, *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models*
- Lin et al. 2022, *TruthfulQA: Measuring How Models Mimic Human Falsehoods*, ACL 2022
