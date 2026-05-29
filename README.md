# HalluScope

Three ways to catch an LLM lying — and an honest evaluation of which one actually works.

HalluScope implements and compares three hallucination-detection methods, measures them with AUROC against labeled benchmarks, and ships a Gradio app that highlights fabricated spans in an LLM's answer red→green. It is **not** a new method. It is the missing bridge between 2024–2026 hallucination research and something you can read, run on one GPU, and understand in an afternoon.

[![CI](https://github.com/SoumilBhandari/halluscope/actions/workflows/ci.yml/badge.svg)](https://github.com/SoumilBhandari/halluscope/actions/workflows/ci.yml)

---

## Install

```bash
pip install -r requirements.txt
```

No W&B. No Modal. No API keys. No gated models. One GPU (RTX 3090 / ~24 GB works; tested with `Qwen/Qwen2.5-3B-Instruct` in bf16 — Apache-2.0, downloads without a login).

---

## Quick start

```bash
# 1. Train the hidden-state probe on HaluEval
python probe.py --sweep                 # optional: sweep layers on the val split
python probe.py --train --layer 20      # writes probe.json

# 2. Run the AUROC evaluation
python eval.py --all --dataset halueval   --out results_halueval.json
python eval.py --all --dataset truthfulqa --out results_truthfulqa.json

# 3. Launch the Gradio demo
python app.py
```

Dev / CPU smoke-test (no GPU, tiny model):
```bash
HALLUSCOPE_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct \
  python eval.py --method baseline --dataset halueval --max_n 200
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -m "not integration"   # fast unit tests — no model, no GPU, no network
pytest -m integration         # end-to-end smoke test on a tiny model
```

CI runs the unit tests on every push and pull request.

---

## Deploy the demo (Hugging Face Spaces)

The Gradio app is Spaces-ready. With a free Hugging Face account, from the repo root:

```bash
pip install -U gradio
gradio deploy            # prompts for an HF token, creates the Space
```

That ships `app.py` with the committed `probe.json`, so the baseline and probe
work immediately (semantic entropy downloads its NLI model on first use). Give
the Space a GPU for the 3B model to feel responsive; on CPU, point
`HALLUSCOPE_MODEL` at a smaller model. To wire it up by hand instead, create a
Gradio Space (app file `app.py`) and add this header to the Space's `README.md`:

```yaml
---
title: HalluScope
emoji: 🔬
sdk: gradio
app_file: app.py
---
```

---

## The scoreboard

Evaluated on `Qwen/Qwen2.5-3B-Instruct`. HaluEval is split question-disjoint into train / val / test: the probe trains on train, its layer is chosen on val, and every number below is measured on data the probe never saw.

| Method | Dataset | AUROC | N |
|---|---|---|---|
| Baseline (log-prob) | HaluEval | 0.254 | 4,000 |
| Probe (hidden-state, layer 20) | HaluEval | **0.988** | 4,000 |
| Baseline (log-prob) | TruthfulQA | 0.480 | 200 |
| Probe (hidden-state, layer 20) | TruthfulQA | 0.523 | 200 |
| Semantic entropy | TruthfulQA | 0.502 | 200 |

Numbers come straight out of `eval.py` (raw output in [`results.json`](results.json)) — measured, not guessed. Reproduce with the two `eval.py` commands above. Semantic entropy is evaluated only on TruthfulQA; on HaluEval `eval.py` reports it as N/A on purpose (see Method 2).

### What the numbers say — the honest read

The honest evaluation is not flattering for two of the three methods:

- **The probe is excellent in-distribution and worthless out of it** — 0.988 on held-out HaluEval, 0.523 (chance) on TruthfulQA. It learned the *style* of HaluEval's hallucinations, not a general "the model is lying" signal. The OOD drop is real, measured, and more severe than the ~0.75 hand-waved in early drafts.
- **The log-prob baseline is *worse than chance* on HaluEval (0.254)** — anti-correlated with truth. The model is consistently *more* confident on HaluEval's fabricated answers than on the correct ones. Log-probability tracks fluency, and a fabricated answer is fluent, so the baseline doesn't just miss confident hallucinations — it inverts on them.
- **Semantic entropy did not beat chance here (0.502)** — it does not reproduce Farquhar et al.'s result in this setup. Flagged as a known failure, not hidden; honest candidate reasons are in the caveats below.

The one solid, reproducible result: a linear probe on layer-20 activations separates HaluEval hallucinations near-perfectly — **but only in-distribution.** Take that, and take how easy it is to mistake an in-distribution probe for a working detector.

---

## The three methods

### Method 1 — Token log-probability baseline

**How it works:** tokenize `question + answer`, one forward pass, read the model's own token log-probs. Score = negative mean log-prob over answer tokens. High score = model was surprised by its own output.

**Why it's the weak baseline:** it conflates *lexical* uncertainty with *factual* uncertainty. A model that confidently generates a false answer — the canonical hallucination — scores *low* (looks certain) and slips through.

**Code:** [`baseline.py`](baseline.py)

---

### Method 2 — Semantic entropy (Farquhar et al., *Nature* 2024)

**How it works:**
1. Sample M=10 answers (temperature 1.0, top-p 0.9), seeded for reproducibility.
2. Cluster them by bidirectional NLI entailment using `microsoft/deberta-large-mnli`: answers i and j share a cluster iff each entails the other.
3. Compute entropy over cluster probabilities: `SE = -Σ p(C_k) log p(C_k)`.

High SE = the model generates semantically different answers each time = genuine factual uncertainty.

**The idea:** measure uncertainty over *meaning*, not *words* — in principle catching the confidently-wrong answers the baseline misses, since genuine uncertainty should produce semantically diverse samples. In this repo's run it did not pan out (0.502 — see the scoreboard and caveats): a useful, honest negative result.

**Evaluated on TruthfulQA, not HaluEval.** Semantic entropy samples *fresh* answers and measures how much they disagree — it judges the model's uncertainty about a *question*, so the model has to generate the answer itself. HaluEval supplies a pre-written answer the method can't use, so pairing SE with HaluEval would only measure chance. `eval.py` reports `N/A` for that pairing on purpose.

**Cost:** 10× model passes per question + O(M²) NLI calls. Worth it on a 3090. The Gradio app uses M=5 to stay interactive.

**Code:** [`semantic_entropy.py`](semantic_entropy.py)

---

### Method 3 — Hidden-state linear probe

**How it works:** train a logistic regression on the model's internal activations (last token, layer 20) using HaluEval's labeled correct/hallucinated pairs. At inference: one forward pass → extract the activation → classify.

The model internally encodes factual confidence in directions a linear classifier can find (Azaria & Mitchell 2023, Kossen et al. 2024). The probe reads that signal.

The trained probe is saved as plain JSON (`probe.json`) — scaler statistics plus the logistic-regression weights — so loading a probe never executes pickled code.

**Honest caveat — the OOD drop:** the probe trained on HaluEval is strong in-distribution but drops on TruthfulQA. It learns the *style* of HaluEval hallucinations, not universal lying. The eval harness measures and reports this explicitly — see the scoreboard.

**Code:** [`probe.py`](probe.py)

---

## Things we tried / honest caveats

**The confident-hallucination failure is real — and measurable.** The log-prob baseline can't catch a fluently-generated false answer; on HaluEval it does worse than that and scores *below chance* (0.254). A fabricated-but-fluent answer looks low-perplexity to the model. Only the probe closes this gap, and only in-distribution.

**The OOD probe drop is severe.** The probe learns a dataset-specific representation of "wrong" that does not transfer from HaluEval to TruthfulQA — 0.988 down to 0.523 (chance). This is the known limitation of all activation-based detectors, measured explicitly here rather than hidden.

**Semantic entropy underperformed, and we're not hiding it.** SE scored 0.502 on TruthfulQA — chance. It does not reproduce the Farquhar et al. result in this setup. Honest candidate reasons, not excuses: (1) `Qwen2.5-3B-Instruct` is RLHF-aligned and produces low-diversity samples even when wrong, so there is little semantic spread for SE to measure; (2) the ROUGE-1 grading oracle is crude on free-form answers and adds label noise; (3) N=200 is a modest sample. A larger run, a less-aligned model, or a stronger grader might change this — but as run, SE failed, and that is what the table shows.

**Semantic entropy is also slow.** M=10 samples + N*(N-1) NLI calls per question — tens of seconds each; the 200-question TruthfulQA run took ~38 minutes. The Gradio app uses M=5 to stay interactive.

**TruthfulQA labels are noisy.** `grade_truthfulqa` labels an answer by ROUGE-1 overlap against reference answer lists. For free-form instruct-model answers that is crude, and it compresses every method's measurable AUROC toward 0.5. Read the TruthfulQA column as a rough OOD indicator, not a precise number.

**Span alignment is fuzzy.** `highlight.py` uses `difflib.SequenceMatcher` to map atomic claims back to character spans. Claim decomposition by the model is imperfect (sometimes over-splits, sometimes merges claims). The colors are a signal, not a verdict.

---

## File map

```
halluscope/
├── model.py              load the LM; chat-templated generate() returns text + logprobs + hidden states
├── baseline.py           Method 1 — negative mean token log-prob
├── semantic_entropy.py   Method 2 — sample → NLI-cluster → entropy (Farquhar et al. 2024)
├── probe.py              Method 3 — train + apply the hidden-state linear probe (saved as JSON)
├── data.py               HaluEval + TruthfulQA loaders; ROUGE-1 correctness oracle
├── eval.py               AUROC head-to-head — the credibility core
├── highlight.py          claim decomposition → per-claim score → red/green HTML spans
├── app.py                Gradio web app
├── tests/                pytest unit + integration tests
├── .github/workflows/    CI
├── requirements.txt      runtime dependencies
└── requirements-dev.txt  + test tooling
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

---

## License

MIT — see [LICENSE](LICENSE).
