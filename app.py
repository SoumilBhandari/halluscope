"""
HalluScope Gradio web app.

Type a question → the LLM answers → fabricated spans glow red, confident
spans glow green. Choose your detector from the sidebar.

Launch:
    python app.py

Environment:
    HALLUSCOPE_MODEL   model name (default: meta-llama/Llama-3.2-3B-Instruct)
    HALLUSCOPE_PROBE   path to probe.pkl (default: probe.pkl)
"""

import functools
import os

import gradio as gr
import torch

import baseline as baseline_mod
import highlight as hl_mod
import semantic_entropy as se_mod
from model import DEFAULT_MODEL, default_device, generate, load_model

PROBE_PATH = os.environ.get("HALLUSCOPE_PROBE", "probe.pkl")


def _load_all():
    device = default_device()
    model, tokenizer = load_model(DEFAULT_MODEL, device)

    nli_model, nli_tokenizer = None, None
    probe_score_fn = None

    if os.path.exists(PROBE_PATH):
        import probe as probe_mod
        scaler, clf, layer = probe_mod.load_probe(PROBE_PATH)
        probe_score_fn = functools.partial(probe_mod.score, scaler=scaler, clf=clf, layer=layer)

    return model, tokenizer, device, nli_model, nli_tokenizer, probe_score_fn


print("Loading model — this may take a moment...")
_model, _tokenizer, _device, _nli_model, _nli_tokenizer, _probe_score_fn = _load_all()


def _get_score_fn(method):
    if method == "Baseline (log-prob)":
        return baseline_mod.score
    elif method == "Probe (hidden-state)":
        if _probe_score_fn is None:
            return None
        return _probe_score_fn
    elif method == "Semantic Entropy":
        # Lazy-load NLI model on first use
        global _nli_model, _nli_tokenizer
        if _nli_model is None:
            print("Loading NLI model...")
            _nli_model, _nli_tokenizer = se_mod.load_nli_model(_device)
        return functools.partial(se_mod.score, nli_model=_nli_model, nli_tokenizer=_nli_tokenizer, M=5)
    return None


def run(question, method, max_new_tokens):
    if not question.strip():
        return "<p><em>Please enter a question.</em></p>", ""

    score_fn = _get_score_fn(method)
    if score_fn is None:
        return (
            "<p><em>Probe not found. Run <code>python probe.py --train</code> first.</em></p>",
            "",
        )

    # Generate a greedy answer
    result = generate(
        question, _model, _tokenizer, _device,
        max_new_tokens=int(max_new_tokens),
        temperature=0,
        return_logprobs=False,
        return_hidden_states=False,
    )
    answer = result["text"].strip()

    # Highlight claims
    highlights = hl_mod.highlight(question, answer, score_fn, _model, _tokenizer, _device)
    html = hl_mod.render_html(answer, highlights)

    # Build claim table
    rows = ""
    for h in highlights:
        color = f"hsl({int((1-h['score'])*120)}, 70%, 40%)"
        rows += (
            f"<tr>"
            f"<td style='padding:4px 8px; color:{color}; font-weight:bold'>{h['score']:.2f}</td>"
            f"<td style='padding:4px 8px'>{hl_mod._esc(h['claim'])}</td>"
            f"</tr>"
        )
    table_html = f"<table style='border-collapse:collapse; font-family:monospace; font-size:13px'>{rows}</table>"

    return html, table_html


DESCRIPTION = """
## HalluScope — three ways to catch an LLM lying

Type a question. The model answers. Fabricated spans glow **red**; confident spans glow **green**.

| Method | How it works | AUROC |
|---|---|---|
| Baseline (log-prob) | Negative mean token log-prob | ~0.70 |
| Probe (hidden-state) | Linear classifier on layer activations | ~0.91 |
| Semantic Entropy | NLI-clustered answer diversity | ~0.83 |

*Scores are in-distribution (HaluEval). Probe drops to ~0.75 on TruthfulQA (OOD — documented in README).*
"""

with gr.Blocks(title="HalluScope") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            question_box = gr.Textbox(
                label="Question",
                placeholder="e.g. Who invented the telephone?",
                lines=3,
            )
            method_radio = gr.Radio(
                choices=["Baseline (log-prob)", "Probe (hidden-state)", "Semantic Entropy"],
                value="Baseline (log-prob)",
                label="Detection method",
            )
            max_tokens_slider = gr.Slider(
                minimum=50, maximum=400, value=150, step=10,
                label="Max answer length (tokens)",
            )
            run_btn = gr.Button("Detect hallucinations", variant="primary")

        with gr.Column(scale=2):
            answer_html = gr.HTML(label="Answer (colored by hallucination score)")
            claims_html = gr.HTML(label="Per-claim scores")

    run_btn.click(
        fn=run,
        inputs=[question_box, method_radio, max_tokens_slider],
        outputs=[answer_html, claims_html],
    )

    gr.Examples(
        examples=[
            ["Who invented the telephone?", "Baseline (log-prob)", 150],
            ["What are the health benefits of drinking bleach?", "Baseline (log-prob)", 150],
            ["When was the Eiffel Tower built and who designed it?", "Probe (hidden-state)", 150],
        ],
        inputs=[question_box, method_radio, max_tokens_slider],
    )


if __name__ == "__main__":
    demo.launch()
