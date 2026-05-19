"""
HalluScope Gradio web app.

Type a question -> the LLM answers -> fabricated spans glow red, confident
spans glow green. Choose your detector from the sidebar.

Launch:
    python app.py

Environment:
    HALLUSCOPE_MODEL     model name (default: Qwen/Qwen2.5-3B-Instruct)
    HALLUSCOPE_PROBE     path to probe.json (default: probe.json)
    HALLUSCOPE_RESULTS   path to results.json for the scoreboard (default: results.json)
"""

import functools
import json
import os

import gradio as gr

import baseline as baseline_mod
import highlight as hl_mod
import semantic_entropy as se_mod
from model import DEFAULT_MODEL, default_device, generate, load_model

PROBE_PATH = os.environ.get("HALLUSCOPE_PROBE", "probe.json")
RESULTS_PATH = os.environ.get("HALLUSCOPE_RESULTS", "results.json")

# Heavy objects load lazily on first request, so importing this module (for
# tests, or just to inspect it) never downloads or loads a multi-GB model.
_STATE = {}


def _ensure_loaded():
    """Load the language model (and probe, if present) once, on first use."""
    if "model" in _STATE:
        return
    print("Loading model - this may take a moment...")
    device = default_device()
    model, tokenizer = load_model(DEFAULT_MODEL, device)
    _STATE.update(
        model=model, tokenizer=tokenizer, device=device,
        nli_model=None, nli_tokenizer=None, probe=None,
    )
    if os.path.exists(PROBE_PATH):
        import probe as probe_mod
        _STATE["probe"] = probe_mod.load_probe(PROBE_PATH)


def _get_score_fn(method):
    if method == "Baseline (log-prob)":
        return baseline_mod.score

    if method == "Probe (hidden-state)":
        if _STATE.get("probe") is None:
            return None
        import probe as probe_mod
        return functools.partial(probe_mod.score, probe=_STATE["probe"])

    if method == "Semantic Entropy":
        if _STATE.get("nli_model") is None:
            print("Loading NLI model...")
            nli_model, nli_tokenizer = se_mod.load_nli_model(_STATE["device"])
            _STATE.update(nli_model=nli_model, nli_tokenizer=nli_tokenizer)
        return functools.partial(
            se_mod.score, nli_model=_STATE["nli_model"],
            nli_tokenizer=_STATE["nli_tokenizer"], M=5,
        )
    return None


def run(question, method, max_new_tokens):
    if not question.strip():
        return "<p><em>Please enter a question.</em></p>", ""

    _ensure_loaded()
    score_fn = _get_score_fn(method)
    if score_fn is None:
        return (
            "<p><em>Probe not found. Run <code>python probe.py --train</code> first.</em></p>",
            "",
        )

    result = generate(
        question, _STATE["model"], _STATE["tokenizer"], _STATE["device"],
        max_new_tokens=int(max_new_tokens),
        temperature=0,
        return_logprobs=False,
        return_hidden_states=False,
    )
    answer = result["text"].strip()

    highlights = hl_mod.highlight(
        question, answer, score_fn,
        _STATE["model"], _STATE["tokenizer"], _STATE["device"],
    )
    html = hl_mod.render_html(answer, highlights)

    rows = ""
    for h in highlights:
        color = f"hsl({int((1 - h['score']) * 120)}, 70%, 40%)"
        rows += (
            f"<tr>"
            f"<td style='padding:4px 8px; color:{color}; font-weight:bold'>{h['score']:.2f}</td>"
            f"<td style='padding:4px 8px'>{hl_mod._esc(h['claim'])}</td>"
            f"</tr>"
        )
    table_html = f"<table style='border-collapse:collapse; font-family:monospace; font-size:13px'>{rows}</table>"

    return html, table_html


def _scoreboard_md():
    """Render the measured scoreboard from results.json, or a placeholder."""
    if not os.path.exists(RESULTS_PATH):
        return (
            "_No measured results yet. Run "
            "`python eval.py --all --dataset truthfulqa --out results.json`._"
        )
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    lines = ["| Method | Dataset | AUROC | N |", "|---|---|---|---|"]
    for r in data.get("rows", []):
        auc = "N/A" if r.get("auroc") is None else f"{r['auroc']:.3f}"
        lines.append(f"| {r['method']} | {r['dataset']} | {auc} | {r.get('n', '')} |")
    return f"Measured on `{data.get('model', '?')}`:\n\n" + "\n".join(lines)


DESCRIPTION = f"""
## HalluScope — three ways to catch an LLM lying

Type a question. The model answers. Fabricated spans glow **red**; confident
spans glow **green**. Pick a detector on the left.

- **Baseline (log-prob)** — negative mean token log-probability.
- **Probe (hidden-state)** — a linear classifier on the model's layer activations.
- **Semantic Entropy** — NLI-clustered diversity across resampled answers.

{_scoreboard_md()}

*Scores are continuous estimates, not verdicts — never threshold and ship.*
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
    _ensure_loaded()
    demo.launch()
