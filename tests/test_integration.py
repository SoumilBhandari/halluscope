"""End-to-end smoke test on a tiny real model.

Marked `integration` so the no-model CI run skips it. To run it locally:
    pytest -m integration
"""

import pytest

SMALL_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.mark.integration
def test_baseline_scores_a_qa_pair():
    import baseline
    from model import load_model

    model, tokenizer = load_model(SMALL_MODEL, "cpu")
    s = baseline.score(
        "What is the capital of France?",
        "Paris is the capital of France.",
        model, tokenizer, "cpu",
    )
    assert isinstance(s, float)
    assert s >= 0.0


@pytest.mark.integration
def test_chat_template_generation_produces_text():
    from model import generate, load_model

    model, tokenizer = load_model(SMALL_MODEL, "cpu")
    result = generate(
        "Name one primary color.", model, tokenizer, "cpu",
        max_new_tokens=16, temperature=0, return_logprobs=False,
    )
    assert isinstance(result["text"], str)
    assert len(result["text"].strip()) > 0
