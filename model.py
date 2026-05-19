"""
Model loading and generation for HalluScope.

Wraps a HuggingFace causal LM so that a single generate() call returns the
generated text, per-token log-probabilities, and the full hidden-state tensor.
All three detection methods draw from these outputs.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("HALLUSCOPE_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name=DEFAULT_MODEL, device=None):
    if device is None:
        device = default_device()

    # bfloat16 on CUDA; float16 on MPS (bfloat16 unstable on M1); float32 on CPU
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    ).to(device).eval()

    return model, tokenizer


@torch.no_grad()
def generate(
    prompt,
    model,
    tokenizer,
    device,
    max_new_tokens=200,
    temperature=1.0,
    top_p=0.9,
    top_k=50,
    num_return_sequences=1,
    return_logprobs=True,
    return_hidden_states=False,
    chat=True,
):
    """
    Generate from prompt. Returns a dict (or list of dicts for num_return_sequences > 1):
      text          : generated string (without the prompt)
      logprobs      : Tensor[T_new] — per-token log-prob of the chosen token
      hidden        : Tensor[L+1, T_prompt+T_new, D] — all layer hidden states
                      use hidden[:, -1, :] for last-token representation

    chat=True wraps the prompt in the model's chat template. Instruction-tuned
    models need this — fed a bare prompt they continue it as a document instead
    of answering. Falls back to raw tokenization if the tokenizer has no template.
    """
    if chat and getattr(tokenizer, "chat_template", None):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # temperature=0 isn't valid; use greedy via do_sample=False
    do_sample = temperature > 0 and temperature != 1e-9
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_scores=return_logprobs,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        gen_kwargs["do_sample"] = False

    outputs = model.generate(**inputs, **gen_kwargs)

    results = []
    for seq_idx in range(num_return_sequences):
        new_ids = outputs.sequences[seq_idx, input_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)

        logprobs = None
        if return_logprobs and outputs.scores:
            # scores: tuple of T_new tensors, each (num_return_sequences, V)
            stacked = torch.stack(outputs.scores, dim=1)[seq_idx]  # (T_new, V)
            log_probs_all = torch.log_softmax(stacked, dim=-1)
            token_ids = new_ids[: stacked.shape[0]]
            logprobs = log_probs_all[range(len(token_ids)), token_ids]

        hidden = None
        if return_hidden_states:
            # Re-run a clean forward pass over the full sequence; generate()'s own
            # hidden_states are emitted per decoding step and awkward to stitch
            # back into a single (L+1, T, D) tensor.
            full_ids = outputs.sequences[seq_idx].unsqueeze(0)
            fwd = model(full_ids, output_hidden_states=True)
            hidden = torch.stack([h[0] for h in fwd.hidden_states], dim=0)  # (L+1, T_full, D)

        results.append({"text": text, "logprobs": logprobs, "hidden": hidden})

    return results[0] if num_return_sequences == 1 else results
