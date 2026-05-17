"""
Model loading and generation for HalluScope.

Wraps a HuggingFace causal LM so that a single generate() call returns the
generated text, per-token log-probabilities, and the full hidden-state tensor.
All three detection methods draw from these outputs.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("HALLUSCOPE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")


def load_model(model_name=DEFAULT_MODEL, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

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
    return_hidden_states=True,
):
    """
    Generate from prompt. Returns a dict (or list of dicts for num_return_sequences > 1):
      text          : generated string (without the prompt)
      logprobs      : Tensor[T_new] — per-token log-prob of the chosen token
      hidden        : Tensor[L+1, T_prompt+T_new, D] — all layer hidden states
                      use hidden[:, -1, :] for last-token representation
    """
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
        if return_hidden_states and outputs.hidden_states:
            # outputs.hidden_states: tuple per decoding step, each is tuple of L+1 tensors (B, T, D)
            # We want the hidden states at the final token position across all layers.
            # Collect last-token hidden state from each step's last layer isn't quite right;
            # instead stack the full sequence hidden states from the last decoding step.
            #
            # For the full sequence (prompt + generated), run a forward pass to get clean hiddens.
            full_ids = outputs.sequences[seq_idx].unsqueeze(0)
            fwd = model(full_ids, output_hidden_states=True)
            # fwd.hidden_states: tuple of L+1 tensors each (1, T_full, D)
            hidden = torch.stack([h[0] for h in fwd.hidden_states], dim=0)  # (L+1, T_full, D)

        results.append({"text": text, "logprobs": logprobs, "hidden": hidden})

    return results[0] if num_return_sequences == 1 else results


@torch.no_grad()
def forward_hidden(prompt, model, tokenizer, device):
    """
    Single forward pass — returns hidden states only (no generation).
    Cheaper than generate() when you only need hidden states for a fixed text.
    Returns: Tensor[L+1, T, D]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    return torch.stack([h[0] for h in outputs.hidden_states], dim=0)  # (L+1, T, D)
