"""04 — Logit lens: decode the residual stream after every layer."""

import torch
from transformer_lens import HookedTransformer

from common.data import FACTUAL_PROMPTS
from common.models import load_gpt2
from common.results import save_results


def top_token_by_layer(model: HookedTransformer, prompt: str) -> list[str]:
    """Top-1 next token at the final position, decoded from the residual stream at each depth.

    Returns n_layers + 1 strings: index 0 is after the embeddings, index i is after block i - 1.
    Decode with the final LayerNorm and unembedding, as the model does at the end.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt, names_filter=lambda n: n.endswith("hook_resid_post") or n == "blocks.0.hook_resid_pre")
        resids = [cache["blocks.0.hook_resid_pre"][0, -1]] + [
            cache[f"blocks.{layer}.hook_resid_post"][0, -1] for layer in range(model.cfg.n_layers)
        ]
        logits = model.unembed(model.ln_final(torch.stack(resids)[None]))[0]
    return [model.tokenizer.decode(i) for i in logits.argmax(-1).tolist()]


def first_correct_layer(top_tokens: list[str], answer: str) -> int | None:
    """Index of the first depth whose top-1 token is the answer, or None if it never is."""
    return next((i for i, token in enumerate(top_tokens) if token == answer), None)


def main() -> None:
    model = load_gpt2()
    entries = []
    for prompt, answer in FACTUAL_PROMPTS:
        tokens = top_token_by_layer(model, prompt)
        layer = first_correct_layer(tokens, answer)
        print(f"{prompt!r} -> {answer!r}: first top-1 at depth {layer}")
        entries.append({"prompt": prompt, "answer": answer, "top_token_by_layer": tokens, "first_correct_layer": layer})
    save_results(__file__, {"prompts": entries})


if __name__ == "__main__":
    main()
