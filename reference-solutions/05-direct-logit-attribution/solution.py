"""05 — Direct logit attribution: split a logit difference across heads, MLPs and embeddings."""

import einops
import torch
from transformer_lens import HookedTransformer

from common.models import load_gpt2

PROMPT = "When Mary and John went to the store, John gave a drink to"
CORRECT, INCORRECT = " Mary", " John"


def logit_diff_attribution(model: HookedTransformer, prompt: str, correct: str, incorrect: str) -> dict[str, float]:
    """Each component's direct contribution to logit(correct) - logit(incorrect) at the final position.

    Keys: "embed", "pos_embed", "L{layer}H{head}" for every head, "L{layer}_mlp" for every MLP,
    and "bias" for all bias terms combined (attention b_O, unembed b_U). Values must sum to the true logit diff.
    """
    correct_id, incorrect_id = model.to_single_token(correct), model.to_single_token(incorrect)
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        direction = model.W_U[:, correct_id] - model.W_U[:, incorrect_id]
        scale = cache["ln_final.hook_scale"][0, -1]

        components: dict[str, torch.Tensor] = {
            "embed": cache["hook_embed"][0, -1],
            "pos_embed": cache["hook_pos_embed"][0, -1],
            "bias": model.b_O.sum(0),
        }
        for layer in range(model.cfg.n_layers):
            z = cache[f"blocks.{layer}.attn.hook_z"][0, -1]
            heads = einops.einsum(z, model.W_O[layer], "head d_head, head d_head d_model -> head d_model")
            for head in range(model.cfg.n_heads):
                components[f"L{layer}H{head}"] = heads[head]
            components[f"L{layer}_mlp"] = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]

        attribution = {}
        for name, vector in components.items():
            centred = vector - vector.mean()
            attribution[name] = (centred / scale @ direction).item()
        attribution["bias"] += (model.b_U[correct_id] - model.b_U[incorrect_id]).item()
    return attribution


if __name__ == "__main__":
    model = load_gpt2()
    attribution = logit_diff_attribution(model, PROMPT, CORRECT, INCORRECT)
    for name, value in sorted(attribution.items(), key=lambda kv: -abs(kv[1]))[:10]:
        print(f"{name:>10}: {value:+.3f}")
    print(f"sum: {sum(attribution.values()):.4f}")
