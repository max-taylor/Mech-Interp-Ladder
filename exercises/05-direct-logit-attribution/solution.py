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
    raise NotImplementedError


if __name__ == "__main__":
    model = load_gpt2()
    attribution = logit_diff_attribution(model, PROMPT, CORRECT, INCORRECT)
    for name, value in sorted(attribution.items(), key=lambda kv: -abs(kv[1]))[:10]:
        print(f"{name:>10}: {value:+.3f}")
    print(f"sum: {sum(attribution.values()):.4f}")
