"""06 — Find GPT-2 small's induction heads, then check they matter by ablating them."""

import random

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from common.models import load_gpt2
from common.results import save_results

HALF_LEN = 50
BATCH = 16
TOP_K = 5
N_RANDOM_DRAWS = 5
SEED = 0


def make_repeated_tokens(model: HookedTransformer, batch: int, half_len: int, seed: int) -> Int[Tensor, "batch seq"]:
    """BOS, then half_len random tokens, then the same tokens again (length 1 + 2 * half_len)."""
    generator = torch.Generator().manual_seed(seed)
    random_tokens = torch.randint(0, model.cfg.d_vocab, (batch, half_len), generator=generator)
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id)
    return torch.cat([bos, random_tokens, random_tokens], dim=1).to(model.cfg.device)


def induction_scores(model: HookedTransformer, tokens: Int[Tensor, "batch seq"], half_len: int) -> Float[Tensor, "layer head"]:
    """Mean attention each head pays from a second-half token to the token just after its earlier copy."""
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_pattern"))
    scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        stripe = pattern.diagonal(offset=-(half_len - 1), dim1=-2, dim2=-1)[..., 1:]
        scores[layer] = stripe.mean(dim=(0, -1)).cpu()
    return scores


def repeated_half_loss(model: HookedTransformer, tokens: Int[Tensor, "batch seq"], half_len: int, heads: list[tuple[int, int]]) -> float:
    """Mean loss on the second (repeated) half, with the given (layer, head)s zero-ablated."""
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    def make_hook(layer_heads: list[int]):
        def hook(z: Float[Tensor, "batch pos head d_head"], hook: HookPoint) -> Float[Tensor, "batch pos head d_head"]:
            z[:, :, layer_heads] = 0.0
            return z

        return hook

    hooks = [(f"blocks.{layer}.attn.hook_z", make_hook(hs)) for layer, hs in by_layer.items()]
    loss = model.run_with_hooks(tokens, return_type="loss", loss_per_token=True, fwd_hooks=hooks)
    return loss[:, half_len + 1 :].mean().item()


def main() -> None:
    model = load_gpt2()
    tokens = make_repeated_tokens(model, BATCH, HALF_LEN, SEED)
    with torch.no_grad():
        scores = induction_scores(model, tokens, HALF_LEN)
        flat = scores.flatten().argsort(descending=True)[:TOP_K]
        top = [(int(i) // model.cfg.n_heads, int(i) % model.cfg.n_heads) for i in flat]
        clean = repeated_half_loss(model, tokens, HALF_LEN, [])
        top_increase = repeated_half_loss(model, tokens, HALF_LEN, top) - clean

        rng = random.Random(SEED)
        others = [(layer, head) for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads) if (layer, head) not in top]
        random_increases = [repeated_half_loss(model, tokens, HALF_LEN, rng.sample(others, TOP_K)) - clean for _ in range(N_RANDOM_DRAWS)]
        random_increase = sum(random_increases) / N_RANDOM_DRAWS

    print(f"top heads {top}; loss increase {top_increase:.3f} vs random {random_increase:.3f}")
    save_results(
        __file__,
        {
            "induction_scores": scores.tolist(),
            "top_heads": [f"L{layer}H{head}" for layer, head in top],
            "clean_repeated_loss": clean,
            "top_heads_loss_increase": top_increase,
            "random_heads_loss_increase": random_increase,
        },
    )


if __name__ == "__main__":
    main()
