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
    raise NotImplementedError


def induction_scores(model: HookedTransformer, tokens: Int[Tensor, "batch seq"], half_len: int) -> Float[Tensor, "layer head"]:
    """Mean attention each head pays from a second-half token to the token just after its earlier copy."""
    raise NotImplementedError


def repeated_half_loss(model: HookedTransformer, tokens: Int[Tensor, "batch seq"], half_len: int, heads: list[tuple[int, int]]) -> float:
    """Mean loss on the second (repeated) half, with the given (layer, head)s zero-ablated."""
    raise NotImplementedError


def main() -> None:
    """Score, rank and ablate heads, then save results.json with induction_scores, top_heads, top_heads_loss_increase, random_heads_loss_increase."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
