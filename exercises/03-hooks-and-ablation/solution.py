"""03 — Zero- and mean-ablate every attention head of GPT-2 small with hooks."""

from collections.abc import Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from common.data import load_pile_texts
from common.models import load_gpt2
from common.results import save_results

N_SEQS = 8
SEQ_LEN = 128

HeadHook = Callable[[Float[Tensor, "batch pos head d_head"], HookPoint], Float[Tensor, "batch pos head d_head"]]


def get_sample_tokens(model: HookedTransformer, n_seqs: int, seq_len: int) -> Int[Tensor, "batch seq"]:
    """n_seqs pile documents, each truncated to seq_len tokens (BOS included)."""
    raise NotImplementedError


def zero_ablation_hook(head: int) -> HeadHook:
    """Hook for hook_z that sets one head's output to zero."""
    raise NotImplementedError


def mean_ablation_hook(head: int, mean_z: Float[Tensor, "head d_head"]) -> HeadHook:
    """Hook for hook_z that replaces one head's output with its mean over batch and positions."""
    raise NotImplementedError


def ablation_loss_increases(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> dict[str, dict[str, float]]:
    """Loss increase over clean for every head, under zero and mean ablation.

    Returns {"zero_ablation": {"L0H0": delta, ...}, "mean_ablation": {...}} with 144 entries each.
    """
    raise NotImplementedError


def main() -> None:
    """Run every ablation, then save results.json with clean_loss, zero_ablation, mean_ablation."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
