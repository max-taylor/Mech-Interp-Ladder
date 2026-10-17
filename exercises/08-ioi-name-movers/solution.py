"""08 — Find IOI name-mover heads with head-level patching, then resample-ablate them."""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from common.data import get_ioi_pairs
from common.models import load_gpt2
from common.results import save_results

N_PAIRS = 16


def logit_diff(logits: Float[Tensor, "batch seq d_vocab"], answers: Int[Tensor, "batch 2"]) -> float:
    """Mean over the batch of logit(correct) - logit(incorrect) at the final position."""
    raise NotImplementedError


def resample_ablated_logit_diff(
    model: HookedTransformer,
    clean: Int[Tensor, "batch seq"],
    corrupt_cache: ActivationCache,
    answers: Int[Tensor, "batch 2"],
    heads: list[tuple[int, int]],
) -> float:
    """Clean-run logit diff with each head's final-position output (hook_z) replaced by its corrupted-run value."""
    raise NotImplementedError


def head_patching_scores(
    model: HookedTransformer, clean: Int[Tensor, "batch seq"], corrupt_cache: ActivationCache, answers: Int[Tensor, "batch 2"]
) -> Float[Tensor, "layer head"]:
    """Fraction of the clean logit diff lost when one head at a time is resample-ablated (1 = all lost)."""
    raise NotImplementedError


def io_attention(model: HookedTransformer, clean_cache: ActivationCache, io_position: int) -> Float[Tensor, "layer head"]:
    """Mean attention from the final position to the indirect object name, per head."""
    raise NotImplementedError


def identify_name_movers(scores: Float[Tensor, "layer head"], attention: Float[Tensor, "layer head"]) -> list[tuple[int, int]]:
    """Heads that both matter under patching and attend to the indirect object (S-inhibition heads attend elsewhere)."""
    raise NotImplementedError


def main() -> None:
    """Score heads, pick name movers, ablate them, then save results.json with head_patching_scores, identified_heads, clean_logit_diff, ablated_logit_diff."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
