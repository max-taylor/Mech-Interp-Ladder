"""07 — Activation patching on IOI: patch clean residual stream into the corrupted run."""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from common.data import get_ioi_pairs
from common.models import load_gpt2
from common.results import save_results


def ioi_tokens(model: HookedTransformer, pairs: list[dict[str, str]]) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"], Int[Tensor, "batch 2"]]:
    """(clean tokens, corrupt tokens, answer ids [correct, incorrect]) for a batch of IOI pairs."""
    raise NotImplementedError


def logit_diff(logits: Float[Tensor, "batch seq d_vocab"], answers: Int[Tensor, "batch 2"]) -> float:
    """Mean over the batch of logit(correct) - logit(incorrect) at the final position."""
    raise NotImplementedError


def patch_residual_grid(
    model: HookedTransformer, clean: Int[Tensor, "batch seq"], corrupt: Int[Tensor, "batch seq"], answers: Int[Tensor, "batch 2"]
) -> Float[Tensor, "layer pos"]:
    """Recovery from patching clean resid_pre at each (layer, position) into the corrupted run.

    recovery = (patched - corrupt) / (clean - corrupt), using logit_diff. 1 = fully restored, 0 = no effect.
    """
    raise NotImplementedError


def main() -> None:
    """Build the grid, then save results.json with tokens, name_positions, recovery_grid."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
