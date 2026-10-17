"""02 — Train a 2-layer attention-only transformer on repeated random tokens."""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig

from common.results import save_results

VOCAB_SIZE = 64
HALF_LEN = 24
SEED = 0


def make_repeated_tokens(batch: int, half_len: int, vocab_size: int, generator: torch.Generator) -> Int[Tensor, "batch seq"]:
    """Random tokens of length half_len, then the same tokens again (total 2 * half_len)."""
    raise NotImplementedError


def build_model(vocab_size: int, n_ctx: int) -> HookedTransformer:
    """A 2-layer, attention-only HookedTransformer (no MLPs)."""
    raise NotImplementedError


def per_position_loss(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "seq_minus_1"]:
    """Cross-entropy (nats) for predicting tokens[:, i + 1], averaged over the batch. Index i is the predicting position."""
    raise NotImplementedError


def train(model: HookedTransformer, n_steps: int, batch_size: int, half_len: int, generator: torch.Generator) -> None:
    """Train in place on fresh repeated-token batches."""
    raise NotImplementedError


def half_losses(model: HookedTransformer, half_len: int, generator: torch.Generator) -> tuple[float, float]:
    """(first-half loss, second-half loss) on a fresh batch.

    First half: predicting tokens 1..half_len-1. Second half: predicting tokens half_len+1..end
    (the first repeated token is excluded, since nothing signals the repeat has started).
    """
    raise NotImplementedError


def main() -> None:
    """Train, then save results.json with vocab_size, first_half_loss, second_half_loss."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
