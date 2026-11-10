"""10 — Train a sparse autoencoder on GPT-2 small's layer-6 residual stream. GPU recommended."""

import time

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from common.data import load_pile_texts
from common.models import load_gpt2
from common.results import save_results

LAYER = 6
HOOK = f"blocks.{LAYER}.hook_resid_pre"
EXPANSION = 8
SEQ_LEN = 128
N_TRAIN_TOKENS = 1_000_000
N_EVAL_SEQS = 64
SEED = 0


class SAE(nn.Module):
    """Sparse autoencoder over d_model-dim activations with a d_sae-wide dictionary."""

    def __init__(self, d_model: int, d_sae: int):
        raise NotImplementedError

    def encode(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, "n d_sae"]:
        raise NotImplementedError

    def decode(self, f: Float[Tensor, "n d_sae"]) -> Float[Tensor, "n d_model"]:
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, "n d_model"]:
        raise NotImplementedError


def token_batches(model: HookedTransformer, texts: list[str], seq_len: int, batch_seqs: int):
    """Yield [batch_seqs, seq_len] token tensors: each text cut into BOS-prefixed chunks, partial chunks dropped."""
    raise NotImplementedError


def residuals(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "n d_model"]:
    """Layer-LAYER residual stream for every non-BOS token, flattened."""
    raise NotImplementedError


def train_sae(model: HookedTransformer, batches, n_tokens: int) -> tuple[SAE, float]:
    """Train on activations streamed from batches until n_tokens are seen. Returns (sae, input scale)."""
    raise NotImplementedError


def evaluate(model: HookedTransformer, sae: SAE, scale: float, eval_tokens: Int[Tensor, "batch seq"]) -> dict[str, float]:
    """Mean L0, clean loss, and loss with the SAE reconstruction spliced in (BOS position left untouched)."""
    raise NotImplementedError


def main() -> None:
    """Train and evaluate, then save results.json with d_model, d_sae, n_train_tokens, l0, clean_loss, spliced_loss."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
