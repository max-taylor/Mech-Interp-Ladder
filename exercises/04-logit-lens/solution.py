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
    raise NotImplementedError


def first_correct_layer(top_tokens: list[str], answer: str) -> int | None:
    """Index of the first depth whose top-1 token is the answer, or None if it never is."""
    raise NotImplementedError


def main() -> None:
    """Run every prompt in FACTUAL_PROMPTS, then save results.json with "prompts": [{prompt, answer, top_token_by_layer, first_correct_layer}]."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
