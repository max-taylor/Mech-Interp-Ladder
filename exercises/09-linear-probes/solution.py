"""09 — Linear probes: is this token inside quotation marks?"""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer

from common.data import make_quote_texts
from common.models import load_gpt2
from common.results import save_results

N_TEXTS = 600
TRAIN_FRACTION = 0.8
SEED = 0


def label_tokens(model: HookedTransformer, text: str) -> tuple[Int[Tensor, "seq"], Int[Tensor, "seq"]]:
    """(tokens, labels) for one text. Label 1 = inside quotes, 0 = outside, -1 = ignore (BOS and quote tokens)."""
    raise NotImplementedError


def collect_activations(model: HookedTransformer, texts: list[str]) -> tuple[Float[Tensor, "depth n d_model"], Int[Tensor, "n"], Int[Tensor, "n"]]:
    """Residual stream at every depth (embeddings + each block) for every labelled token.

    Returns (activations, labels, text_index) where text_index says which text each token came from.
    """
    raise NotImplementedError


def train_probe(x: Float[Tensor, "n d"], y: Int[Tensor, "n"], steps: int) -> torch.nn.Linear:
    """Logistic regression probe on standardised features (mean/std baked into the returned layer)."""
    raise NotImplementedError


def accuracy(probe: torch.nn.Linear, x: Float[Tensor, "n d"], y: Int[Tensor, "n"]) -> float:
    """Fraction of tokens classified correctly."""
    raise NotImplementedError


def main() -> None:
    """Probe every depth, then save results.json with accuracy_by_layer, control_accuracy_by_layer, majority_baseline."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
