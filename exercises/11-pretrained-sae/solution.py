"""11 — Use a pretrained GPT-2 small SAE: find a feature, interpret it, steer with it."""

import torch
from jaxtyping import Float
from sae_lens import SAE
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from common.data import load_pile_texts
from common.device import get_device
from common.models import load_gpt2
from common.results import save_results

RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.7.hook_resid_pre"
STEER_PROMPT = "I went out yesterday and"
N_TOP_EXAMPLES = 10
N_SCAN_DOCS = 300
SEQ_LEN = 128
N_NEW_TOKENS = 40
SEED = 0


def load_model_and_sae() -> tuple[HookedTransformer, SAE]:
    """GPT-2 small and the SAE, loaded with the model settings the SAE was trained against."""
    raise NotImplementedError


def feature_acts(model: HookedTransformer, sae: SAE, text: str) -> Float[Tensor, "seq d_sae"]:
    """SAE feature activations at every token of text (BOS dropped)."""
    raise NotImplementedError


def pick_feature(model: HookedTransformer, sae: SAE) -> int:
    """Feature that fires on the most concept-text tokens relative to neutral-text tokens.

    Firing frequency favours features spread across the whole concept, over ones that spike on a single token.
    """
    raise NotImplementedError


def top_activating_examples(model: HookedTransformer, sae: SAE, feature: int, texts: list[str], n: int) -> list[dict[str, str | float]]:
    """The n tokens with the highest activation of feature across texts, each with ~10 tokens of left context."""
    raise NotImplementedError


def generate(model: HookedTransformer, sae: SAE, feature: int, coeff: float) -> str:
    """Sampled continuation of STEER_PROMPT, adding coeff * the feature's decoder direction at the SAE hook."""
    raise NotImplementedError


def mean_activation(model: HookedTransformer, sae: SAE, feature: int, text: str) -> float:
    """Mean activation of feature over the generated part of text, re-encoded with no steering."""
    raise NotImplementedError


def main() -> None:
    """Pick, interpret and steer a feature, then save results.json with feature, interpretation, top_examples, unsteered_text, steered_text, unsteered_activation, steered_activation."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
