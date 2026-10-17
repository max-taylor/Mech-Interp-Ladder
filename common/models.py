import torch
from typing import Any

from transformer_lens import HookedTransformer

from common.device import get_device


def load_gpt2(device: torch.device | str | None = None, processing: bool = True, **kwargs: Any) -> HookedTransformer:
    """Load GPT-2 small as a HookedTransformer.

    processing=True folds LayerNorm weights and centres writing weights (TransformerLens default).
    processing=False keeps the raw Hugging Face weights. Extra kwargs go to the TransformerLens loader.
    """
    device = device or get_device()
    loader = HookedTransformer.from_pretrained if processing else HookedTransformer.from_pretrained_no_processing
    model = loader("gpt2", device=str(device), **kwargs)
    model.eval()
    return model
