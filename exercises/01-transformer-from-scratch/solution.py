"""01 — GPT-2 small from scratch.

Parameter names and shapes match TransformerLens, so
`model.load_state_dict(hooked.state_dict(), strict=False)` loads GPT-2's weights.
"""

from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


@dataclass
class Config:
    d_model: int = 768
    d_vocab: int = 50257
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    layer_norm_eps: float = 1e-5


class LayerNorm(nn.Module):
    """Params: w [d_model], b [d_model]."""

    def __init__(self, cfg: Config):
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        raise NotImplementedError


class Embed(nn.Module):
    """Params: W_E [d_vocab, d_model]."""

    def __init__(self, cfg: Config):
        raise NotImplementedError

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_model"]:
        raise NotImplementedError


class PosEmbed(nn.Module):
    """Params: W_pos [n_ctx, d_model]."""

    def __init__(self, cfg: Config):
        raise NotImplementedError

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_model"]:
        raise NotImplementedError


class Attention(nn.Module):
    """Causal multi-head attention.

    Params: W_Q, W_K, W_V [n_heads, d_model, d_head]; b_Q, b_K, b_V [n_heads, d_head];
    W_O [n_heads, d_head, d_model]; b_O [d_model].
    """

    def __init__(self, cfg: Config):
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        raise NotImplementedError


class MLP(nn.Module):
    """Params: W_in [d_model, d_mlp], b_in [d_mlp], W_out [d_mlp, d_model], b_out [d_model]. GELU (tanh approx)."""

    def __init__(self, cfg: Config):
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        raise NotImplementedError


class TransformerBlock(nn.Module):
    """Pre-LN block. Submodules: ln1, attn, ln2, mlp."""

    def __init__(self, cfg: Config):
        raise NotImplementedError

    def forward(self, resid: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        raise NotImplementedError


class Unembed(nn.Module):
    """Params: W_U [d_model, d_vocab], b_U [d_vocab]."""

    def __init__(self, cfg: Config):
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_vocab"]:
        raise NotImplementedError


class Transformer(nn.Module):
    """Submodules: embed, pos_embed, blocks (nn.ModuleList), ln_final, unembed."""

    def __init__(self, cfg: Config):
        raise NotImplementedError

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_vocab"]:
        raise NotImplementedError


if __name__ == "__main__":
    from common.models import load_gpt2

    reference = load_gpt2("cpu", processing=False)
    model = Transformer(Config())
    model.load_state_dict(reference.state_dict(), strict=False)
    tokens = reference.to_tokens("The quick brown fox jumps over the lazy dog.")
    diff = (model(tokens) - reference(tokens)).abs().max().item()
    print(f"max |logit diff| = {diff:.2e}")
