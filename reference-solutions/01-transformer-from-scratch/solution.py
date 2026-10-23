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
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        x = x - x.mean(-1, keepdim=True)
        scale = (x.pow(2).mean(-1, keepdim=True) + self.cfg.layer_norm_eps).sqrt()
        return x / scale * self.w + self.b


class Embed(nn.Module):
    """Params: W_E [d_vocab, d_model]."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.W_E = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model))

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    """Params: W_pos [n_ctx, d_model]."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model))

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "pos d -> batch pos d", batch=batch)


class Attention(nn.Module):
    """Causal multi-head attention.

    Params: W_Q, W_K, W_V [n_heads, d_model, d_head]; b_Q, b_K, b_V [n_heads, d_head];
    W_O [n_heads, d_head, d_model]; b_O [d_model].
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        for name in ("Q", "K", "V"):
            setattr(self, f"W_{name}", nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head)))
            setattr(self, f"b_{name}", nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head)))
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        q = einops.einsum(x, self.W_Q, "b p d, h d e -> b p h e") + self.b_Q
        k = einops.einsum(x, self.W_K, "b p d, h d e -> b p h e") + self.b_K
        v = einops.einsum(x, self.W_V, "b p d, h d e -> b p h e") + self.b_V
        scores = einops.einsum(q, k, "b q h e, b k h e -> b h q k") / self.cfg.d_head**0.5
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        pattern = scores.masked_fill(mask, float("-inf")).softmax(-1)
        z = einops.einsum(pattern, v, "b h q k, b k h e -> b q h e")
        return einops.einsum(z, self.W_O, "b q h e, h e d -> b q d") + self.b_O


class MLP(nn.Module):
    """Params: W_in [d_model, d_mlp], b_in [d_mlp], W_out [d_mlp, d_model], b_out [d_model]. GELU (tanh approx)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        hidden = nn.functional.gelu(x @ self.W_in + self.b_in, approximate="tanh")
        return hidden @ self.W_out + self.b_out


class TransformerBlock(nn.Module):
    """Pre-LN block. Submodules: ln1, attn, ln2, mlp."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        resid = resid + self.attn(self.ln1(resid))
        return resid + self.mlp(self.ln2(resid))


class Unembed(nn.Module):
    """Params: W_U [d_model, d_vocab], b_U [d_vocab]."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.W_U = nn.Parameter(torch.empty(cfg.d_model, cfg.d_vocab))
        self.b_U = nn.Parameter(torch.zeros(cfg.d_vocab))

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_vocab"]:
        return x @ self.W_U + self.b_U


class Transformer(nn.Module):
    """Submodules: embed, pos_embed, blocks (nn.ModuleList), ln_final, unembed."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_vocab"]:
        resid = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            resid = block(resid)
        return self.unembed(self.ln_final(resid))


if __name__ == "__main__":
    from common.models import load_gpt2

    reference = load_gpt2("cpu", processing=False)
    model = Transformer(Config())
    model.load_state_dict(reference.state_dict(), strict=False)
    tokens = reference.to_tokens("The quick brown fox jumps over the lazy dog.")
    diff = (model(tokens) - reference(tokens)).abs().max().item()
    print(f"max |logit diff| = {diff:.2e}")
