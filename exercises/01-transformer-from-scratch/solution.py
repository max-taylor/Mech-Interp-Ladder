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
        super().__init__();
        self.cfg: Config = cfg
        self.w: nn.Parameter = nn.Parameter(torch.empty(cfg.d_model))
        self.b: nn.Parameter = nn.Parameter(torch.empty(cfg.d_model))

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        # Formula is: (x - mean(x)) / std(x) . y + B
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (var + self.cfg.layer_norm_eps).sqrt() * self.w + self.b


class Embed(nn.Module):
    """Params: W_E [d_vocab, d_model]."""

    def __init__(self, cfg: Config):
        super().__init__();
        self.W_E: nn.Parameter = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model));

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_model"]:
        return self.W_E[tokens]

class PosEmbed(nn.Module):
    """Params: W_pos [n_ctx, d_model]."""

    def __init__(self, cfg: Config):
        super().__init__();
        self.W_pos: nn.Parameter = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model));

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_model"]:
        batch, pos = tokens.shape
        rows = self.W_pos[:pos]
        repeated = einops.repeat(rows, "pos d_model -> batch pos d_model", batch=batch)
        return repeated


class Attention(nn.Module):
    """Causal multi-head attention.

    Params: W_Q, W_K, W_V [n_heads, d_model, d_head]; b_Q, b_K, b_V [n_heads, d_head];
    W_O [n_heads, d_head, d_model]; b_O [d_model].
    """

    def __init__(self, cfg: Config):
        super().__init__();
        self.cfg = cfg;
        self.W_Q: nn.Parameter = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head));
        self.W_K: nn.Parameter = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head));
        self.W_V: nn.Parameter = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head));
        
        self.b_Q: nn.Parameter = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head));
        self.b_K: nn.Parameter = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head));
        self.b_V: nn.Parameter = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head));

        self.W_O: nn.Parameter = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model));

        self.b_O: nn.Parameter = nn.Parameter(torch.empty(cfg.d_model));

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        q = einops.einsum(x, self.W_Q, "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head") + self.b_Q
        k = einops.einsum(x, self.W_K, "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head") + self.b_K
        v = einops.einsum(x, self.W_V, "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head") + self.b_V

        # Calculate the q and k scores
        # Omit d_head here, so we take the dot product of a q and k
        scores: Float[Tensor, "batch n_heads pos_q pos_k"] = einops.einsum(q, k, "batch pos_q n_heads d_head, batch pos_k n_heads d_head -> batch n_heads pos_q pos_k") / self.cfg.d_head ** 0.5;

        # Scores shape: 0-batch, 1-n_heads, 2-pos_q, 3-pos_k
        # -1 is pos_k
        pos = scores.shape[-1]

        # Zero out values that are "above the diagonal" this prevents a look-ahead.
        mask = torch.triu(torch.ones(pos, pos, dtype=torch.bool, device=scores.device), diagonal=1)
        # Sets values above the diagonal to be very large negatives
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax the values, the large negatives become zero. the large positives become 1
        pattern = scores.softmax(dim=-1)

        z = einops.einsum(v, pattern, "batch pos_k n_heads d_head, batch n_heads pos_q pos_k -> batch pos_q n_heads d_head")

        return einops.einsum(z, self.W_O, "batch pos n_heads d_head, n_heads d_head d_model -> batch pos d_model") + self.b_O


class MLP(nn.Module):
    """Params: W_in [d_model, d_mlp], b_in [d_mlp], W_out [d_mlp, d_model], b_out [d_model]. GELU (tanh approx)."""

    def __init__(self, cfg: Config):
        super().__init__();
        self.W_in: nn.Parameter = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp));
        self.b_in: nn.Parameter = nn.Parameter(torch.empty(cfg.d_mlp));
        self.W_out: nn.Parameter = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model));
        self.b_out: nn.Parameter = nn.Parameter(torch.empty(cfg.d_model));

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        expanded = einops.einsum(x, self.W_in, "batch pos d_model, d_model d_mlp -> batch pos d_mlp") + self.b_in;
        gelu_applied = torch.nn.functional.gelu(expanded, approximate="tanh");
        return einops.einsum(gelu_applied, self.W_out, "batch pos d_mlp, d_mlp d_model -> batch pos d_model") + self.b_out;


class TransformerBlock(nn.Module):
    """Pre-LN block. Submodules: ln1, attn, ln2, mlp."""

    def __init__(self, cfg: Config):
        super().__init__();
        self.ln1: LayerNorm = LayerNorm(cfg);
        self.attn: Attention = Attention(cfg);
        self.ln2: LayerNorm = LayerNorm(cfg);
        self.mlp: MLP = MLP(cfg);

    def forward(self, resid: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_model"]:
        resid_mid: Float[Tensor, "batch pos d_model"] = resid + self.attn(self.ln1(resid));
        resid_post: Float[Tensor, "batch pos d_model"] = resid_mid + self.mlp(self.ln2(resid_mid));
        return resid_post;


class Unembed(nn.Module):
    """Params: W_U [d_model, d_vocab], b_U [d_vocab]."""

    def __init__(self, cfg: Config):
        super().__init__();
        self.W_U: nn.Parameter = nn.Parameter(torch.empty(cfg.d_model, cfg.d_vocab));
        self.b_U: nn.Parameter = nn.Parameter(torch.empty(cfg.d_vocab));

    def forward(self, x: Float[Tensor, "batch pos d_model"]) -> Float[Tensor, "batch pos d_vocab"]:
        out = einops.einsum(x, self.W_U, "batch pos d_model, d_model d_vocab -> batch pos d_vocab")
        return out + self.b_U


class Transformer(nn.Module):
    """Submodules: embed, pos_embed, blocks (nn.ModuleList), ln_final, unembed."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = Embed(cfg);
        self.pos_embed = PosEmbed(cfg);
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers));

        self.ln_final = LayerNorm(cfg);
        self.unembed = Unembed(cfg);

    def forward(self, tokens: Int[Tensor, "batch pos"]) -> Float[Tensor, "batch pos d_vocab"]:
        resid = self.embed(tokens) + self.pos_embed(tokens);
        for layer in self.blocks:
            resid = layer(resid);
        ln = self.ln_final(resid);

        return self.unembed(ln);


if __name__ == "__main__":
    from common.models import load_gpt2

    reference = load_gpt2("cpu", processing=False)
    model = Transformer(Config())
    _ = model.load_state_dict(reference.state_dict(), strict=False)
    tokens = reference.to_tokens("The quick brown fox jumps over the lazy dog.")
    diff = (model(tokens) - reference(tokens)).abs().max().item()
    print(f"max |logit diff| = {diff:.2e}")
