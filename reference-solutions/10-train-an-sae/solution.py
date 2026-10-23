"""10 — Train a TopK sparse autoencoder on GPT-2 small's layer-6 residual stream. GPU recommended."""

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
K = 32
SEQ_LEN = 128
BATCH_SEQS = 32
N_TRAIN_TOKENS = 3_000_000
N_EVAL_SEQS = 64
LR = 4e-4
SEED = 0


class SAE(nn.Module):
    """TopK sparse autoencoder: keep the k largest pre-activations, decode with unit-norm decoder rows."""

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.k = k
        self.W_dec = nn.Parameter(nn.functional.normalize(torch.randn(d_sae, d_model), dim=-1))
        self.W_enc = nn.Parameter(self.W_dec.data.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, "n d_sae"]:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, top.indices, top.values.relu())

    def decode(self, f: Float[Tensor, "n d_sae"]) -> Float[Tensor, "n d_model"]:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, "n d_model"]:
        return self.decode(self.encode(x))


def token_batches(model: HookedTransformer, texts: list[str], seq_len: int, batch_seqs: int):
    """Yield [batch_seqs, seq_len] token tensors: each text cut into BOS-prefixed chunks, partial chunks dropped."""
    batch, bos = [], torch.tensor([model.tokenizer.bos_token_id])
    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=False)[0].cpu()
        for start in range(0, len(tokens) - seq_len + 2, seq_len - 1):
            batch.append(torch.cat([bos, tokens[start : start + seq_len - 1]]))
            if len(batch) == batch_seqs:
                yield torch.stack(batch).to(model.cfg.device)
                batch = []


def residuals(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "n d_model"]:
    """Layer-LAYER residual stream for every non-BOS token, flattened."""
    resid = model(tokens, stop_at_layer=LAYER)
    return resid[:, 1:].reshape(-1, model.cfg.d_model)


def train_sae(model: HookedTransformer, batches, n_tokens: int) -> tuple[SAE, float]:
    """Train on activations streamed from batches until n_tokens are seen. Returns (sae, input scale)."""
    sae = SAE(model.cfg.d_model, EXPANSION * model.cfg.d_model, K).to(model.cfg.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR, betas=(0.9, 0.999))
    scale, seen, start = None, 0, time.time()
    for tokens in batches:
        with torch.no_grad():
            x = residuals(model, tokens)
        if scale is None:
            scale = x.norm(dim=-1).mean().item()
            sae.b_dec.data = x.median(0).values / scale
        x = x / scale
        loss = (sae(x) - x).pow(2).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            sae.W_dec.data = nn.functional.normalize(sae.W_dec.data, dim=-1)
        seen += len(x)
        if seen // len(x) % 100 == 0:
            print(f"{seen / 1e6:.2f}M tokens, mse {loss.item():.4f}, {time.time() - start:.0f}s")
        if seen >= n_tokens:
            break
    return sae, scale


def evaluate(model: HookedTransformer, sae: SAE, scale: float, eval_tokens: Int[Tensor, "batch seq"]) -> dict[str, float]:
    """Mean L0, clean loss, and loss with the SAE reconstruction spliced in (BOS position left untouched)."""

    def splice(resid: Float[Tensor, "batch pos d_model"], hook: HookPoint) -> Float[Tensor, "batch pos d_model"]:
        resid[:, 1:] = sae(resid[:, 1:] / scale) * scale
        return resid

    with torch.no_grad():
        x = residuals(model, eval_tokens) / scale
        l0 = (sae.encode(x) > 0).float().sum(-1).mean().item()
        clean = model(eval_tokens, return_type="loss").item()
        spliced = model.run_with_hooks(eval_tokens, return_type="loss", fwd_hooks=[(HOOK, splice)]).item()
    return {"l0": l0, "clean_loss": clean, "spliced_loss": spliced}


def main() -> None:
    torch.manual_seed(SEED)
    model = load_gpt2()
    texts = load_pile_texts(10_000)
    eval_texts, train_texts = texts[:200], texts[200:]
    eval_tokens = next(token_batches(model, eval_texts, SEQ_LEN, N_EVAL_SEQS))
    sae, scale = train_sae(model, token_batches(model, train_texts, SEQ_LEN, BATCH_SEQS), N_TRAIN_TOKENS)
    metrics = evaluate(model, sae, scale, eval_tokens)
    print(metrics)
    save_results(
        __file__,
        {"hook": HOOK, "d_model": model.cfg.d_model, "d_sae": sae.W_dec.shape[0], "n_train_tokens": N_TRAIN_TOKENS, **metrics},
    )


if __name__ == "__main__":
    main()
