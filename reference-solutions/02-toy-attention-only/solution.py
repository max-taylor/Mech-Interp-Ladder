"""02 — Train a 2-layer attention-only transformer on repeated random tokens."""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig

from common.results import save_results

VOCAB_SIZE = 64
HALF_LEN = 24
N_STEPS = 3000
BATCH_SIZE = 64
SEED = 0


def make_repeated_tokens(batch: int, half_len: int, vocab_size: int, generator: torch.Generator) -> Int[Tensor, "batch seq"]:
    """Random tokens of length half_len, then the same tokens again (total 2 * half_len)."""
    first = torch.randint(0, vocab_size, (batch, half_len), generator=generator)
    return torch.cat([first, first], dim=1)


def build_model(vocab_size: int, n_ctx: int) -> HookedTransformer:
    """A 2-layer, attention-only HookedTransformer (no MLPs)."""
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=128,
        d_head=32,
        n_heads=4,
        n_ctx=n_ctx,
        d_vocab=vocab_size,
        attn_only=True,
        normalization_type="LN",
        act_fn=None,
        seed=SEED,
    )
    return HookedTransformer(cfg)


def per_position_loss(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "seq_minus_1"]:
    """Cross-entropy (nats) for predicting tokens[:, i + 1], averaged over the batch. Index i is the predicting position."""
    return model(tokens, return_type="loss", loss_per_token=True).mean(0)


def train(model: HookedTransformer, n_steps: int, batch_size: int, half_len: int, generator: torch.Generator) -> None:
    """Train in place on fresh repeated-token batches."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    for step in range(n_steps):
        tokens = make_repeated_tokens(batch_size, half_len, model.cfg.d_vocab, generator)
        loss = model(tokens, return_type="loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.3f}")


def half_losses(model: HookedTransformer, half_len: int, generator: torch.Generator) -> tuple[float, float]:
    """(first-half loss, second-half loss) on a fresh batch.

    First half: predicting tokens 1..half_len-1. Second half: predicting tokens half_len+1..end
    (the first repeated token is excluded, since nothing signals the repeat has started).
    """
    tokens = make_repeated_tokens(512, half_len, model.cfg.d_vocab, generator)
    with torch.no_grad():
        losses = per_position_loss(model, tokens)
    return losses[: half_len - 1].mean().item(), losses[half_len:].mean().item()


def main() -> None:
    generator = torch.Generator().manual_seed(SEED)
    model = build_model(VOCAB_SIZE, 2 * HALF_LEN)
    train(model, N_STEPS, BATCH_SIZE, HALF_LEN, generator)
    first, second = half_losses(model, HALF_LEN, generator)
    save_results(
        __file__,
        {"vocab_size": VOCAB_SIZE, "first_half_loss": first, "second_half_loss": second},
    )


if __name__ == "__main__":
    main()
