"""03 — Zero- and mean-ablate every attention head of GPT-2 small with hooks."""

from collections.abc import Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from common.data import load_pile_texts
from common.models import load_gpt2
from common.results import save_results

N_SEQS = 8
SEQ_LEN = 128

HeadHook = Callable[[Float[Tensor, "batch pos head d_head"], HookPoint], Float[Tensor, "batch pos head d_head"]]


def get_sample_tokens(model: HookedTransformer, n_seqs: int, seq_len: int) -> Int[Tensor, "batch seq"]:
    """n_seqs pile documents, each truncated to seq_len tokens (BOS included)."""
    texts = load_pile_texts(n_seqs)
    tokens = torch.stack([model.to_tokens(t)[0, :seq_len] for t in texts])
    return tokens.to(model.cfg.device)


def zero_ablation_hook(head: int) -> HeadHook:
    """Hook for hook_z that sets one head's output to zero."""

    def hook(z: Float[Tensor, "batch pos head d_head"], hook: HookPoint) -> Float[Tensor, "batch pos head d_head"]:
        z[:, :, head] = 0.0
        return z

    return hook


def mean_ablation_hook(head: int, mean_z: Float[Tensor, "head d_head"]) -> HeadHook:
    """Hook for hook_z that replaces one head's output with its mean over batch and positions."""

    def hook(z: Float[Tensor, "batch pos head d_head"], hook: HookPoint) -> Float[Tensor, "batch pos head d_head"]:
        z[:, :, head] = mean_z[head]
        return z

    return hook


def ablation_loss_increases(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> dict[str, dict[str, float]]:
    """Loss increase over clean for every head, under zero and mean ablation.

    Returns {"zero_ablation": {"L0H0": delta, ...}, "mean_ablation": {...}} with 144 entries each.
    """
    with torch.no_grad():
        clean_loss, cache = model.run_with_cache(tokens, return_type="loss", names_filter=lambda n: n.endswith("hook_z"))
        results: dict[str, dict[str, float]] = {"zero_ablation": {}, "mean_ablation": {}}
        for layer in range(model.cfg.n_layers):
            name = f"blocks.{layer}.attn.hook_z"
            mean_z = cache[name].mean(dim=(0, 1))
            for head in range(model.cfg.n_heads):
                for kind, hook in (("zero_ablation", zero_ablation_hook(head)), ("mean_ablation", mean_ablation_hook(head, mean_z))):
                    loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(name, hook)])
                    results[kind][f"L{layer}H{head}"] = (loss - clean_loss).item()
    return {"clean_loss": clean_loss.item(), **results}


def main() -> None:
    model = load_gpt2()
    tokens = get_sample_tokens(model, N_SEQS, SEQ_LEN)
    save_results(__file__, ablation_loss_increases(model, tokens))


if __name__ == "__main__":
    main()
