"""03 — Zero- and mean-ablate every attention head of GPT-2 small with hooks."""
from torchgen.api.cpp import return_type

from collections.abc import Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from transformer_lens.hook_points import HookPoint
from typing import cast

from common.data import load_pile_texts
from common.models import load_gpt2
from common.results import save_results

N_SEQS = 8
SEQ_LEN = 128

HeadHook = Callable[[Float[Tensor, "batch pos head d_head"], HookPoint], Float[Tensor, "batch pos head d_head"]]


def get_sample_tokens(model: HookedTransformer, n_seqs: int, seq_len: int) -> Int[Tensor, "batch seq"]:
    """n_seqs pile documents, each truncated to seq_len tokens (BOS included)."""
    seqs = load_pile_texts(n_seqs, 1000);
    tokenized = model.to_tokens(seqs);
    return tokenized[:, :seq_len];


def zero_ablation_hook(head: int) -> HeadHook:
    # Zeros out all the values at the head's position
    def hook(z: Float[Tensor, "batch pos head d_head"], hook: HookPoint) -> Float[Tensor, "batch pos head d_head"]:
        z[:, :, head, :] = 0;
        return z;

    return hook;

def mean_ablation_hook(head: int, mean_z: Float[Tensor, "head d_head"]) -> HeadHook:
    """Hook for hook_z that replaces one head's output with its mean over batch and positions."""
    def hook(z: Float[Tensor, "batch pos head d_head"], hook: HookPoint) -> Float[Tensor, "batch pos head d_head"]:
        z[:, :, head, :] = mean_z[head, :];
        return z;

    return hook;

def ablation_loss_increases(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> tuple[float, dict[str, dict[str, float]]]:
    """Loss increase over clean for every head, under zero and mean ablation.

    Returns {"zero_ablation": {"L0H0": delta, ...}, "mean_ablation": {...}} with 144 entries each.
    """
    clean_loss, activations = model.run_with_cache(tokens, return_type="loss")
    clean_loss = cast(Tensor, clean_loss).item()
    results = {"zero_ablation": {}, "mean_ablation": {}}

    for head in range(model.cfg.n_heads):
        zero_hook = zero_ablation_hook(head)

        for layer in range(model.cfg.n_layers):
            act_name = utils.get_act_name("z", layer)

            zero_loss = model.run_with_hooks(tokens, fwd_hooks=[(act_name, zero_hook)], return_type="loss")
            zero_loss = cast(Tensor, zero_loss).item()

            acts: Float[Tensor, "batch pos head d_head"] = activations[act_name]
            mean_z: Float[Tensor, "head d_head"] = acts.mean(dim=(0, 1))
            mean_hook = mean_ablation_hook(head, mean_z)
            mean_loss = model.run_with_hooks(tokens, fwd_hooks=[(act_name, mean_hook)], return_type="loss")
            mean_loss = cast(Tensor, mean_loss).item()

            head_key = f"L{layer}H{head}"
            results["zero_ablation"][head_key] = zero_loss - clean_loss
            results["mean_ablation"][head_key] = mean_loss - clean_loss

    return (clean_loss, results)


def main() -> None:
    """Run every ablation, then save results.json with clean_loss, zero_ablation, mean_ablation."""
    model = load_gpt2()
    sample_tokens = get_sample_tokens(model, N_SEQS, SEQ_LEN)
    clean_loss, ablation_losses = ablation_loss_increases(model, sample_tokens)
    results = {"clean_loss": clean_loss, **ablation_losses}
    save_results(__file__, results)

if __name__ == "__main__":
    main()
