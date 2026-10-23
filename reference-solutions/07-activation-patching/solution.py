"""07 — Activation patching on IOI: patch clean residual stream into the corrupted run."""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from common.data import get_ioi_pairs
from common.models import load_gpt2
from common.results import save_results


def ioi_tokens(model: HookedTransformer, pairs: list[dict[str, str]]) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"], Int[Tensor, "batch 2"]]:
    """(clean tokens, corrupt tokens, answer ids [correct, incorrect]) for a batch of IOI pairs."""
    clean = model.to_tokens([p["clean"] for p in pairs])
    corrupt = model.to_tokens([p["corrupt"] for p in pairs])
    answers = torch.tensor([[model.to_single_token(p["correct"]), model.to_single_token(p["incorrect"])] for p in pairs])
    return clean, corrupt, answers.to(model.cfg.device)


def logit_diff(logits: Float[Tensor, "batch seq d_vocab"], answers: Int[Tensor, "batch 2"]) -> float:
    """Mean over the batch of logit(correct) - logit(incorrect) at the final position."""
    final = logits[:, -1]
    return (final.gather(1, answers[:, :1]) - final.gather(1, answers[:, 1:])).mean().item()


def patch_residual_grid(
    model: HookedTransformer, clean: Int[Tensor, "batch seq"], corrupt: Int[Tensor, "batch seq"], answers: Int[Tensor, "batch 2"]
) -> Float[Tensor, "layer pos"]:
    """Recovery from patching clean resid_pre at each (layer, position) into the corrupted run.

    recovery = (patched - corrupt) / (clean - corrupt), using logit_diff. 1 = fully restored, 0 = no effect.
    """
    clean_logits, clean_cache = model.run_with_cache(clean, names_filter=lambda n: n.endswith("hook_resid_pre"))
    clean_diff = logit_diff(clean_logits, answers)
    corrupt_diff = logit_diff(model(corrupt), answers)
    grid = torch.zeros(model.cfg.n_layers, clean.shape[1])
    for layer in range(model.cfg.n_layers):
        name = f"blocks.{layer}.hook_resid_pre"
        for pos in range(clean.shape[1]):

            def hook(resid: Float[Tensor, "batch pos d_model"], hook: HookPoint) -> Float[Tensor, "batch pos d_model"]:
                resid[:, pos] = clean_cache[name][:, pos]
                return resid

            patched = logit_diff(model.run_with_hooks(corrupt, fwd_hooks=[(name, hook)]), answers)
            grid[layer, pos] = (patched - corrupt_diff) / (clean_diff - corrupt_diff)
    return grid


def main() -> None:
    model = load_gpt2()
    clean, corrupt, answers = ioi_tokens(model, get_ioi_pairs())
    with torch.no_grad():
        grid = patch_residual_grid(model, clean, corrupt, answers)
    name_positions = (clean[0] != corrupt[0]).nonzero().flatten().tolist()
    save_results(
        __file__,
        {
            "tokens": model.to_str_tokens(clean[0]),
            "name_positions": name_positions,
            "recovery_grid": grid.tolist(),
        },
    )


if __name__ == "__main__":
    main()
