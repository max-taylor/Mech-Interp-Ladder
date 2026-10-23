"""08 — Find IOI name-mover heads with head-level patching, then resample-ablate them."""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint

from common.data import get_ioi_pairs
from common.models import load_gpt2
from common.results import save_results

N_PAIRS = 16
MIN_PATCH_SCORE = 0.05
MIN_IO_ATTENTION = 0.2
IO_POSITION = 2  # "<bos> When A ..." -> A is token 2


def logit_diff(logits: Float[Tensor, "batch seq d_vocab"], answers: Int[Tensor, "batch 2"]) -> float:
    """Mean over the batch of logit(correct) - logit(incorrect) at the final position."""
    final = logits[:, -1]
    return (final.gather(1, answers[:, :1]) - final.gather(1, answers[:, 1:])).mean().item()


def resample_ablated_logit_diff(
    model: HookedTransformer,
    clean: Int[Tensor, "batch seq"],
    corrupt_cache: ActivationCache,
    answers: Int[Tensor, "batch 2"],
    heads: list[tuple[int, int]],
) -> float:
    """Clean-run logit diff with each head's final-position output (hook_z) replaced by its corrupted-run value."""

    def make_hook(head: int):
        def hook(z: Float[Tensor, "batch pos head d_head"], hook: HookPoint) -> Float[Tensor, "batch pos head d_head"]:
            z[:, -1, head] = corrupt_cache[hook.name][:, -1, head]
            return z

        return hook

    hooks = [(f"blocks.{layer}.attn.hook_z", make_hook(head)) for layer, head in heads]
    return logit_diff(model.run_with_hooks(clean, fwd_hooks=hooks), answers)


def head_patching_scores(
    model: HookedTransformer, clean: Int[Tensor, "batch seq"], corrupt_cache: ActivationCache, answers: Int[Tensor, "batch 2"]
) -> Float[Tensor, "layer head"]:
    """Fraction of the clean logit diff lost when one head at a time is resample-ablated (1 = all lost)."""
    clean_diff = logit_diff(model(clean), answers)
    scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            scores[layer, head] = 1 - resample_ablated_logit_diff(model, clean, corrupt_cache, answers, [(layer, head)]) / clean_diff
    return scores


def io_attention(model: HookedTransformer, clean_cache: ActivationCache, io_position: int) -> Float[Tensor, "layer head"]:
    """Mean attention from the final position to the indirect object name, per head."""
    return torch.stack(
        [clean_cache[f"blocks.{layer}.attn.hook_pattern"][:, :, -1, io_position].mean(0) for layer in range(model.cfg.n_layers)]
    ).cpu()


def identify_name_movers(scores: Float[Tensor, "layer head"], attention: Float[Tensor, "layer head"]) -> list[tuple[int, int]]:
    """Heads that both matter under patching and attend to the indirect object (S-inhibition heads attend elsewhere)."""
    mask = (scores > MIN_PATCH_SCORE) & (attention > MIN_IO_ATTENTION)
    return [(int(layer), int(head)) for layer, head in mask.nonzero()]


def main() -> None:
    model = load_gpt2()
    pairs = get_ioi_pairs(N_PAIRS)
    clean = model.to_tokens([p["clean"] for p in pairs])
    corrupt = model.to_tokens([p["corrupt"] for p in pairs])
    answers = torch.tensor([[model.to_single_token(p["correct"]), model.to_single_token(p["incorrect"])] for p in pairs]).to(model.cfg.device)
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean)
        _, corrupt_cache = model.run_with_cache(corrupt)
        scores = head_patching_scores(model, clean, corrupt_cache, answers)
        heads = identify_name_movers(scores, io_attention(model, clean_cache, IO_POSITION))
        ablated = resample_ablated_logit_diff(model, clean, corrupt_cache, answers, heads)
    clean_diff = logit_diff(clean_logits, answers)
    print(f"name movers {heads}: logit diff {clean_diff:.3f} -> {ablated:.3f}")
    save_results(
        __file__,
        {
            "head_patching_scores": scores.tolist(),
            "identified_heads": [f"L{layer}H{head}" for layer, head in heads],
            "clean_logit_diff": clean_diff,
            "ablated_logit_diff": ablated,
        },
    )


if __name__ == "__main__":
    main()
