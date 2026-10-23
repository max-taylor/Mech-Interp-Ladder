"""09 — Linear probes: is this token inside quotation marks?"""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer

from common.data import make_quote_texts
from common.models import load_gpt2
from common.results import save_results

N_TEXTS = 600
TRAIN_FRACTION = 0.8
PROBE_STEPS = 300
SEED = 0


def label_tokens(model: HookedTransformer, text: str) -> tuple[Int[Tensor, "seq"], Int[Tensor, "seq"]]:
    """(tokens, labels) for one text. Label 1 = inside quotes, 0 = outside, -1 = ignore (BOS and quote tokens)."""
    tokens = model.to_tokens(text)[0]
    labels, inside = [], False
    for i, token in enumerate(model.to_str_tokens(tokens)):
        if i == 0 or '"' in token:
            labels.append(-1)
            inside ^= '"' in token
        else:
            labels.append(int(inside))
    return tokens, torch.tensor(labels)


def collect_activations(model: HookedTransformer, texts: list[str]) -> tuple[Float[Tensor, "depth n d_model"], Int[Tensor, "n"], Int[Tensor, "n"]]:
    """Residual stream at every depth (embeddings + each block) for every labelled token.

    Returns (activations, labels, text_index) where text_index says which text each token came from.
    """
    acts, labels, text_index = [], [], []
    for i, text in enumerate(texts):
        tokens, token_labels = label_tokens(model, text)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_resid_post") or n == "blocks.0.hook_resid_pre")
        stack = torch.stack([cache["blocks.0.hook_resid_pre"][0]] + [cache[f"blocks.{l}.hook_resid_post"][0] for l in range(model.cfg.n_layers)])
        keep = token_labels >= 0
        acts.append(stack[:, keep.to(stack.device)].cpu())
        labels.append(token_labels[keep])
        text_index.append(torch.full((int(keep.sum()),), i))
    return torch.cat(acts, dim=1), torch.cat(labels), torch.cat(text_index)


def train_probe(x: Float[Tensor, "n d"], y: Int[Tensor, "n"], steps: int) -> torch.nn.Linear:
    """Logistic regression probe on standardised features (mean/std baked into the returned layer)."""
    mean, std = x.mean(0), x.std(0) + 1e-6
    probe = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    xs = (x - mean) / std
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(xs).squeeze(-1), y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probe.weight /= std
        probe.bias -= (probe.weight * mean).sum()
    return probe


def accuracy(probe: torch.nn.Linear, x: Float[Tensor, "n d"], y: Int[Tensor, "n"]) -> float:
    """Fraction of tokens classified correctly."""
    with torch.no_grad():
        return ((probe(x).squeeze(-1) > 0).long() == y).float().mean().item()


def main() -> None:
    torch.manual_seed(SEED)
    model = load_gpt2()
    with torch.no_grad():
        acts, labels, text_index = collect_activations(model, make_quote_texts(N_TEXTS, SEED))
    train = text_index < int(TRAIN_FRACTION * N_TEXTS)
    shuffled = labels[train][torch.randperm(int(train.sum()))]

    real, control = [], []
    for depth in range(acts.shape[0]):
        x = acts[depth]
        real.append(accuracy(train_probe(x[train], labels[train], PROBE_STEPS), x[~train], labels[~train]))
        control.append(accuracy(train_probe(x[train], shuffled, PROBE_STEPS), x[~train], labels[~train]))
        print(f"depth {depth:2d}: probe {real[-1]:.3f}  control {control[-1]:.3f}")

    test_labels = labels[~train].float()
    save_results(
        __file__,
        {
            "accuracy_by_layer": real,
            "control_accuracy_by_layer": control,
            "majority_baseline": max(test_labels.mean().item(), 1 - test_labels.mean().item()),
        },
    )


if __name__ == "__main__":
    main()
