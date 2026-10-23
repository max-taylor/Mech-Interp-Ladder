"""11 — Use a pretrained GPT-2 small SAE: find a feature, interpret it, steer with it."""

import torch
from jaxtyping import Float
from sae_lens import SAE
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from common.data import load_pile_texts
from common.device import get_device
from common.models import load_gpt2
from common.results import save_results

RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.7.hook_resid_pre"
CONCEPT_TEXT = "The bride and groom exchanged vows at the wedding. The wedding ceremony and the marriage."
NEUTRAL_TEXT = "The report was filed on Tuesday and the meeting was moved to the afternoon by the committee."
STEER_PROMPT = "I went out yesterday and"
N_TOP_EXAMPLES = 10
N_SCAN_DOCS = 300
SEQ_LEN = 128
STEER_COEFF = 30.0
N_NEW_TOKENS = 40
SEED = 0


def load_model_and_sae() -> tuple[HookedTransformer, SAE]:
    """GPT-2 small and the SAE, loaded with the model settings the SAE was trained against."""
    device = str(get_device())
    sae = SAE.from_pretrained(RELEASE, SAE_ID, device=device)
    model = load_gpt2(device, **sae.cfg.metadata.model_from_pretrained_kwargs)
    return model, sae


def feature_acts(model: HookedTransformer, sae: SAE, text: str) -> Float[Tensor, "seq d_sae"]:
    """SAE feature activations at every token of text (BOS dropped)."""
    _, cache = model.run_with_cache(text, names_filter=SAE_ID)
    return sae.encode(cache[SAE_ID][0, 1:])


def pick_feature(model: HookedTransformer, sae: SAE) -> int:
    """Feature that fires on the most concept-text tokens relative to neutral-text tokens.

    Firing frequency favours features spread across the whole concept, over ones that spike on a single token.
    """
    concept = (feature_acts(model, sae, CONCEPT_TEXT) > 0).float().mean(0)
    neutral = (feature_acts(model, sae, NEUTRAL_TEXT) > 0).float().mean(0)
    return int((concept - neutral).argmax())


def top_activating_examples(model: HookedTransformer, sae: SAE, feature: int, texts: list[str], n: int) -> list[dict[str, str | float]]:
    """The n tokens with the highest activation of feature across texts, each with ~10 tokens of left context."""
    hits = []
    for text in texts:
        tokens = model.to_tokens(text)[:, :SEQ_LEN]
        _, cache = model.run_with_cache(tokens, names_filter=SAE_ID)
        acts = sae.encode(cache[SAE_ID][0])[:, feature]
        pos = int(acts[1:].argmax()) + 1
        str_tokens = model.to_str_tokens(tokens[0])
        hits.append({"activation": acts[pos].item(), "context": "".join(str_tokens[max(1, pos - 10) : pos]), "token": str_tokens[pos]})
    return sorted(hits, key=lambda h: -h["activation"])[:n]


def generate(model: HookedTransformer, sae: SAE, feature: int, coeff: float) -> str:
    """Sampled continuation of STEER_PROMPT, adding coeff * the feature's decoder direction at the SAE hook."""

    def steer(resid: Float[Tensor, "batch pos d_model"], hook: HookPoint) -> Float[Tensor, "batch pos d_model"]:
        return resid + coeff * sae.W_dec[feature]

    torch.manual_seed(SEED)
    with model.hooks(fwd_hooks=[(SAE_ID, steer)] if coeff else []):
        return model.generate(STEER_PROMPT, max_new_tokens=N_NEW_TOKENS, temperature=0.7, verbose=False)


def mean_activation(model: HookedTransformer, sae: SAE, feature: int, text: str) -> float:
    """Mean activation of feature over the generated part of text, re-encoded with no steering."""
    n_prompt = model.to_tokens(STEER_PROMPT).shape[1] - 1
    return feature_acts(model, sae, text)[n_prompt:, feature].mean().item()


def main() -> None:
    model, sae = load_model_and_sae()
    with torch.no_grad():
        feature = pick_feature(model, sae)
        examples = top_activating_examples(model, sae, feature, load_pile_texts(N_SCAN_DOCS), N_TOP_EXAMPLES)
        unsteered = generate(model, sae, feature, 0.0)
        steered = generate(model, sae, feature, STEER_COEFF)
        results = {
            "release": RELEASE,
            "sae_id": SAE_ID,
            "feature": feature,
            "interpretation": "Weddings, marriage and family life events (births, royalty) in running text.",
            "top_examples": examples,
            "unsteered_text": unsteered,
            "steered_text": steered,
            "unsteered_activation": mean_activation(model, sae, feature, unsteered),
            "steered_activation": mean_activation(model, sae, feature, steered),
        }
    for e in examples:
        print(f"{e['activation']:6.2f}  {e['context']}[[{e['token']}]]")
    print(f"\nunsteered: {unsteered}\nsteered:   {steered}")
    save_results(__file__, results)


if __name__ == "__main__":
    main()
