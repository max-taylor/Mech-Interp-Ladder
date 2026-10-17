import pytest
import torch

from common.checks import load_results
from common.data import FACTUAL_PROMPTS
from common.models import load_gpt2

N_DEPTHS = 13


@pytest.fixture(scope="module")
def entries():
    entries = load_results(__file__, ["prompts"])["prompts"]
    reported = {e["prompt"] for e in entries}
    missing = [p for p, _ in FACTUAL_PROMPTS if p not in reported]
    assert not missing, f"results.json is missing {len(missing)} of {len(FACTUAL_PROMPTS)} prompts, e.g. {missing[0]!r}"
    return entries


@pytest.fixture(scope="module")
def model():
    return load_gpt2("cpu")


def test_one_top_token_per_depth(entries):
    for e in entries:
        n = len(e["top_token_by_layer"])
        assert n == N_DEPTHS, f"{e['prompt']!r}: {n} decoded depths, expected {N_DEPTHS} (embeddings + 12 blocks)"


def test_final_depth_matches_model_prediction(entries, model):
    for e in entries:
        with torch.no_grad():
            predicted = model.tokenizer.decode(model(e["prompt"])[0, -1].argmax().item())
        final = e["top_token_by_layer"][-1]
        assert final == predicted, f"{e['prompt']!r}: your final depth decodes to {final!r}, the model predicts {predicted!r}"


def test_first_correct_layer_is_consistent(entries):
    for e in entries:
        tokens, layer = e["top_token_by_layer"], e["first_correct_layer"]
        expected = next((i for i, t in enumerate(tokens) if t == e["answer"]), None)
        assert layer == expected, f"{e['prompt']!r}: first_correct_layer is {layer}, but your own top tokens say {expected}"
