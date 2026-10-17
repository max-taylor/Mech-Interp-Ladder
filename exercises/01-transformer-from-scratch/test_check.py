import pytest
import torch

from common.checks import call, load_solution
from common.models import load_gpt2

TOLERANCE = 1e-4
PROMPT = "Mechanistic interpretability reverse-engineers neural networks, one circuit at a time."

sol = load_solution(__file__)


@pytest.fixture(scope="module")
def reference():
    # float64 so float32 rounding noise (~2e-4 on logits of size ~100) does not mask real bugs
    return load_gpt2("cpu", processing=False).double()


@pytest.fixture(scope="module")
def model(reference):
    model = call(sol.Transformer, sol.Config())
    result = model.load_state_dict(reference.state_dict(), strict=False)
    assert not result.missing_keys, f"Your model has parameters GPT-2 doesn't provide: {result.missing_keys[:5]}"
    return model.double().eval()


def test_logits_match_hooked_transformer(model, reference):
    tokens = reference.to_tokens(PROMPT)
    with torch.no_grad():
        ours = call(model, tokens)
        theirs = reference(tokens)
    assert ours.shape == theirs.shape, f"logits shape {tuple(ours.shape)}, expected {tuple(theirs.shape)}"
    diff = (ours - theirs).abs().max().item()
    assert diff < TOLERANCE, f"max |logit difference| = {diff:.2e}, needs < {TOLERANCE:.0e}"
