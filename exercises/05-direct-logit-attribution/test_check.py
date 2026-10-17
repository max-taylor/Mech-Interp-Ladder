import pytest
import torch

from common.checks import KNOWN_NAME_MOVERS, call, head_keys, load_solution
from common.models import load_gpt2

TOLERANCE = 1e-3
PROMPT = "When Mary and John went to the store, John gave a drink to"
CORRECT, INCORRECT = " Mary", " John"

sol = load_solution(__file__)


@pytest.fixture(scope="module")
def model():
    return load_gpt2("cpu")


@pytest.fixture(scope="module")
def attribution(model):
    return call(sol.logit_diff_attribution, model, PROMPT, CORRECT, INCORRECT)


def test_has_every_component(attribution):
    expected = ["embed", "pos_embed", *head_keys(), *[f"L{layer}_mlp" for layer in range(12)]]
    missing = [k for k in expected if k not in attribution]
    assert not missing, f"missing {len(missing)} components, e.g. {missing[:4]}"


def test_contributions_sum_to_logit_diff(model, attribution):
    with torch.no_grad():
        logits = model(PROMPT)[0, -1]
    true_diff = (logits[model.to_single_token(CORRECT)] - logits[model.to_single_token(INCORRECT)]).item()
    total = sum(attribution.values())
    assert abs(total - true_diff) < TOLERANCE, (
        f"contributions sum to {total:.4f}, true logit diff is {true_diff:.4f} (off by {abs(total - true_diff):.4f}, "
        f"needs < {TOLERANCE}). Did you divide by the final LayerNorm scale and include every bias?"
    )


def test_top_head_is_a_name_mover(attribution):
    top = max(head_keys(), key=lambda k: attribution[k])
    assert top in KNOWN_NAME_MOVERS, f"largest positive head is {top} ({attribution[top]:+.3f}); expected one of {sorted(KNOWN_NAME_MOVERS)}"
