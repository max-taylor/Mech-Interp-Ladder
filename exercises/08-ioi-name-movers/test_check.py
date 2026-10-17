import pytest

from common.checks import KNOWN_NAME_MOVERS, assert_grid_shape, load_results

# S-inhibition heads: they matter for IOI but move no names.
KNOWN_S_INHIBITION = {"L7H3", "L7H9", "L8H6", "L8H10"}
MIN_OVERLAP = 2
MAX_REMAINING = 0.5


@pytest.fixture(scope="module")
def results():
    return load_results(__file__, ["head_patching_scores", "identified_heads", "clean_logit_diff", "ablated_logit_diff"])


def test_scores_are_a_layer_by_head_grid(results):
    assert_grid_shape(results["head_patching_scores"], (12, 12), "head_patching_scores")


def test_identified_heads_are_name_movers(results):
    heads = set(results["identified_heads"])
    overlap = heads & KNOWN_NAME_MOVERS
    assert len(overlap) >= MIN_OVERLAP, f"your heads {sorted(heads)} contain {len(overlap)} known name movers; needs >= {MIN_OVERLAP}"
    wrong = heads & KNOWN_S_INHIBITION
    assert not wrong, f"{sorted(wrong)} are S-inhibition heads, not name movers. Check where each head attends."


def test_ablation_cuts_logit_diff_in_half(results):
    clean, ablated = results["clean_logit_diff"], results["ablated_logit_diff"]
    assert clean > 0, f"clean logit diff is {clean:.3f}; GPT-2 should prefer the indirect object"
    remaining = ablated / clean
    assert remaining < MAX_REMAINING, f"ablating your heads leaves {remaining:.1%} of the logit diff, needs < {MAX_REMAINING:.0%}"
