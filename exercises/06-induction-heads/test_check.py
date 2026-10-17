import pytest

from common.checks import assert_grid_shape, load_results

# Heads with the highest induction scores in GPT-2 small (ARENA 1.2; Olsson et al. 2022).
KNOWN_INDUCTION_HEADS = {"L5H1", "L5H5", "L6H9", "L7H2", "L7H10"}
MIN_OVERLAP = 3
MIN_RATIO = 2.0


@pytest.fixture(scope="module")
def results():
    return load_results(
        __file__, ["induction_scores", "top_heads", "top_heads_loss_increase", "random_heads_loss_increase"]
    )


def test_scores_are_a_layer_by_head_grid(results):
    scores = results["induction_scores"]
    assert_grid_shape(scores, (12, 12), "induction_scores")
    assert all(0.0 <= s <= 1.0 for row in scores for s in row), "induction scores are attention probabilities, so must lie in [0, 1]"


def test_top_heads_overlap_known_set(results):
    top = results["top_heads"][: len(KNOWN_INDUCTION_HEADS)]
    overlap = len(set(top) & KNOWN_INDUCTION_HEADS)
    assert overlap >= MIN_OVERLAP, f"only {overlap} of your top {len(top)} heads {top} are known induction heads; needs >= {MIN_OVERLAP}"


def test_ablating_induction_heads_hurts_more_than_random(results):
    top, rand = results["top_heads_loss_increase"], results["random_heads_loss_increase"]
    floor = max(rand, 0.01)
    assert top >= MIN_RATIO * floor, (
        f"ablating your top heads raised repeated-half loss by {top:.3f}, random heads by {rand:.3f}: "
        f"ratio {top / floor:.2f}x, needs >= {MIN_RATIO}x"
    )
