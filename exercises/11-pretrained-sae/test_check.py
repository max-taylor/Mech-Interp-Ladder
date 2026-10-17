import pytest

from common.checks import load_results

MIN_EXAMPLES = 5


@pytest.fixture(scope="module")
def results():
    return load_results(
        __file__,
        ["feature", "interpretation", "top_examples", "unsteered_text", "steered_text", "unsteered_activation", "steered_activation"],
    )


def test_has_interpretation_and_examples(results):
    assert results["interpretation"].strip(), "interpretation is empty; write one line saying what the feature means"
    n = len(results["top_examples"])
    assert n >= MIN_EXAMPLES, f"only {n} top-activating examples, needs >= {MIN_EXAMPLES}"


def test_steering_changes_output(results):
    assert results["steered_text"] != results["unsteered_text"], "steered and unsteered text are identical; is your hook running?"


def test_steering_raises_feature_activation(results):
    before, after = results["unsteered_activation"], results["steered_activation"]
    assert after > before, f"feature activation on steered text {after:.3f} is not above unsteered {before:.3f}"
