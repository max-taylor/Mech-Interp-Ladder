import pytest

from common.checks import load_results

MAX_L0 = 50
MAX_LOSS_INCREASE = 0.15
EXPANSION = 8
MIN_TOKENS = 1_000_000


@pytest.fixture(scope="module")
def results():
    return load_results(__file__, ["d_model", "d_sae", "n_train_tokens", "l0", "clean_loss", "spliced_loss"])


def test_dictionary_is_8x_wide(results):
    expected = EXPANSION * results["d_model"]
    assert results["d_sae"] == expected, f"d_sae is {results['d_sae']}, expected {EXPANSION} x d_model = {expected}"


def test_trained_on_enough_tokens(results):
    n = results["n_train_tokens"]
    assert n >= MIN_TOKENS, f"trained on {n:,} tokens, needs >= {MIN_TOKENS:,}"


def test_sparse_enough(results):
    l0 = results["l0"]
    assert 0 < l0 < MAX_L0, f"mean L0 is {l0:.1f}, needs to be in (0, {MAX_L0})"


def test_splicing_keeps_loss_close(results):
    clean, spliced = results["clean_loss"], results["spliced_loss"]
    increase = spliced / clean - 1
    assert increase < MAX_LOSS_INCREASE, (
        f"spliced loss {spliced:.3f} vs clean {clean:.3f}: +{increase:.1%}, needs < +{MAX_LOSS_INCREASE:.0%}"
    )
