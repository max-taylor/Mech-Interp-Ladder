import math

import pytest

from common.checks import load_results

SECOND_HALF_MAX = 0.5
FIRST_HALF_REL_TOL = 0.05


@pytest.fixture(scope="module")
def results():
    return load_results(__file__, ["vocab_size", "first_half_loss", "second_half_loss"])


def test_second_half_is_learned(results):
    loss = results["second_half_loss"]
    assert loss < SECOND_HALF_MAX, f"second-half loss {loss:.3f} nats, needs < {SECOND_HALF_MAX} (over by {loss - SECOND_HALF_MAX:.3f})"


def test_first_half_stays_near_log_vocab(results):
    loss, log_vocab = results["first_half_loss"], math.log(results["vocab_size"])
    gap = abs(loss - log_vocab) / log_vocab
    assert gap < FIRST_HALF_REL_TOL, (
        f"first-half loss {loss:.3f} vs log(vocab) {log_vocab:.3f}: off by {gap:.1%}, needs < {FIRST_HALF_REL_TOL:.0%}. "
        "Random tokens can't be predicted; check which positions you average over."
    )
