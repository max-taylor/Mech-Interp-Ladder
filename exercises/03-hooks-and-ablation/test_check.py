import math

import pytest

from common.checks import head_keys, load_results

CLEAN_LOSS_RANGE = (2.0, 5.0)


@pytest.fixture(scope="module")
def results():
    return load_results(__file__, ["clean_loss", "zero_ablation", "mean_ablation"])


@pytest.mark.parametrize("kind", ["zero_ablation", "mean_ablation"])
def test_every_head_has_a_finite_entry(results, kind):
    missing = [k for k in head_keys() if k not in results[kind]]
    assert not missing, f"{kind} is missing {len(missing)} of 144 heads, e.g. {missing[:3]} (keys look like 'L0H0')"
    bad = [k for k in head_keys() if not math.isfinite(results[kind][k])]
    assert not bad, f"{kind} has non-finite values for {bad[:3]}"


def test_clean_loss_is_plausible(results):
    lo, hi = CLEAN_LOSS_RANGE
    loss = results["clean_loss"]
    assert lo < loss < hi, f"clean loss {loss:.3f} outside the plausible GPT-2 range {CLEAN_LOSS_RANGE}"


def test_mean_ablation_is_gentler_on_average(results):
    zero = sum(results["zero_ablation"].values()) / 144
    mean = sum(results["mean_ablation"].values()) / 144
    assert mean <= zero, f"average loss increase: mean ablation {mean:.4f} > zero ablation {zero:.4f} (by {mean - zero:.4f})"


def test_some_head_matters(results):
    biggest = max(results["zero_ablation"].values())
    assert biggest > 0.05, f"largest zero-ablation loss increase is only {biggest:.4f}; are your hooks actually editing activations?"
