import pytest

from common.checks import load_results

BEST_MIN = 0.85
CONTROL_SLACK = 0.05
VALID_DEPTHS = (12, 13)


@pytest.fixture(scope="module")
def results():
    return load_results(__file__, ["accuracy_by_layer", "control_accuracy_by_layer", "majority_baseline"])


def test_one_accuracy_per_layer(results):
    for key in ("accuracy_by_layer", "control_accuracy_by_layer"):
        n = len(results[key])
        assert n in VALID_DEPTHS, f"{key} has {n} entries, expected 12 (blocks) or 13 (embeddings + blocks)"


def test_best_layer_beats_threshold(results):
    accs = results["accuracy_by_layer"]
    best = max(accs)
    assert best > BEST_MIN, f"best held-out accuracy is {best:.1%} (layer {accs.index(best)}), needs > {BEST_MIN:.0%}"


def test_control_probe_stays_near_chance(results):
    ceiling = results["majority_baseline"] + CONTROL_SLACK
    worst = max(results["control_accuracy_by_layer"])
    assert worst <= ceiling, (
        f"shuffled-label control reaches {worst:.1%}, above majority baseline {results['majority_baseline']:.1%} "
        f"+ {CONTROL_SLACK:.0%}. Is test data leaking into training, or are you evaluating on shuffled labels?"
    )
