import pytest

from common.checks import assert_grid_shape, load_results

FINAL_LAYER_MIN = 0.95
EARLY_LAYERS = range(0, 6)
EARLY_NAME_MIN = 0.8
UNCHANGED_POSITION_MAX = 0.1


@pytest.fixture(scope="module")
def results():
    return load_results(__file__, ["tokens", "name_positions", "recovery_grid"])


def test_grid_is_layer_by_position(results):
    assert_grid_shape(results["recovery_grid"], (12, len(results["tokens"])), "recovery_grid (layers x positions)")


def test_final_layer_final_position_recovers(results):
    value = results["recovery_grid"][-1][-1]
    assert value > FINAL_LAYER_MIN, f"patching the final layer at the final position recovers {value:.1%}, needs > {FINAL_LAYER_MIN:.0%}"


def test_name_positions_matter_early(results):
    grid, names = results["recovery_grid"], results["name_positions"]
    assert names, "name_positions is empty; it should list the positions where clean and corrupt tokens differ"
    early = [max(grid[layer][pos] for pos in names) for layer in EARLY_LAYERS]
    worst = min(early)
    assert worst > EARLY_NAME_MIN, (
        f"in layers {EARLY_LAYERS.start}-{EARLY_LAYERS.stop - 1} the best name position recovers as little as {worst:.1%}, "
        f"needs > {EARLY_NAME_MIN:.0%}"
    )


def test_unchanged_positions_do_nothing_at_layer_0(results):
    grid, names = results["recovery_grid"], set(results["name_positions"])
    leaks = [(pos, grid[0][pos]) for pos in range(len(grid[0])) if pos not in names and abs(grid[0][pos]) > UNCHANGED_POSITION_MAX]
    assert not leaks, f"at layer 0, positions with identical tokens should recover ~0, but got {leaks[:3]}"
