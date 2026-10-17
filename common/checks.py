"""Test helpers: turn missing work into clear pytest failures instead of crashes."""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from common.results import results_path

# Name movers from Wang et al. 2022, "Interpretability in the Wild" (IOI paper).
KNOWN_NAME_MOVERS = {"L9H9", "L9H6", "L10H0"}


def load_solution(test_file: str) -> ModuleType:
    """Import solution.py from the test's own directory under a unique module name."""
    path = Path(test_file).resolve().parent / "solution.py"
    spec = importlib.util.spec_from_file_location(f"solution_{path.parent.name.replace('-', '_')}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a learner function; fail clearly if it is still a stub."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.fail(f"{fn.__qualname__} is not implemented yet (raised NotImplementedError).")


def load_results(test_file: str, required_keys: list[str]) -> dict[str, Any]:
    """Load results.json next to the test; fail clearly if missing or incomplete."""
    path = results_path(test_file)
    if not path.exists():
        pytest.fail(f"{path.name} not found in {path.parent.name}/. Run `uv run python exercises/{path.parent.name}/solution.py` first.")
    try:
        results = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        pytest.fail(f"results.json is not valid JSON: {err}")
    missing = [k for k in required_keys if k not in results]
    if missing:
        pytest.fail(f"results.json is missing keys: {missing}")
    return results


def assert_grid_shape(grid: list[list[float]], shape: tuple[int, int], name: str) -> None:
    """Fail clearly unless grid is a nested list of the given (rows, cols) shape."""
    actual = (len(grid), len(grid[0]) if grid else 0)
    assert actual == shape, f"{name} has shape {actual}, expected {shape}"


def head_keys(n_layers: int = 12, n_heads: int = 12) -> list[str]:
    """Canonical head names "L{layer}H{head}" used across exercises."""
    return [f"L{layer}H{head}" for layer in range(n_layers) for head in range(n_heads)]
