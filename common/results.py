import json
from pathlib import Path
from typing import Any


def results_path(script_file: str) -> Path:
    """results.json next to the given script (pass __file__)."""
    return Path(script_file).resolve().parent / "results.json"


def save_results(script_file: str, results: dict[str, Any]) -> Path:
    """Write results to results.json next to the given script (pass __file__)."""
    path = results_path(script_file)
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    return path
