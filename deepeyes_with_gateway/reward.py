"""Delegate reward scoring to the legacy DeepEyes recipe without duplicating logic.

The legacy module is loaded lazily by file path so this wrapper can be executed
directly from a file loader without depending on the superproject `recipe`
package layout or triggering the legacy module's import-time side effects until
`compute_score(...)` is actually called.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable

_LEGACY_COMPUTE_SCORE: Callable[[str, str, str, object], float] | None = None


def _load_legacy_compute_score() -> Callable[[str, str, str, object], float]:
    global _LEGACY_COMPUTE_SCORE
    if _LEGACY_COMPUTE_SCORE is not None:
        return _LEGACY_COMPUTE_SCORE

    legacy_path = Path(__file__).resolve().parent.parent / "deepeyes" / "deepeyes.py"
    spec = spec_from_file_location("deepeyes_legacy_reward", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create a module spec for legacy reward module: {legacy_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    compute_score = getattr(module, "compute_score", None)
    if not callable(compute_score):
        raise AttributeError(f"Legacy reward module does not define callable compute_score: {legacy_path}")
    _LEGACY_COMPUTE_SCORE = compute_score
    return _LEGACY_COMPUTE_SCORE


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    return _load_legacy_compute_score()(data_source, solution_str, ground_truth, extra_info)
