"""Reward function for DeepEyes gateway recipe.

Re-exports legacy compute_score. Loaded by trainer via load_extern_object.
"""

from __future__ import annotations

from pathlib import Path


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """Delegate to legacy DeepEyes compute_score, loaded lazily to avoid import-time side effects."""
    global _legacy_fn
    if _legacy_fn is None:
        from verl.utils.import_utils import load_extern_object

        legacy_path = str(Path(__file__).resolve().parent.parent / "deepeyes" / "deepeyes.py")
        _legacy_fn = load_extern_object(module_path=legacy_path, object_name="compute_score")
    return _legacy_fn(data_source, solution_str, ground_truth, extra_info)


_legacy_fn = None