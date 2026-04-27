"""Reward function for DeepEyes gateway recipe.

Re-exports legacy compute_score. Loaded by trainer via load_extern_object.
"""

from __future__ import annotations

from pathlib import Path
import re


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """Delegate to legacy DeepEyes compute_score, loaded lazily to avoid import-time side effects."""
    if data_source == "deepeyes_gateway_smoke":
        return _compute_smoke_score(solution_str, ground_truth)

    global _legacy_fn
    if _legacy_fn is None:
        from verl.utils.import_utils import load_extern_object

        legacy_path = str(Path(__file__).resolve().parent.parent / "deepeyes" / "deepeyes.py")
        _legacy_fn = load_extern_object(module_path=legacy_path, object_name="compute_score")
    return _legacy_fn(data_source, solution_str, ground_truth, extra_info)


def _compute_smoke_score(solution_str: str, ground_truth: str) -> float:
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL | re.IGNORECASE)
    answer = answer_match.group(1) if answer_match else solution_str
    return 1.0 if _normalize(answer) == _normalize(ground_truth) else 0.0


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


_legacy_fn = None
