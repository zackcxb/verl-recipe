import sys
import pytest
from swe_agent.reward.reward import compute_score


def test_swe_bench_reward_with_patch():
    """Test SWE-bench reward computation with patch in extra_info."""
    data_source = "swe_bench_verified"
    solution_str = "some model output text"  # Not used for SWE-bench

    ground_truth = {
        "gold_patch": "diff --git a/file.py\n-old line\n+new line",
    }

    extra_info = {
        "patch": "diff --git a/file.py\n-old line\n+new line",  # Exact match
    }

    score = compute_score(data_source, solution_str, ground_truth, extra_info)

    # Should get high score for exact match
    assert score > 0.9


def test_swe_bench_reward_no_patch():
    """Test SWE-bench reward when no patch is generated."""
    data_source = "swe_bench_verified"
    solution_str = "some model output"

    ground_truth = {"gold_patch": "diff --git a/file.py"}
    extra_info = {}  # No patch

    score = compute_score(data_source, solution_str, ground_truth, extra_info)

    assert score == 0.0


def test_swe_bench_reward_from_reward_model():
    """Test extracting gold_patch from reward_model dict."""
    data_source = "swe_bench_verified"
    solution_str = ""

    ground_truth = {
        "gold_patch": "diff --git a/file.py\n-old\n+new",
    }

    extra_info = {
        "patch": "diff --git a/file.py\n-old\n+new",
    }

    score = compute_score(data_source, solution_str, ground_truth, extra_info)

    assert score > 0.9


def test_swe_bench_reward_partial_match():
    """Test partial file overlap scoring."""
    data_source = "swe_bench_verified"
    solution_str = ""

    ground_truth = {
        "gold_patch": "diff --git a/file1.py b/file1.py\n+change1\ndiff --git a/file2.py b/file2.py\n+change2",
    }

    extra_info = {
        "patch": "diff --git a/file1.py b/file1.py\n+different_change",  # Only one file matches
    }

    score = compute_score(data_source, solution_str, ground_truth, extra_info)

    # Should get credit for partial file overlap
    # With 1 out of 2 files matching: 0.2 + 0.3 * 0.5 = 0.35
    assert 0.3 < score < 0.4


def test_non_swe_bench_fallback(mocker):
    """Test fallback to default compute_score for non-SWE-bench data."""
    # Mock the verl module and its default_compute_score function
    mock_verl_module = mocker.MagicMock()
    mock_verl_module.utils.reward_score.default_compute_score.return_value = 1.0
    mocker.patch.dict("sys.modules", {"verl": mock_verl_module, "verl.utils": mock_verl_module.utils, "verl.utils.reward_score": mock_verl_module.utils.reward_score})

    data_source = "math_qa"
    solution_str = "42"
    ground_truth = "42"

    score = compute_score(data_source, solution_str, ground_truth)

    # Should use default scoring
    assert score == 1.0
