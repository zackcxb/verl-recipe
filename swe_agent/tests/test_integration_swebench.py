"""Integration tests for SWE-bench_Verified workflow."""
import json
import os
import pandas as pd
import pytest
from swe_agent.prepare.load_swebench import load_swebench_verified
from swe_agent.reward.reward import compute_score


@pytest.mark.integration
def test_end_to_end_data_flow(tmp_path, mocker):
    """Test complete data flow from parquet to reward computation."""
    # 1. Create mock SWE-bench_Verified data
    test_data = pd.DataFrame({
        "instance_id": ["test__repo-123"],
        "repo": ["test/repo"],
        "base_commit": ["abc123"],
        "version": ["1.0"],
        "problem_statement": ["Fix the bug in function foo()"],
        "patch": ["diff --git a/foo.py\n-    return x\n+    return x + 1"],
        "test_patch": ["diff --git a/test_foo.py"],
        "FAIL_TO_PASS": [json.dumps(["test_foo::test_increment"])],
        "PASS_TO_PASS": [json.dumps(["test_foo::test_basic"])],
        "hints_text": ["Check the increment logic"],
        "created_at": ["2023-01-01"],
        "environment_setup_commit": ["def456"],
        "difficulty": ["easy"],
    })

    parquet_path = tmp_path / "test.parquet"
    test_data.to_parquet(parquet_path)

    # Mock Docker image exists
    mocker.patch("swe_agent.prepare.load_swebench.check_docker_image_exists", return_value=True)

    # 2. Load data through preprocessing pipeline
    df = load_swebench_verified(
        swebench_path=str(parquet_path),
        split="test",
        skip_missing_images=True,
    )

    assert len(df) == 1
    row = df.iloc[0]

    # 3. Verify data structure
    assert row["data_source"] == "swe_bench_verified"
    assert "docker_image" in row["extra_info"]
    assert row["extra_info"]["docker_image"] == "sweb.eval.x86_64.test__repo-123:latest"

    # 4. Simulate agent generating a patch
    generated_patch = "diff --git a/foo.py\n-    return x\n+    return x + 1"
    extra_info_with_patch = row["extra_info"].copy()
    extra_info_with_patch["patch"] = generated_patch

    # 5. Compute reward
    score = compute_score(
        data_source=row["data_source"],
        solution_str="",  # Not used for SWE-bench
        ground_truth=row["reward_model"],
        extra_info=extra_info_with_patch,
    )

    # Should get high score for exact match
    assert score > 0.9


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists("/data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet"),
    reason="Real SWE-bench_Verified dataset not available"
)
def test_real_dataset_loading():
    """Test loading actual SWE-bench_Verified dataset (if available)."""
    df = load_swebench_verified(
        swebench_path="/data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet",
        split="test",
        skip_missing_images=True,
    )

    # Should load some instances (at least the ones with available images)
    assert len(df) > 0

    # Verify structure
    row = df.iloc[0]
    assert "docker_image" in row["extra_info"]
    assert row["extra_info"]["docker_image"].startswith("sweb.eval.x86_64.")
    assert "prompt" in row
    assert isinstance(row["prompt"], list)
    assert "reward_model" in row
    assert row["reward_model"]["style"] == "swe_bench"
