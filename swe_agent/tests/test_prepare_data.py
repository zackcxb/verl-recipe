import json
import pandas as pd
import pytest
from swe_agent.prepare.prepare_data import load_swebench_verified


def test_load_swebench_verified_basic(tmp_path, mocker):
    """Test loading SWE-bench_Verified data with Docker images."""
    # Create mock parquet data
    test_data = pd.DataFrame({
        "instance_id": ["django__django-12345"],
        "repo": ["django/django"],
        "base_commit": ["abc123"],
        "version": ["3.2"],
        "problem_statement": ["Fix bug in admin"],
        "patch": ["diff --git a/file.py"],
        "test_patch": ["diff --git a/test.py"],
        "FAIL_TO_PASS": [json.dumps(["test1", "test2"])],
        "PASS_TO_PASS": [json.dumps(["test3"])],
        "hints_text": [""],
        "created_at": ["2023-01-01"],
        "environment_setup_commit": ["def456"],
        "difficulty": ["medium"],
    })

    parquet_path = tmp_path / "test.parquet"
    test_data.to_parquet(parquet_path)

    # Mock Docker image check to return True
    mocker.patch("swe_agent.prepare.prepare_data.check_docker_image_exists", return_value=True)

    # Load data
    result = load_swebench_verified(
        swebench_path=str(parquet_path),
        split="test",
        agent_name="swe_agent",
        skip_missing_images=False,
    )

    # Verify structure
    assert len(result) == 1
    assert result.iloc[0]["data_source"] == "swe_bench_verified"
    assert result.iloc[0]["agent_name"] == "swe_agent"

    # Verify prompt format
    prompt = result.iloc[0]["prompt"]
    assert isinstance(prompt, list)
    assert "Fix bug in admin" in str(prompt)

    # Verify Docker image in extra_info
    extra_info = result.iloc[0]["extra_info"]
    assert "docker_image" in extra_info
    assert extra_info["docker_image"] == "sweb.eval.x86_64.django__django-12345:latest"

    # Verify reward_model format
    reward_model = result.iloc[0]["reward_model"]
    assert reward_model["style"] == "swe_bench"
    assert reward_model["instance_id"] == "django__django-12345"


def test_skip_missing_images(tmp_path, mocker):
    """Test filtering instances with missing Docker images."""
    test_data = pd.DataFrame({
        "instance_id": ["django__django-111", "django__django-222", "django__django-333"],
        "repo": ["django/django"] * 3,
        "base_commit": ["abc"] * 3,
        "version": ["3.2"] * 3,
        "problem_statement": ["Bug 1", "Bug 2", "Bug 3"],
        "patch": ["diff1", "diff2", "diff3"],
        "test_patch": ["test1", "test2", "test3"],
        "FAIL_TO_PASS": [json.dumps([]), json.dumps([]), json.dumps([])],
        "PASS_TO_PASS": [json.dumps([]), json.dumps([]), json.dumps([])],
        "hints_text": [""] * 3,
        "created_at": ["2023-01-01"] * 3,
        "environment_setup_commit": ["def"] * 3,
        "difficulty": ["easy"] * 3,
    })

    parquet_path = tmp_path / "test.parquet"
    test_data.to_parquet(parquet_path)

    # Mock: only instance-111 and instance-333 have images
    def mock_check(image_name):
        return "111" in image_name or "333" in image_name

    mocker.patch("swe_agent.prepare.prepare_data.check_docker_image_exists", side_effect=mock_check)

    # Load with skip_missing_images=True
    result = load_swebench_verified(
        swebench_path=str(parquet_path),
        split="test",
        skip_missing_images=True,
    )

    # Should only have 2 instances (111 and 333)
    assert len(result) == 2
    instance_ids = [row["extra_info"]["instance_id"] for _, row in result.iterrows()]
    assert "django__django-111" in instance_ids
    assert "django__django-333" in instance_ids
    assert "django__django-222" not in instance_ids
