"""SWE-bench dataset loaders for VERL.

This module provides loaders for:
- SWE-bench Lite (JSON/JSONL format)
- SWE-bench_Verified (Parquet format with Docker images)

Both loaders produce VERL-compatible DataFrames with:
- prompt: Minimal chat messages (runtime templates applied by SWE-Agent)
- reward_model: Evaluation configuration
- extra_info: Contains problem_statement, metadata, and data-affine overrides
- agent_name: "swe_agent"
"""

import json
from typing import Any

import pandas as pd

# Handle imports for both module and script execution
try:
    from ..utils.docker_utils import build_image_name, check_docker_image_exists
except ImportError:
    from swe_agent.utils.docker_utils import build_image_name, check_docker_image_exists


def _make_minimal_prompt(problem_statement: str) -> list[dict[str, str]]:
    """Create a minimal prompt that satisfies VERL's ``raw_prompt`` requirement.

    The real system/instance templates are injected by SWE-Agent at runtime
    (via swe_agent_config.yaml).  This prompt is only used for:
      - ``_agent_loop_postprocess`` (stores ``raw_prompt`` in extra_fields)
      - The reward loop (reconstructs the chat for RM scoring)

    Args:
        problem_statement: Problem description text.

    Returns:
        Minimal conversation in ``[{role, content}]`` format.
    """
    return [{"role": "user", "content": problem_statement}]


# ---------------------------------------------------------------------------
# SWE-bench Lite
# ---------------------------------------------------------------------------


def load_swebench_lite(
    swebench_path: str,
    split: str,
    agent_name: str = "swe_agent",
) -> pd.DataFrame:
    """Load SWE-bench Lite dataset from JSON/JSONL.

    Args:
        swebench_path: Path to SWE-bench Lite JSON or JSONL file.
        split: Dataset split (train/test/val).
        agent_name: Agent name (default: "swe_agent").

    Returns:
        Dataset in DataFrame format compatible with VERL.
    """
    if swebench_path.endswith(".json"):
        with open(swebench_path) as f:
            data = json.load(f)
    elif swebench_path.endswith(".jsonl"):
        data = []
        with open(swebench_path) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {swebench_path}")

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        instance_id = item.get("instance_id", f"instance_{idx}")
        problem_statement = item.get("problem_statement", "")

        # Per-instance sandbox overrides
        sandbox_overrides: dict[str, Any] = {}
        if item.get("docker_image"):
            sandbox_overrides["docker_image"] = item["docker_image"]
        if item.get("max_steps"):
            sandbox_overrides["max_steps"] = item["max_steps"]

        # Per-instance agent overrides (templates)
        agent_overrides: dict[str, Any] = {}
        if item.get("system_template"):
            agent_overrides.setdefault("templates", {})["system_template"] = item["system_template"]
        if item.get("instance_template"):
            agent_overrides.setdefault("templates", {})["instance_template"] = item["instance_template"]

        rows.append(
            {
                "prompt": _make_minimal_prompt(problem_statement),
                "data_source": "swe_bench_lite",
                "ability": "software_engineering",
                "reward_model": {
                    "style": "swe_bench",
                    "instance_id": instance_id,
                    "test_patch": item.get("test_patch", ""),
                    "gold_patch": item.get("patch", ""),
                },
                "extra_info": {
                    "index": idx,
                    "split": split,
                    "instance_id": instance_id,
                    "repo": item.get("repo", ""),
                    "base_commit": item.get("base_commit", ""),
                    "problem_statement": problem_statement,
                    "hints_text": item.get("hints_text", ""),
                    "created_at": item.get("created_at", ""),
                    "version": item.get("version", ""),
                    "FAIL_TO_PASS": item.get("FAIL_TO_PASS", ""),
                    "PASS_TO_PASS": item.get("PASS_TO_PASS", ""),
                    "sandbox_overrides": sandbox_overrides,
                    "agent_overrides": agent_overrides,
                },
                "agent_name": agent_name,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SWE-bench_Verified
# ---------------------------------------------------------------------------


def load_swebench_verified(
    swebench_path: str,
    split: str,
    agent_name: str = "swe_agent",
    skip_missing_images: bool = False,
    arch: str = "x86_64",
) -> pd.DataFrame:
    """Load SWE-bench_Verified dataset from Parquet.

    Args:
        swebench_path: Path to SWE-bench_Verified parquet file.
        split: Dataset split (train/test/val).
        agent_name: Agent name (default: "swe_agent").
        skip_missing_images: If True, filter out instances with missing Docker images.
        arch: Docker image architecture (default: x86_64).

    Returns:
        Dataset in DataFrame format compatible with VERL.
    """
    rows: list[dict[str, Any]] = []

    # Read SWE-bench_Verified parquet data
    df = pd.read_parquet(swebench_path)

    skipped_count = 0

    for idx, row in df.iterrows():
        instance_id = row.get("instance_id", f"instance_{idx}")

        # Build Docker image name
        docker_image = build_image_name(instance_id, arch=arch)

        # Check if image exists (if filtering enabled)
        if skip_missing_images:
            if not check_docker_image_exists(docker_image):
                skipped_count += 1
                continue

        problem_statement = row.get("problem_statement", "")

        # Create minimal prompt (runtime templates applied by SWE-Agent)
        prompt = _make_minimal_prompt(problem_statement)

        # Parse FAIL_TO_PASS and PASS_TO_PASS (stored as JSON strings)
        fail_to_pass = row.get("FAIL_TO_PASS", "")
        pass_to_pass = row.get("PASS_TO_PASS", "")

        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass) if fail_to_pass else []
            except json.JSONDecodeError:
                fail_to_pass = []

        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass) if pass_to_pass else []
            except json.JSONDecodeError:
                pass_to_pass = []

        # Reward model config for SWE-bench evaluation
        reward_model = {
            "style": "swe_bench",
            "instance_id": instance_id,
            "test_patch": row.get("test_patch", ""),
            "gold_patch": row.get("patch", ""),
            "FAIL_TO_PASS": fail_to_pass,
            "PASS_TO_PASS": pass_to_pass,
        }

        # Sandbox overrides for Docker image
        sandbox_overrides = {
            "docker_image": docker_image,
        }

        # Extra info
        extra_info = {
            "index": idx,
            "split": split,
            "instance_id": instance_id,
            "repo": row.get("repo", ""),
            "base_commit": row.get("base_commit", ""),
            "version": row.get("version", ""),
            "problem_statement": problem_statement,
            "hints_text": row.get("hints_text", ""),
            "created_at": row.get("created_at", ""),
            "environment_setup_commit": row.get("environment_setup_commit", ""),
            "difficulty": row.get("difficulty", ""),
            "docker_image": docker_image,
            "sandbox_overrides": sandbox_overrides,
        }

        rows.append(
            {
                "prompt": prompt,
                "data_source": "swe_bench_verified",
                "ability": "software_engineering",
                "reward_model": reward_model,
                "extra_info": extra_info,
                "agent_name": agent_name,
            }
        )

    if skip_missing_images and skipped_count > 0:
        print(f"Skipped {skipped_count} instances with missing Docker images")

    return pd.DataFrame(rows)
