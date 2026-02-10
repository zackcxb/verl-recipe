# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SWE Agent Dataset Generator.

Supports two data sources:
1. Simple test cases (for quick validation)
2. SWE-bench Lite (for full evaluation)

Data format is VERL-compatible:
- prompt: Minimal chat messages (satisfies framework's ``raw_prompt`` requirement).
          The *real* system/instance templates are applied at runtime by
          SWE-Agent via ``swe_agent_config.yaml`` — this avoids duplicating
          prompt templates between data preparation and runtime.
- reward_model: Evaluation configuration
- extra_info: Contains problem_statement, repo_content, expected_patch,
              and data-affine overrides (sandbox_overrides / agent_overrides).
- agent_name: "swe_agent"

Data-affine override mechanism:
  extra_info may contain two special dicts that override swe_agent_config.yaml
  defaults at runtime (per-instance granularity):

  - sandbox_overrides: e.g. {"docker_image": "...", "max_steps": 50}
  - agent_overrides:   e.g. {"templates": {"system_template": "..."}}

  See swe_agent_config.yaml for the full list (marked [DATA-AFFINE]).
"""

import argparse
import json
import os
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Prompt helper
# ---------------------------------------------------------------------------


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
# Simple test data
# ---------------------------------------------------------------------------


def generate_simple_test_data(
    num_samples: int,
    split: str,
    agent_name: str = "swe_agent",
) -> pd.DataFrame:
    """Generate simple test data for quick validation."""

    test_cases = [
        {
            "problem_statement": "rename 1.txt to 2.txt",
            "repo_content": {"1.txt": "Hello World"},
            "expected_patch": ("diff --git a/1.txt b/2.txt\nsimilarity index 100%\nrename from 1.txt\nrename to 2.txt"),
        },
        {
            "problem_statement": "Create a new file called hello.py that prints 'Hello, World!'",
            "repo_content": {},
            "expected_patch": (
                "diff --git a/hello.py b/hello.py\n"
                "new file mode 100644\n"
                "--- /dev/null\n"
                "+++ b/hello.py\n"
                "@@ -0,0 +1 @@\n"
                "+print('Hello, World!')"
            ),
        },
        {
            "problem_statement": ("Fix the bug in calculator.py: the add function should return a + b, not a - b"),
            "repo_content": {"calculator.py": "def add(a, b):\n    return a - b"},
            "expected_patch": (
                "diff --git a/calculator.py b/calculator.py\n"
                "--- a/calculator.py\n"
                "+++ b/calculator.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return a - b\n"
                "+    return a + b"
            ),
        },
    ]

    rows: list[dict[str, Any]] = []
    for idx in range(num_samples):
        case = test_cases[idx % len(test_cases)]
        rows.append(
            {
                "prompt": _make_minimal_prompt(case["problem_statement"]),
                "data_source": "swe_agent_simple",
                "ability": "software_engineering",
                "reward_model": {
                    "style": "swe_agent",
                    "ground_truth": case["expected_patch"],
                },
                "extra_info": {
                    "index": idx,
                    "split": split,
                    "repo_content": case["repo_content"],
                    "expected_patch": case["expected_patch"],
                    "problem_statement": case["problem_statement"],
                    # Data-affine overrides — simple tasks use smaller limits.
                    "sandbox_overrides": {"max_steps": 10, "max_turns": 8},
                },
                "agent_name": agent_name,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SWE-bench Lite
# ---------------------------------------------------------------------------


def load_swebench_lite(
    swebench_path: str,
    split: str,
    agent_name: str = "swe_agent",
) -> pd.DataFrame:
    """Load SWE-bench Lite dataset."""

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
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE Agent Dataset Generator")
    parser.add_argument(
        "--mode",
        choices=["simple", "swebench"],
        default="simple",
        help="Data generation mode",
    )
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--swebench_train", type=str, default=None)
    parser.add_argument("--swebench_test", type=str, default=None)
    parser.add_argument("--output_dir", default="data/swe_agent")
    parser.add_argument("--agent_name", default="swe_agent")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check pyarrow
    try:
        import importlib.util

        if importlib.util.find_spec("pyarrow") is None:
            raise ImportError("pyarrow not found")
    except ImportError as err:
        raise ImportError("pyarrow is required for parquet support. Install with: pip install pyarrow") from err

    if args.mode == "simple":
        print("Generating simple test data...")
        train_df = generate_simple_test_data(args.train_size, "train", args.agent_name)
        test_df = generate_simple_test_data(args.test_size, "test", args.agent_name)
    else:
        print("Loading SWE-bench Lite data...")
        if args.swebench_train is None or args.swebench_test is None:
            raise ValueError("--swebench_train and --swebench_test are required for swebench mode")
        train_df = load_swebench_lite(args.swebench_train, "train", args.agent_name)
        test_df = load_swebench_lite(args.swebench_test, "test", args.agent_name)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print("\nDataset generation completed!")
    print(f"Train: {len(train_df)} samples -> {train_path}")
    print(f"Test:  {len(test_df)} samples -> {test_path}")


if __name__ == "__main__":
    main()
