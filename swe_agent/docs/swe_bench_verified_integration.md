# SWE-bench_Verified Integration Guide

## Overview

This guide describes how to use SWE-bench_Verified dataset with Verl's SWE Agent recipe.

## Architecture

The integration follows a three-stage pipeline:

1. **Data Preprocessing**: Convert SWE-bench_Verified parquet files to VERL format
2. **RL Training**: Run agent in Docker containers, collect trajectories
3. **Reward Computation**: Calculate rewards based on patch similarity

## Prerequisites

- Docker with pre-built SWE-bench images (format: `sweb.eval.x86_64.{instance_id}:latest`)
- SWE-bench_Verified dataset in parquet format
- Python packages: pandas, pyarrow, docker

## Quick Start

### Step 1: Prepare Data

```bash
cd /home/cxb/verl

python -m swe_agent.prepare.prepare_data \
  --mode swebench_verified \
  --swebench_verified_train /data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet \
  --swebench_verified_test /data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet \
  --skip_missing_images \
  --output_dir data/swe_agent_verified
```

**Options:**
- `--skip_missing_images`: Filter out instances without Docker images (recommended for initial testing)
- `--docker_arch`: Specify architecture (default: x86_64)

### Step 2: Configure Training

Update your training config YAML to point to the new dataset:

```yaml
data:
  train_files: data/swe_agent_verified/train.parquet
  val_files: data/swe_agent_verified/test.parquet

custom_reward_function:
  path: recipe/swe_agent/reward/reward.py
  name: compute_score
```

### Step 3: Run Training

```bash
python -m verl.trainer.main_ppo \
  --config config/swe_agent_swebench.yaml
```

## Data Format

### Input (SWE-bench_Verified Parquet)

Required fields:
- `instance_id`: Unique identifier (e.g., "django__django-12345")
- `repo`: Repository name (e.g., "django/django")
- `base_commit`: Git commit hash
- `version`: Repository version
- `problem_statement`: Task description
- `patch`: Gold patch (solution)
- `test_patch`: Test case modifications
- `FAIL_TO_PASS`: Tests that should pass after fix (JSON array)
- `PASS_TO_PASS`: Tests that should remain passing (JSON array)

### Output (VERL Format)

Generated parquet with fields:
- `prompt`: Conversation format with system + user message
- `data_source`: "swe_bench_verified"
- `ability`: "software_engineering"
- `reward_model`: Dict with style, instance_id, gold_patch, test info
- `extra_info`: Dict with metadata + **docker_image** + **sandbox_overrides**
- `agent_name`: "swe_agent"

## Docker Image Management

### Image Naming Convention

Format: `sweb.eval.{arch}.{instance_id}:{tag}`
Example: `sweb.eval.x86_64.django__django-12345:latest`

### Checking Available Images

```python
from swe_agent.utils.docker_utils import check_docker_image_exists

image_name = "sweb.eval.x86_64.django__django-12345:latest"
exists = check_docker_image_exists(image_name)
```

### Building Missing Images

For instances without pre-built images, use SWE-bench's docker building tools:

```bash
cd /home/cxb/SWE-bench
python -m swebench.harness.docker_build \
  --instances_path /data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet \
  --instance_ids django__django-12345
```

## Reward Function

Current implementation uses **simple patch comparison**:
- Compares generated patch vs. gold patch
- Score based on text similarity and file overlap
- Fast but approximate

### Scoring System

- **1.0**: Exact match (after normalization)
- **0.5**: All changed files match
- **0.2-0.5**: Partial file overlap
- **0.1**: Patch generated but no file overlap
- **0.0**: No patch generated

### Future: Test Execution Rewards

To integrate real test execution (SWE-bench standard evaluation):

1. Modify `recipe/swe_agent/reward/reward.py`
2. Import SWE-bench harness: `from swebench.harness.grading import get_eval_tests_report`
3. Apply patch in Docker container and run tests
4. Calculate score from FAIL_TO_PASS results

Reference: `/home/cxb/SWE-bench/swebench/harness/grading.py`

## Architecture Details

### Data-Affine Override Mechanism

SWE-bench_Verified instances require specific Docker images. The system supports per-instance configuration through `sandbox_overrides`:

```python
extra_info = {
    "docker_image": "sweb.eval.x86_64.django__django-12345:latest",
    "sandbox_overrides": {
        "docker_image": "sweb.eval.x86_64.django__django-12345:latest",
    }
}
```

The agent loop automatically:
1. Extracts `sandbox_overrides` from `extra_info`
2. Applies them to the runtime config via `apply_data_overrides()`
3. Passes the Docker image to the YAML builder
4. Configures the container with the specified image

### Module Structure

```
recipe/swe_agent/
├── prepare/
│   ├── prepare_data.py          # CLI for data preparation
│   └── load_swebench.py          # SWE-bench loaders (Lite + Verified)
├── reward/
│   └── reward.py                 # Reward function with patch comparison
├── utils/
│   └── docker_utils.py           # Docker image utilities
├── config/
│   ├── yaml_builder.py           # SWE-Agent YAML configuration
│   └── runtime_config.py         # Runtime config with data overrides
└── swe_agent_loop.py             # Main agent loop
```

## Troubleshooting

### Issue: "Docker image not found"

**Solution:** Use `--skip_missing_images` flag or build missing images.

```bash
# Check which images are available
python swe_agent/scripts/check_docker_images.py \
  /data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet
```

### Issue: "Container fails to start"

**Check:**
1. Docker daemon running: `docker ps`
2. Image exists: `docker images | grep sweb.eval`
3. Sufficient disk space: `df -h`
4. Check logs: `docker logs <container_id>`

### Issue: Low reward scores

**Possible causes:**
1. Agent not generating valid patches (check logs)
2. Patch format mismatch (check diff format)
3. Timeout before completion (increase max_steps)
4. Wrong files being modified

**Debug:**
```bash
# Check agent output
cat /tmp/swe_agent_verified_output/<instance_id>/trajectory.jsonl

# Check generated patch
cat /tmp/swe_agent_verified_output/<instance_id>/patch.diff
```

### Issue: "Module not found" errors

**Solution:** Ensure you're running as a module:
```bash
# Correct
python -m swe_agent.prepare.prepare_data --mode swebench_verified ...

# Incorrect (may fail with imports)
python swe_agent/prepare/prepare_data.py --mode swebench_verified ...
```

## Performance Considerations

- **Dataset size**: 500 instances in SWE-bench_Verified
- **Available images**: ~264 pre-built (varies by setup)
- **Docker overhead**: ~2-5 seconds per container start
- **Memory**: ~2-3GB per Docker container
- **Disk**: ~50-100GB for all images

**Recommendation**: Start with `skip_missing_images=True` to train on ~264 available instances.

### Parallelization

The agent loop processes instances sequentially. For faster training:
- Increase batch size (limited by GPU memory)
- Use multiple training processes
- Consider distributed training setup

## Testing

### Unit Tests

```bash
# Test Docker utilities
python -m pytest swe_agent/tests/test_docker_utils.py -v

# Test data preprocessing
python -m pytest swe_agent/tests/test_prepare_data.py -v

# Test reward function
python -m pytest swe_agent/tests/test_reward_swebench.py -v

# Test YAML builder
python -m pytest swe_agent/tests/test_yaml_builder_docker.py -v
```

### Integration Tests

```bash
# Run integration tests (requires real data)
python -m pytest swe_agent/tests/test_integration_swebench.py -v -m integration
```

### End-to-End Test

```bash
# Prepare small dataset
python -m swe_agent.prepare.prepare_data \
  --mode swebench_verified \
  --swebench_verified_train /data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet \
  --swebench_verified_test /data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet \
  --skip_missing_images \
  --output_dir /tmp/test_swe_verified

# Verify output
python -c "
import pandas as pd
df = pd.read_parquet('/tmp/test_swe_verified/train.parquet')
print(f'Loaded {len(df)} instances')
print(f'Sample docker_image: {df.iloc[0][\"extra_info\"][\"docker_image\"]}')
"
```

## References

- [SWE-bench Paper](https://arxiv.org/abs/2310.06770)
- [SWE-bench GitHub](https://github.com/princeton-nlp/SWE-bench)
- [Verl Documentation](https://github.com/volcengine/verl)
- [Implementation Plan](../../../docs/plans/2026-02-10-swe-bench-verified-integration.md)
