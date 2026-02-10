# SWE-bench_Verified Training Configuration

## Overview

This directory contains example configurations for training with SWE-bench_Verified dataset.

## Files

- `run_swebench_verified.sh`: Example training script with data preparation
- This README: Configuration guide and customization options

## Quick Start

### 1. Set Environment Variables

```bash
# Required: Path to your model
export MODEL_PATH=/path/to/your/model  # or HuggingFace model ID

# Required: Path to SWE-bench_Verified dataset
export SWEBENCH_VERIFIED_DATA=/data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet

# Optional: Workspace directory (default: /data1/workspace)
export WORK_BASE=/your/workspace

# Optional: GPU configuration (defaults: 8 GPUs, 1 node)
export GPUS_PER_NODE=8
export NNODES=1
```

### 2. Run the Script

```bash
cd /home/cxb/verl
chmod +x recipe/swe_agent/example/run_swebench_verified.sh
./recipe/swe_agent/example/run_swebench_verified.sh
```

## Configuration Details

### Data Preparation

The script automatically:
1. Loads SWE-bench_Verified parquet data
2. Filters instances with missing Docker images (`--skip_missing_images`)
3. Generates VERL-compatible train/test parquet files
4. Adds Docker image specifications to each instance

**Options:**
- `--skip_missing_images`: Recommended for initial testing (uses ~264 instances)
- `--docker_arch`: Specify architecture if not using x86_64
- `--output_dir`: Custom output location for prepared data

### Training Parameters

Key parameters to customize in your training command:

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `max_turns` | Maximum conversation turns | 50 |
| `execution_timeout` | Timeout per instance (seconds) | 600 (10 min) |
| `train_batch_size` | Instances per batch | 8 |
| `num_epochs` | PPO epochs per batch | 3 |
| `learning_rate` | Actor learning rate | 1e-6 |

### Docker Configuration

Docker images are specified per-instance via `sandbox_overrides.docker_image` in the dataset.
The agent loop automatically:
- Extracts the Docker image from `extra_info`
- Applies it via the data-affine override mechanism
- Configures the container with the specified image

**Requirements:**
- Docker daemon running
- Images available locally (check with `docker images | grep sweb.eval`)
- Sufficient memory (2-3GB per container)

### Reward Function

The reward function is configured in your training YAML:

```yaml
reward:
  custom_function:
    path: recipe/swe_agent/reward/reward.py
    name: compute_score
```

Current implementation uses patch similarity scoring:
- 1.0: Exact match
- 0.5: All files match
- 0.2-0.5: Partial file overlap
- 0.1: Patch generated, no overlap
- 0.0: No patch

## Customization Examples

### Faster Iteration (Lower Quality)

```bash
# In your training command, adjust:
--agent.max_turns=20  # Reduce from 50
--ppo.num_epochs=1    # Reduce from 3
```

### Longer Timeout for Complex Instances

```bash
# In your training command:
--agent.sandbox.execution_timeout=900  # 15 minutes
```

### Full Dataset (No Filtering)

```bash
# In data preparation:
python -m swe_agent.prepare.prepare_data \
    --mode swebench_verified \
    ... \
    # Remove --skip_missing_images flag
```

Note: This requires building all Docker images first.

### Custom Data Split

```bash
# Prepare with different train/test files:
python -m swe_agent.prepare.prepare_data \
    --mode swebench_verified \
    --swebench_verified_train /path/to/train.parquet \
    --swebench_verified_test /path/to/test.parquet \
    ...
```

## Monitoring

### Training Progress

```bash
# View logs
tail -f $WORK_BASE/logs/swe_agent_verified/train.log

# TensorBoard
tensorboard --logdir $WORK_BASE/logs/swe_agent_verified
```

### Docker Resource Usage

```bash
# Monitor running containers
docker stats

# List active SWE agent containers
docker ps --filter "label=verl.instance_id"

# Check disk usage
docker system df
```

### Dataset Statistics

```bash
# Check prepared data
python -c "
import pandas as pd
df = pd.read_parquet('$WORK_BASE/data/swe_agent_verified/train.parquet')
print(f'Total instances: {len(df)}')
print(f'Sample docker_image: {df.iloc[0][\"extra_info\"][\"docker_image\"]}')
print(f'Data sources: {df[\"data_source\"].unique()}')
"
```

## Troubleshooting

### "Docker image not found" during training

**Check available images:**
```bash
python recipe/swe_agent/scripts/check_docker_images.py \
    /data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet
```

**Solution:** Rerun data preparation with `--skip_missing_images`

### Container startup timeout

**Increase timeout in training config:**
```yaml
agent:
  sandbox:
    docker_startup_timeout: 300  # seconds
```

### Out of memory errors

**Reduce batch size or container memory limit:**
```bash
# In training command:
--data.train_batch_size=4  # Reduce from 8

# Or adjust container memory in runtime config:
--agent.sandbox.docker_memory_limit=4g  # Reduce from 8g
```

### Import errors when running script

**Ensure PYTHONPATH is set:**
```bash
export PYTHONPATH=/home/cxb/verl:$PYTHONPATH
```

**Or run as module:**
```bash
python -m swe_agent.prepare.prepare_data ...
```

## Advanced Configuration

### Multi-Node Training

```bash
export NNODES=4
export GPUS_PER_NODE=8
# Configure your distributed training setup accordingly
```

### Custom Templates

You can override SWE-Agent prompts per-instance via `agent_overrides` in the data:

```python
# In custom data preparation:
agent_overrides = {
    "templates": {
        "system_template": "Custom system prompt...",
        "instance_template": "Custom instance prompt..."
    }
}
```

### Experiment Tracking

Enable Weights & Biases logging:

```yaml
logging:
  wandb:
    enabled: true
    project: verl-swe-bench
    entity: your-team
    name: swebench-verified-exp1
```

## Related Documentation

- [Integration Guide](../docs/swe_bench_verified_integration.md): Detailed architecture and usage
- [SWE-bench Documentation](https://www.swebench.com/): Official SWE-bench resources
- [VERL Documentation](https://github.com/volcengine/verl): VERL framework documentation

## Support

For issues:
1. Check [Integration Guide Troubleshooting](../docs/swe_bench_verified_integration.md#troubleshooting)
2. Review training logs for error messages
3. Verify Docker images are available
4. Check dataset preparation output
