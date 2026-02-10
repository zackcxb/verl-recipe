# SWE Agent Recipe for VERL

A reinforcement learning recipe for training software engineering agents using the VERL framework.

## Overview

This recipe enables training LLMs to solve software engineering tasks through:
- Code analysis and modification
- Bug fixing
- Feature implementation
- Test-driven development

The agent operates in a sandboxed environment (Docker or local) with access to:
- File operations (read, write, edit)
- Shell commands
- Git operations
- Test execution

## Features

- **Multiple Dataset Support**: Simple test cases, SWE-bench Lite, and SWE-bench_Verified
- **Docker Integration**: Isolated execution environments per task
- **Flexible Reward System**: Patch-based evaluation with extensible scoring
- **Data-Affine Overrides**: Per-instance configuration for Docker images, timeouts, templates
- **Comprehensive Testing**: Unit, integration, and end-to-end tests

## Quick Start

### 1. Generate Simple Test Data

```bash
python -m swe_agent.prepare.prepare_data \
  --mode simple \
  --train_size 100 \
  --test_size 10 \
  --output_dir data/swe_agent
```

### 2. Train

```bash
# Configure your training script with:
# - Data path: data/swe_agent/train.parquet
# - Reward function: recipe/swe_agent/reward/reward.py:compute_score

python -m verl.trainer.main_ppo \
  --config your_config.yaml
```

## SWE-bench_Verified Integration

### Overview

The SWE Agent recipe now supports training with [SWE-bench_Verified](https://www.swebench.com/), a curated dataset of 500 real-world software engineering tasks from 12 popular Python repositories.

### Quick Start

#### 1. Check Docker Images

```bash
python -m swe_agent.scripts.check_docker_images \
  /data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet
```

#### 2. Prepare Dataset

```bash
# Using helper script
./recipe/swe_agent/scripts/prepare_swebench_verified.sh

# Or manually
python -m swe_agent.prepare.prepare_data \
  --mode swebench_verified \
  --swebench_verified_train /data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet \
  --swebench_verified_test /data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet \
  --skip_missing_images \
  --output_dir data/swe_agent_verified
```

#### 3. Train

```bash
# See example configuration in:
# recipe/swe_agent/example/run_swebench_verified.sh

# Adapt to your setup:
export MODEL_PATH=/path/to/your/model
export SWEBENCH_VERIFIED_DATA=/data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet
./recipe/swe_agent/example/run_swebench_verified.sh
```

### Features

- **Docker Integration**: Each instance runs in its own Docker container with pre-configured environment
- **Image Filtering**: Automatically skip instances with missing Docker images
- **Standard Evaluation**: Compatible with SWE-bench evaluation harness
- **Flexible Reward**: Currently uses patch similarity; extensible to test execution

### Dataset Statistics

- Total instances: 500
- Repositories: 12 (django, sympy, matplotlib, scikit-learn, etc.)
- Pre-built images: ~264 (varies by setup)
- Docker image format: `sweb.eval.x86_64.{instance_id}:latest`

### Documentation

- [Integration Guide](docs/swe_bench_verified_integration.md) - Detailed setup and usage
- [Training Configuration](example/README_swebench_verified.md) - Configuration options
- [Architecture Details](docs/swe_bench_verified_integration.md#architecture-details)

## Module Structure

```
recipe/swe_agent/
├── prepare/
│   ├── prepare_data.py          # CLI for data preparation
│   └── load_swebench.py          # SWE-bench loaders (Lite + Verified)
├── reward/
│   └── reward.py                 # Reward function with patch comparison
├── utils/
│   ├── docker_utils.py           # Docker image utilities
│   ├── patch_extractor.py        # Patch extraction from agent output
│   └── repo_manager.py           # Repository management
├── config/
│   ├── yaml_builder.py           # SWE-Agent YAML configuration
│   ├── runtime_config.py         # Runtime config with data overrides
│   └── swe_agent_config.yaml    # Default agent configuration
├── scripts/
│   ├── check_docker_images.py   # Docker image availability checker
│   └── prepare_swebench_verified.sh  # Data preparation helper
├── example/
│   ├── run_swebench_verified.sh # Example training script
│   └── README_swebench_verified.md  # Configuration guide
├── docs/
│   └── swe_bench_verified_integration.md  # Integration guide
├── tests/
│   ├── test_docker_utils.py     # Docker utilities tests
│   ├── test_prepare_data.py     # Data preprocessing tests
│   ├── test_reward_swebench.py  # Reward function tests
│   ├── test_yaml_builder_docker.py  # YAML builder tests
│   └── test_integration_swebench.py  # Integration tests
└── swe_agent_loop.py             # Main agent loop

```

## Data Format

All datasets are converted to VERL-compatible format:

```python
{
    "prompt": [{"role": "user", "content": "Problem description"}],
    "data_source": "swe_bench_verified",  # or "swe_agent_simple", "swe_bench_lite"
    "ability": "software_engineering",
    "reward_model": {
        "style": "swe_bench",
        "instance_id": "django__django-12345",
        "gold_patch": "diff --git ...",
        "test_patch": "diff --git ...",
        "FAIL_TO_PASS": ["test1", "test2"],
        "PASS_TO_PASS": ["test3"]
    },
    "extra_info": {
        "instance_id": "django__django-12345",
        "problem_statement": "Fix bug in admin...",
        "docker_image": "sweb.eval.x86_64.django__django-12345:latest",
        "sandbox_overrides": {
            "docker_image": "sweb.eval.x86_64.django__django-12345:latest"
        },
        ...
    },
    "agent_name": "swe_agent"
}
```

## Testing

```bash
# Unit tests
python -m pytest swe_agent/tests/ -v --ignore=swe_agent/tests/test_integration_swebench.py

# Integration tests (requires real data)
python -m pytest swe_agent/tests/test_integration_swebench.py -v -m integration
```

## Troubleshooting

See the [Integration Guide Troubleshooting](docs/swe_bench_verified_integration.md#troubleshooting) section for common issues and solutions.

### Common Issues

- **Docker image not found**: Use `--skip_missing_images` or build missing images
- **Container fails to start**: Check Docker daemon, image existence, disk space
- **Low reward scores**: Check agent logs, patch format, timeout settings
- **Import errors**: Use `python -m` to run as modules

## References

- [SWE-bench Paper](https://arxiv.org/abs/2310.06770)
- [SWE-bench GitHub](https://github.com/princeton-nlp/SWE-bench)
- [VERL Documentation](https://github.com/volcengine/verl)
- [SWE-Agent](https://github.com/princeton-nlp/SWE-agent)

## License

Copyright 2026 Bytedance Ltd. and/or its affiliates. Licensed under the Apache License, Version 2.0.
