#!/usr/bin/env bash
# =============================================================================
# SWE-bench_Verified Training Example
#
# This script demonstrates how to train with SWE-bench_Verified dataset.
# Adapt the configuration variables below for your environment.
# =============================================================================

set -euo pipefail

# ================= Configuration =================

# [REQUIRED] Path to your model
# Examples:
#   - Local: export MODEL_PATH=/data/models/Qwen/Qwen3-4B-Instruct
#   - HuggingFace: export MODEL_PATH=Qwen/Qwen3-4B-Instruct
MODEL_PATH=${MODEL_PATH:-/data1/models/Qwen/Qwen3-4B-Instruct-2507}

# [REQUIRED] SWE-bench_Verified dataset paths
SWEBENCH_VERIFIED_DATA=${SWEBENCH_VERIFIED_DATA:-/data1/dataset/SWE-bench_Verified/data/test-00000-of-00001.parquet}

# Output directories
WORK_BASE=${WORK_BASE:-/data1/workspace}
DATA_DIR=$WORK_BASE/data/swe_agent_verified
CHECKPOINT_DIR=$WORK_BASE/checkpoints/swe_agent_verified
LOG_DIR=$WORK_BASE/logs/swe_agent_verified

# Training configuration
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}

# ================= Setup =================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_ROOT="$(cd "$RECIPE_DIR/../.." && pwd)"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

mkdir -p $DATA_DIR $CHECKPOINT_DIR $LOG_DIR

echo "=========================================="
echo "SWE-bench_Verified Training Setup"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $SWEBENCH_VERIFIED_DATA"
echo "Nodes: $NNODES x $GPUS_PER_NODE GPUs"
echo "=========================================="

# ================= Step 1: Prepare Data =================

echo ""
echo "Step 1: Preparing SWE-bench_Verified dataset..."
echo "This will filter instances with available Docker images."

python -m swe_agent.prepare.prepare_data \
    --mode swebench_verified \
    --swebench_verified_train $SWEBENCH_VERIFIED_DATA \
    --swebench_verified_test $SWEBENCH_VERIFIED_DATA \
    --skip_missing_images \
    --output_dir $DATA_DIR

echo "✓ Data prepared at: $DATA_DIR"
echo "  Training instances: $(python -c "import pandas as pd; print(len(pd.read_parquet('$DATA_DIR/train.parquet')))")"

# ================= Step 2: Train =================

echo ""
echo "Step 2: Starting training..."
echo "NOTE: Adjust the training command below for your setup."
echo "This is a template - you need to adapt it to your VERL training script."
echo ""

# Example training command (adapt to your setup):
# python -m verl.trainer.main_ppo \
#     --config swe_agent/config/swe_bench_verified_example.yaml \
#     --data.train_files $DATA_DIR/train.parquet \
#     --data.val_files $DATA_DIR/test.parquet \
#     --model.actor.path $MODEL_PATH \
#     --model.critic.path $MODEL_PATH \
#     --model.ref.path $MODEL_PATH \
#     --checkpoint.save_dir $CHECKPOINT_DIR \
#     --logging.log_dir $LOG_DIR

echo "Training command template shown above."
echo "Please customize it for your VERL setup and uncomment to run."

# ================= Monitoring =================

echo ""
echo "To monitor training:"
echo "  - Logs: $LOG_DIR"
echo "  - Checkpoints: $CHECKPOINT_DIR"
echo "  - TensorBoard: tensorboard --logdir $LOG_DIR"
