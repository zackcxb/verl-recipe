#!/bin/bash
# Helper script to prepare SWE-bench_Verified dataset

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-/data1/dataset/SWE-bench_Verified}"
OUTPUT_DIR="${OUTPUT_DIR:-$VERL_ROOT/data/swe_agent_verified}"
SKIP_MISSING="${SKIP_MISSING:-true}"

echo "=== SWE-bench_Verified Data Preparation ==="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Skip missing images: $SKIP_MISSING"
echo ""

# Check if input data exists
if [ ! -f "$DATA_DIR/data/test-00000-of-00001.parquet" ]; then
    echo "Error: SWE-bench_Verified data not found at $DATA_DIR"
    echo "Please download the dataset or set DATA_DIR environment variable"
    exit 1
fi

# Run data preparation
cd "$VERL_ROOT"

SKIP_FLAG=""
if [ "$SKIP_MISSING" = "true" ]; then
    SKIP_FLAG="--skip_missing_images"
fi

python -m swe_agent.prepare.prepare_data \
    --mode swebench_verified \
    --swebench_verified_train "$DATA_DIR/data/test-00000-of-00001.parquet" \
    --swebench_verified_test "$DATA_DIR/data/test-00000-of-00001.parquet" \
    $SKIP_FLAG \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Data Preparation Complete ==="
echo "Train data: $OUTPUT_DIR/train.parquet"
echo "Test data: $OUTPUT_DIR/test.parquet"
echo ""
echo "Next steps:"
echo "1. Review the generated data"
echo "2. Update training config to point to $OUTPUT_DIR"
echo "3. Run training: python -m verl.trainer.main_ppo --config <your_config.yaml>"
