#!/bin/bash
# Pipecleaner training run - 10 steps on 4x H100 GPUs
# Run from project root: ./scripts/run_training_pipecleaner.sh

set -e

WORKSPACE_DIR=$(pwd)
CONFIG_FILE="config/automodel_training_config_pipecleaner.yaml"
CONTAINER="nvcr.io/nvidian/nemo-automodel:26.02.rc0"
NUM_GPUS=4

echo "=============================================="
echo "PIPECLEANER TRAINING RUN (4x H100)"
echo "=============================================="
echo "Workspace: $WORKSPACE_DIR"
echo "Config: $CONFIG_FILE"
echo "Container: $CONTAINER"
echo "GPUs: $NUM_GPUS"
echo "Steps: 10 (quick test)"
echo "Global batch size: 12 (4 GPUs x 3 samples)"
echo ""
echo "Training data: $(wc -l < data/processed/train.jsonl 2>/dev/null || echo '?') examples"
echo "Validation data: $(wc -l < data/processed/val.jsonl 2>/dev/null || echo '?') examples"
echo "=============================================="
echo ""

# Check if data exists
if [ ! -f "data/processed/train.jsonl" ]; then
    echo "ERROR: Training data not found!"
    echo "Run: DataDesigner/.venv/bin/python scripts/2_prepare_training_data.py"
    exit 1
fi

# Handle existing checkpoints - auto-clear for pipecleaner
if [ -d "checkpoints" ] && [ "$(ls -A checkpoints 2>/dev/null)" ]; then
    echo ""
    echo "Existing checkpoints found. Backing up and starting fresh..."
    BACKUP_DIR="checkpoints.bak.$(date +%Y%m%d_%H%M%S)"
    mv checkpoints "$BACKUP_DIR"
    mkdir -p checkpoints
    echo "Backed up to $BACKUP_DIR"
fi

# Option to override GPU count
read -p "Number of GPUs to use (default $NUM_GPUS): " INPUT_GPUS
NUM_GPUS=${INPUT_GPUS:-$NUM_GPUS}

echo ""
echo "Starting distributed training on $NUM_GPUS GPUs..."
echo ""

# Run training in Docker container with all GPUs
docker run --rm --gpus all \
    --ipc=host --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WORKSPACE_DIR":/workspace \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $CONTAINER \
    bash -c "cd /workspace && torchrun --nproc_per_node=$NUM_GPUS /opt/Automodel/examples/llm_finetune/finetune.py -c $CONFIG_FILE"

echo ""
echo "=============================================="
echo "PIPECLEANER COMPLETE"
echo "=============================================="
echo "Checkpoints saved to: checkpoints/"
