#!/bin/bash
# Full training run on 4x H100 GPUs
# Run from project root: ./scripts/run_training_4gpu.sh

set -e

WORKSPACE_DIR=$(pwd)
CONFIG_FILE="config/automodel_training_config.yaml"
CONTAINER="nvcr.io/nvidian/nemo-automodel:26.02.rc0"
NUM_GPUS=4

echo "=============================================="
echo "FULL TRAINING RUN (4x H100)"
echo "=============================================="
echo "Workspace: $WORKSPACE_DIR"
echo "Config: $CONFIG_FILE"
echo "Container: $CONTAINER"
echo "GPUs: $NUM_GPUS"
echo ""
echo "Training settings:"
echo "  - LoRA rank: 64"
echo "  - Epochs: 20"
echo "  - Global batch size: 12 (4 GPUs x 3 samples)"
echo "  - Local batch size: 3 per GPU (~65 GiB memory)"
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

# Handle existing checkpoints
if [ -d "checkpoints" ] && [ "$(ls -A checkpoints 2>/dev/null)" ]; then
    echo ""
    echo "WARNING: Existing checkpoints found in checkpoints/"
    echo "NeMo Automodel will auto-resume from LATEST checkpoint."
    echo ""
    read -p "Clear checkpoints and start fresh? (y/N): " CLEAR_CKPT
    if [ "$CLEAR_CKPT" = "y" ] || [ "$CLEAR_CKPT" = "Y" ]; then
        BACKUP_DIR="checkpoints.bak.$(date +%Y%m%d_%H%M%S)"
        echo "Backing up to $BACKUP_DIR..."
        mv checkpoints "$BACKUP_DIR"
        mkdir -p checkpoints
        echo "Checkpoints cleared. Starting fresh."
    else
        echo "Resuming from existing checkpoint."
    fi
fi

read -p "Continue with training? (y/N): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting distributed training on $NUM_GPUS GPUs..."
echo "Estimated time: ~5 minutes"
echo ""

# Run training in Docker container with all GPUs
# Key flags:
#   --ipc=host --shm-size=16g: Required for PyTorch distributed
#   --ulimit memlock=-1: Allows unlimited memory locking (CUDA pinned memory)
#   --ulimit stack=67108864: 64MB stack size (prevents stack overflow)
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: Better memory fragmentation handling
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
echo "TRAINING COMPLETE"
echo "=============================================="
echo ""
echo "Checkpoints saved to: checkpoints/"
echo ""
echo "Best checkpoint (lowest val loss):"
ls -la checkpoints/LOWEST_VAL 2>/dev/null || echo "  (symlink not found)"
echo ""
echo "Next steps:"
echo "1. Best LoRA adapter is at: checkpoints/LOWEST_VAL/model/"
echo "2. Run evaluation:"
echo "   python scripts/4_evaluate_model.py \\"
echo "       --finetuned-model checkpoints/LOWEST_VAL/model \\"
echo "       --no-llm-judge"
