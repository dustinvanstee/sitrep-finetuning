#!/bin/bash
# Run model evaluation with 4 GPUs and cloud baseline comparison
#
# Usage:
#   ./scripts/4_run_evaluation.sh                       # Full eval (all samples)
#   ./scripts/4_run_evaluation.sh --num-samples 5       # Quick test (5 samples)
#   ./scripts/4_run_evaluation.sh --no-cloud-baseline   # Skip cloud model (faster)
#
# Pass any additional arguments to the Python script

set -e

cd /home/dvanstee/projects/2026-01-nt3-sitrep

echo "=============================================="
echo "SITREP Model Evaluation (4 GPU)"
echo "=============================================="
echo "Checkpoint: checkpoints/LOWEST_VAL/model"
echo "Cloud baseline: ENABLED (use --no-cloud-baseline to skip)"
echo "Additional args: $@"
echo "=============================================="

EXTRA_ARGS="$*"

docker run --gpus '"device=0,1,2,3"' --rm \
  --ipc=host --shm-size=32g \
  -v $(pwd):/workspace \
  -w /workspace \
  --env-file .env \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  nvcr.io/nvidian/nemo-automodel:26.02.rc0 \
  bash -c "pip install -q python-dotenv openai matplotlib seaborn && \
    python scripts/4_evaluate_model.py \
      --compare \
      --finetuned-model checkpoints/LOWEST_VAL/model \
      $EXTRA_ARGS"

echo ""
echo "Results saved to: outputs/evaluation_results_v2/"
