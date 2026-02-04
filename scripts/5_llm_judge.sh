#!/bin/bash
# Run LLM-as-a-Judge evaluation using GPT 5.2 on NVIDIA Astra API
#
# Prerequisites:
#   - Step 4 outputs must exist in outputs/evaluation_results_v2/
#   - Python venv with dependencies installed
#
# Usage:
#   ./scripts/5_llm_judge.sh                    # Default: 1 sample, all models
#   ./scripts/5_llm_judge.sh --num-samples 5    # Evaluate 5 samples
#   ./scripts/5_llm_judge.sh --finetuned-only   # Only evaluate fine-tuned model
#   ./scripts/5_llm_judge.sh --no-cloud-baseline # Skip cloud baseline
#
# Pass any additional arguments to the Python script

set -e

cd /home/dvanstee/projects/2026-01-nt3-sitrep

echo "=============================================="
echo "LLM-as-a-Judge Evaluation (GPT 5.2 / Astra)"
echo "=============================================="
echo "Judge Model: openai/openai/gpt-5.2"
echo "API: inference-api.nvidia.com"
echo "Additional args: $@"
echo "=============================================="

# Check for ASTRA_API_KEY (set in .bashrc or environment)
if [ -z "$ASTRA_API_KEY" ]; then
    echo "ERROR: ASTRA_API_KEY not set. Add to .bashrc or export before running."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Run LLM judge with Astra provider and GPT 5.2
python scripts/5_llm_judge_eval.py \
    --judge-provider astra \
    --judge-model "openai/openai/gpt-5.2" \
    --num-samples 1 \
    "$@"

echo ""
echo "Results saved to: outputs/evaluation_results_v2/llm_judge/"
