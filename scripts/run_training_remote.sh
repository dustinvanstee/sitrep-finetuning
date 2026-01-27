#!/bin/bash
set -e

cd /workspace

echo "=== Starting Fine-Tuning ==="
echo "Config: config/automodel_training_config.yaml"
echo "Model: nvidia/Nemotron-Mini-4B-Instruct"
echo "GPU: GPU 3 (mapped to device 0 inside container)"
echo "Training data: 6 examples"
echo "Validation data: 1 example"
echo ""

python Automodel/examples/llm_finetune/finetune.py -c config/automodel_training_config.yaml
