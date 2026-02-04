# INDOPACOM SITREP Fine-Tuning with NeMo Automodel

Fine-tune NVIDIA Nemotron-Mini-4B-Instruct to generate structured military situation reports (SITREPs) using NeMo Automodel and NeMo Data Designer.

## Overview

Pipeline stages:
1. **Synthetic data generation** - NeMo Data Designer (v2)
2. **Training data preparation** - Convert to JSONL format
3. **Fine-tuning with LoRA** - NeMo Automodel on 4x H100 GPUs
4. **Evaluation** - Compare baseline vs fine-tuned model

**Results:** Val loss improved from 1.55 → 0.83 (47% reduction). LoRA adapter is ~180MB.

## Prerequisites

### Local Machine
- Python 3.10+ (3.12 recommended)
- NVIDIA API key ([get one](https://build.nvidia.com/))

### GPU Server
- 4x NVIDIA H100 GPUs (or adjust batch size for other GPUs)
- Docker with NVIDIA Container Toolkit
- NGC credentials for NeMo Automodel container
  - **Note:** Access to `nvcr.io/nvidian/nemo-automodel:26.02.rc0` requires membership in the **nvidian** group within NGC.

## Quick Setup

### Data Designer (for data generation)
```bash
git clone https://github.com/NVIDIA-NeMo/DataDesigner.git
cd DataDesigner && python -m venv .venv && source .venv/bin/activate
make install
data-designer config providers && data-designer config models
cd ..
```

### GPU Server
```bash
echo 'nvapi-YOUR_NGC_API_KEY' | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker pull nvcr.io/nvidian/nemo-automodel:26.02.rc0
```

## Usage

### 1. Generate Synthetic Data
```bash
DataDesigner/.venv/bin/python scripts/1_generate_synthetic_data.py --count 100
```

### 2. Prepare Training Data
```bash
DataDesigner/.venv/bin/python scripts/2_prepare_training_data.py
```

### 3. Train (4x H100 GPUs)
```bash
./scripts/run_training_4gpu.sh
```

Or run directly:
```bash
docker run --rm --gpus all \
    --ipc=host --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nvcr.io/nvidian/nemo-automodel:26.02.rc0 \
    bash -c "cd /workspace && torchrun --nproc_per_node=4 \
        /opt/Automodel/examples/llm_finetune/finetune.py \
        -c config/automodel_training_config.yaml"
```

### 4. Evaluate
```bash
python scripts/4_evaluate_model.py \
    --finetuned-model checkpoints/LOWEST_VAL/model \
    --no-llm-judge
```

## Training Configuration

**File:** `config/automodel_training_config.yaml`

Key settings:
```yaml
model:
  pretrained_model_name_or_path: nvidia/Nemotron-Mini-4B-Instruct
  force_hf: true  # CRITICAL - forces fresh HuggingFace load

peft:
  dim: 64        # LoRA rank (~92M trainable params)
  alpha: 128     # LoRA scaling (alpha/rank = 2.0)

step_scheduler:
  global_batch_size: 12   # 4 GPUs x 3 samples
  local_batch_size: 3     # ~65 GiB per GPU on H100
  num_epochs: 20
  max_steps: 140

distributed:
  dp_size: 4              # 4-GPU FSDP sharding
  dp_replicate_size: 1    # Pure FSDP (no replication)
```

See `docs/TRAINING_V2.md` for full training details and batch size tuning results.

## Batch Size Guide (H100 94GB)

| Local Batch | Memory | Throughput | Recommendation |
|-------------|--------|------------|----------------|
| 1 | ~25 GiB | ~16K tps | Conservative |
| 2 | ~45 GiB | ~27K tps | Good |
| **3** | **~65 GiB** | **~46K tps** | **Optimal** |
| 4 | ~84 GiB | ~49K tps | OOM risk |

## Troubleshooting

### `Missing key in checkpoint state_dict: lm_head.weight`
Set `force_hf: true` in config.

### GPU Out of Memory
- Reduce `local_batch_size` to 2 or 1
- Enable `activation_checkpointing: true` in distributed config

### NCCL Timeout
- Increase `dist_env.timeout_minutes` (default 5)
- Check GPU connectivity with `nvidia-smi topo -m`

## Inference

Load the fine-tuned model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-Mini-4B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-Mini-4B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)
model = PeftModel.from_pretrained(base_model, "checkpoints/LOWEST_VAL/model")

# Apply chat template for inference
messages = [{"role": "user", "content": your_intel_reports}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/run_training_4gpu.sh` | Main training script |
| `scripts/run_training_pipecleaner.sh` | Quick 10-step test |
| `config/automodel_training_config.yaml` | Full training config |
| `config/automodel_training_config_pipecleaner.yaml` | Test config |
| `checkpoints/LOWEST_VAL/model/` | Best LoRA adapter |
| `docs/TRAINING_V2.md` | Detailed training summary |

## References

- [NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel)
- [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)
- [Nemotron Models](https://huggingface.co/collections/nvidia/nemotron-mini-6797b7e73cdd12e5e7c30960)
