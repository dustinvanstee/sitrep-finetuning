# INDOPACOM SITREP Fine-Tuning - Training Summary V2

**Date**: January 29, 2026  
**Model**: nvidia/Nemotron-Mini-4B-Instruct  
**Method**: LoRA (Parameter-Efficient Fine-Tuning)  
**Hardware**: 4x NVIDIA H100 (94GB each)

---

## Training Configuration

### Final Settings (`config/automodel_training_config.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Base Model** | nvidia/Nemotron-Mini-4B-Instruct | 4B parameter model |
| **LoRA Rank** | 64 | ~92M trainable params (2.2% of model) |
| **LoRA Alpha** | 128 | Scaling factor = alpha/rank = 2 |
| **Learning Rate** | 1e-5 → 1e-6 | Cosine decay |
| **Global Batch Size** | 12 | 4 GPUs × 3 samples each |
| **Local Batch Size** | 3 | Per-GPU batch size |
| **Epochs** | 20 | ~140 steps total |
| **Precision** | BF16 | Mixed precision |

### Distributed Training

```yaml
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: 4                # Data parallel sharding
  dp_replicate_size: 1      # Pure FSDP (no replication)
  tp_size: 1                # No tensor parallelism needed
  defer_fsdp_grad_sync: true
```

---

## Dataset

| Split | Samples | Source |
|-------|---------|--------|
| Train | 83 | `data/processed/train.jsonl` |
| Val | 27 | `data/processed/val.jsonl` |

**Format**: JSONL with `{"input": "...", "output": "..."}` pairs  
**Task**: Generate 8-section military SITREP from intelligence reports

---

## Training Results

### Final Run (LoRA rank=64, 20 epochs)

| Metric | Start | Best (epoch 14) | Final (epoch 19) |
|--------|-------|-----------------|------------------|
| **Train Loss** | 1.57 | 0.63 | 0.57 |
| **Val Loss** | 1.55 | **0.83** | 0.83 |

**Best Checkpoint**: `checkpoints/LOWEST_VAL/model/` (epoch 14, step 104)

### Loss Progression

```
Epoch 0:  train=1.57, val=1.55
Epoch 5:  train=1.06, val=1.12
Epoch 10: train=0.78, val=0.89
Epoch 14: train=0.63, val=0.83  ← BEST
Epoch 19: train=0.57, val=0.83  (slight overfit)
```

### Training Speed

| Metric | Value |
|--------|-------|
| **Wall Time** | ~5 minutes |
| **Throughput** | ~42-47K tokens/sec |
| **Per-GPU Throughput** | ~10-12K tokens/sec/GPU |
| **Memory Usage** | 45-70 GiB per GPU (peak ~70 GiB) |

---

## Batch Size Tuning Results

Tested various batch sizes to maximize throughput:

| Local Batch | Global Batch | Peak Memory | Throughput | Status |
|-------------|--------------|-------------|------------|--------|
| 1 | 4 | ~25 GiB | ~16K tps | ✓ Conservative |
| 2 | 8 | ~45 GiB | ~27K tps | ✓ Good |
| **3** | **12** | **~65 GiB** | **~46K tps** | **✓ Optimal** |
| 4 | 16 | ~84 GiB | ~49K tps | ✗ OOM on long sequences |

**Recommendation**: `local_batch_size: 3` for H100 with variable-length sequences.

---

## Commands

### Run Training

```bash
cd /home/dvanstee/projects/2026-01-nt3-sitrep

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

### Pipecleaner (Quick Test)

```bash
# Uses config/automodel_training_config_pipecleaner.yaml (10 steps only)
docker run --rm --gpus all \
    --ipc=host --shm-size=16g \
    -v $(pwd):/workspace \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nvcr.io/nvidian/nemo-automodel:26.02.rc0 \
    bash -c "cd /workspace && torchrun --nproc_per_node=4 \
        /opt/Automodel/examples/llm_finetune/finetune.py \
        -c config/automodel_training_config_pipecleaner.yaml"
```

---

## Key Files

| File | Purpose |
|------|---------|
| `config/automodel_training_config.yaml` | Main training config (20 epochs, rank=64) |
| `config/automodel_training_config_pipecleaner.yaml` | Quick test config (10 steps) |
| `checkpoints/LOWEST_VAL/model/` | Best LoRA adapter (~180MB) |
| `checkpoints/training.jsonl` | Step-by-step training metrics |
| `checkpoints/validation.jsonl` | Validation metrics |
| `data/processed/train.jsonl` | Training data (83 samples) |
| `data/processed/val.jsonl` | Validation data (27 samples) |

---

## LoRA Configuration Notes

### Rank Selection Guide

| Rank | Trainable Params | Use Case |
|------|------------------|----------|
| 8 | ~12M | Simple tasks, tiny datasets |
| 16 | ~23M | Light adaptation |
| 32 | ~46M | Good default |
| **64** | **~92M** | **Used here - complex formatting task** |
| 128 | ~184M | Near full fine-tune quality |

### Memory Optimization Options (if needed)

1. **Activation Checkpointing**: `activation_checkpointing: true` saves ~30% memory, costs ~25% speed
2. **Gradient Accumulation**: Reduce `local_batch_size`, keep `global_batch_size` same
3. **CPU Offloading**: Last resort, significant slowdown

---

## Evaluation

Run evaluation with:

```bash
# Regex-based formatting check (fast)
python scripts/4_evaluate_model_v2.py \
    --finetuned-model checkpoints/LOWEST_VAL/model \
    --no-llm-judge

# With LLM-as-judge (requires API key)
python scripts/4_evaluate_model_v2.py \
    --finetuned-model checkpoints/LOWEST_VAL/model \
    --judge-provider nvidia \
    --judge-model nvidia/llama-3.1-nemotron-70b-instruct
```

---

## Comparison: Run 1 vs Run 2

| Config | Run 1 | Run 2 (Final) |
|--------|-------|---------------|
| LoRA Rank | 32 | **64** |
| Epochs | 10 | **20** |
| Train Loss | 0.98 | **0.57** |
| Val Loss | 1.06 | **0.83** |
| Improvement | - | **22% lower val loss** |
