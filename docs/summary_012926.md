# SITREP Fine-Tuning Project Summary - January 29, 2026

## Current State

### Data Generation Complete
- **203 scenarios** generated in `data/scenarios/`
- **202 training samples** in `data/scenarios/training_data.jsonl`
- Previous data backed up to `data/scenarios.v0/`
- Generation took ~5.1 hours using local NIM

### Key Configuration Files

| File | Purpose |
|------|---------|
| `config/automodel_training_config.yaml` | NeMo Automodel training config (LoRA, batch size, etc.) |
| `config/evaluation_config_v2.yaml` | 3-way evaluation config with instruction prompt |
| `.data-designer/model_providers.yaml` | Data Designer LLM provider (local-nim) |

### Training Data Format

The training data now includes an **explicit instruction prompt** in the `input` field:

```
input: [INSTRUCTION_PROMPT] + [RAW_INTEL_INPUTS]
output: [SITREP]
raw_input: [RAW_INTEL_INPUTS only - for reference]
```

The instruction prompt ensures the model learns to generate properly formatted SITREPs with:
- Numbered section headers (1. BLUF, 2. FRIENDLY FORCES, etc.)
- Section dividers (`---`)
- Bullet points with `-`

---

## Next Steps

### 1. Prepare Training Data
```bash
python scripts/2_prepare_training_data_v2.py
```
Creates train/val splits from `training_data.jsonl`.

### 2. Run Training
```bash
# 4-GPU training
./scripts/run_training_4gpu.sh

# Or pipecleaner (quick test)
./scripts/run_training_pipecleaner.sh
```

### 3. Evaluate Model
```bash
# Full evaluation (regex + LLM judge, 3-way comparison)
docker run --gpus all --rm -it \
  -v $(pwd):/workspace \
  -w /workspace \
  --env-file .env \
  nvcr.io/nvidian/nemo-automodel:26.02.rc0 \
  bash -c "pip install python-dotenv openai matplotlib seaborn && python scripts/4_evaluate_model_v2.py"

# Pipecleaner (1 sample)
# Add: --num-samples 1
```

---

## Evaluation Strategy (3-Way Comparison)

| Model | Description |
|-------|-------------|
| **Finetuned** | Your LoRA-adapted model |
| **Local Baseline** | Base Nemotron-Mini-4B-Instruct |
| **Cloud Baseline** | Nemotron-Super-49B-v1.5 (robust reference) |

### Metrics
1. **Regex Formatting** - Fast, deterministic section/structure checks
2. **LLM-as-Judge** - Nemotron-Super-49B scores quality aspects

### Key Insight
Local models require the **same instruction prompt** during inference that was used during training. The evaluation script automatically prepends this prompt (disable with `--no-instruction-prompt`).

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/1_generate_synthetic_data_v2.py` | Generate scenarios with local NIM |
| `scripts/2_prepare_training_data_v2.py` | Create train/val splits |
| `scripts/run_training_4gpu.sh` | Launch training in Docker |
| `scripts/4_evaluate_model_v2.py` | 3-way evaluation with regex + LLM judge |

---

## Model Outputs

- **Checkpoints**: `checkpoints/epoch_N_step_M/`
- **Final adapter**: `outputs/models/finetuned-4b/`
- **Evaluation results**: `outputs/evaluation_results/`

---

## Infrastructure Notes

- **Local NIM**: Required for data generation. Check with:
  ```bash
  curl http://localhost:8000/v1/models
  ```
- **Docker container**: `nvcr.io/nvidian/nemo-automodel:26.02.rc0`
- **Training**: Requires 4x GPUs (adjust `run_training_4gpu.sh` for different configs)
