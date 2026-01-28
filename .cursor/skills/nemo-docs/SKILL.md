---
name: nemo-docs
description: Fetch and explain NeMo Automodel documentation from NVIDIA. Use when the user asks about NeMo Automodel features, APIs, configuration options, supported models, or best practices. Triggers on questions about automodel, recipes, YAML config fields, PEFT/LoRA settings, or dataset configuration.
---

# NeMo Automodel Documentation Lookup

Fetch live documentation from NVIDIA's NeMo Automodel docs to answer framework questions accurately.

## When to Use

- User asks "how do I configure X in NeMo Automodel?"
- User asks about supported models, datasets, or features
- User needs clarification on YAML config fields
- User asks about PEFT, LoRA, or training options
- User asks "what does X mean in the automodel config?"

## Workflow

### Step 1: Fetch Documentation

Use WebFetch to retrieve the relevant documentation page:

**Main index (overview, features, supported models):**
```
WebFetch: https://docs.nvidia.com/nemo/automodel/latest/index.html
```

**Specific topics - append to base URL:**
| Topic | URL Path |
|-------|----------|
| SFT/PEFT Guide | `/recipes/llm-peft.html` |
| Dataset Configuration | `/datasets/column-mapped-text-instruction-dataset.html` |
| Checkpointing | `/development/checkpointing.html` |
| LoRA/PEFT Config | `/development/api-reference.html` |
| Supported LLMs | `/model-coverage/llms.html` |
| Supported VLMs | `/model-coverage/vlms.html` |

### Step 2: Extract Relevant Information

After fetching, extract:
1. The specific config fields or API mentioned
2. Code examples or YAML snippets
3. Any warnings or critical notes (like `force_hf: true`)

### Step 3: Relate to This Project

Connect the documentation to this project's files:
- `config/automodel_training_config.yaml` - main training config
- `src/evaluation/inference_utils.py` - chat template usage
- `data/processed/*.jsonl` - dataset format

## Key Documentation Sections

### Config Field Reference
When asked about YAML fields, look for:
- `_target_:` - The Python class being instantiated
- `model:` - Base model settings
- `peft:` - LoRA adapter configuration  
- `step_scheduler:` - Batch size, steps, checkpointing
- `dataset:` - Data loading and column mapping
- `optimizer:` / `lr_scheduler:` - Training hyperparameters

### Common Questions

**"What models are supported?"**
→ Fetch index.html, look at Supported Models table

**"How do I configure LoRA?"**
→ Fetch the PEFT guide, extract `peft:` section examples

**"What dataset format does it expect?"**
→ Fetch dataset docs, explain ColumnMappedTextInstructionDataset

**"Why do I need force_hf: true?"**
→ Explain: prevents loading cached checkpoint metadata that may be incompatible

## Example Usage

User: "What LoRA settings should I use for a 4B model?"

1. Fetch: `https://docs.nvidia.com/nemo/automodel/latest/recipes/llm-peft.html`
2. Extract relevant PEFT config examples
3. Compare to `config/automodel_training_config.yaml` in this project
4. Provide recommendation with context
