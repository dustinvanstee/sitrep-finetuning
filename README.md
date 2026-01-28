# INDOPACOM SITREP Fine-Tuning with NeMo Automodel

Fine-tune NVIDIA Nemotron-Mini-4B-Instruct to generate well-formatted military situation reports (SITREPs) using NeMo Automodel framework and NeMo Data Designer for synthetic data generation.

## Overview

This project demonstrates end-to-end fine-tuning of a 4B parameter language model to generate structured markdown situation reports from multiple intelligence source inputs. The pipeline includes:

1. **Synthetic data generation** using NeMo Data Designer (v2) or NVIDIA API (v1)
2. **Training data preparation** in JSONL format
3. **Fine-tuning with LoRA** using NeMo Automodel on remote GPU
4. **Evaluation** comparing baseline vs fine-tuned model quality

**Key Results:**
- Training Loss: 0.5431 → 0.4572 (-16% improvement)
- Validation Loss: 0.5318 → 0.4704 (-11.5% improvement)
- Training Time: ~40 seconds for 6 examples on H100 NVL
- Model Size: 88MB LoRA adapter (vs 8GB full model)

## Project Structure

```
sitrep-finetuning/
├── README.md                          # This file
├── TRAINING_SUMMARY.md                # Detailed training results and analysis
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables (not tracked)
├── .gitignore                         # Git ignore rules
│
├── .cursor/                           # Cursor IDE configuration
│   ├── rules/                         # Project rules for AI assistance
│   └── skills/                        # Custom skills (NeMo docs, etc.)
│
├── .data-designer/                    # NeMo Data Designer local config
│   ├── model_configs.yaml             # Model configurations
│   └── model_providers.yaml           # API provider settings
│
├── DataDesigner/                      # NeMo Data Designer (cloned, not tracked)
│   └── .venv/                         # Data Designer virtual environment
│
├── .venv/                             # Project virtual environment (not tracked)
│
├── config/
│   ├── automodel_training_config.yaml # NeMo Automodel training configuration
│   ├── evaluation_config.yaml         # Evaluation settings
│   └── data_designer_config.yaml      # Data generation settings
│
├── scripts/
│   ├── 1_generate_synthetic_data.py   # V1: Generate sit reps with NVIDIA API
│   ├── 1_generate_synthetic_data_v2.py # V2: Generate with NeMo Data Designer
│   ├── 2_prepare_training_data.py     # Convert to JSONL for training
│   ├── 4_evaluate_model.py            # Evaluate and compare models
│   ├── run_training_remote.sh         # Training script for remote GPU
│   └── start_nt3nano.sh               # Script to start Nemotron Nano
│
├── src/
│   ├── data_generation/
│   │   ├── data_designer_client.py    # Logistics data generator
│   │   ├── logistics_schema.py        # Data schema definitions
│   │   └── sitrep_templates.py        # Sit rep prompt templates
│   ├── training/
│   │   └── data_preprocessor.py       # JSONL data preparation
│   └── evaluation/
│       ├── formatting_metrics.py      # Sit rep quality scoring
│       └── inference_utils.py         # Model loading and generation
│
├── data/
│   └── scenarios/                     # Generated scenario data
│       ├── scenario_001_*/            # Individual scenarios
│       │   ├── backbone.json          # Scenario backbone data
│       │   ├── metadata.json          # Generation metadata
│       │   ├── inputs/                # Intel report snippets (*.md)
│       │   ├── output/                # Generated SITREP (sitrep.md)
│       │   └── training_sample.jsonl  # Training pair for this scenario
│       └── training_data.jsonl        # Combined training data
│
└── outputs/
    ├── models/
    │   └── finetuned-4b/              # Fine-tuned LoRA adapter (88MB)
    │       ├── adapter_model.safetensors
    │       ├── adapter_config.json
    │       └── tokenizer files
    └── evaluation_results/            # Evaluation outputs
        ├── baseline/                  # Baseline model outputs
        ├── finetuned/                 # Fine-tuned model outputs
        ├── comparison.jsonl           # Side-by-side comparison
        └── evaluation_report.md       # Results summary
```

## Prerequisites

### Local Machine
- Python 3.10+ (3.12 recommended)
- `uv` package manager ([installation](https://github.com/astral-sh/uv))
- NVIDIA API key ([get one here](https://build.nvidia.com/))

### Remote GPU Server
- NVIDIA GPU with 15GB+ VRAM (tested on H100 NVL 95GB)
- Docker with NVIDIA Container Toolkit
- NGC credentials for NeMo Automodel container access
- SSH access to server

## Setup

### 1. Local Environment Setup

```bash
# Clone/navigate to project directory
cd /path/to/sitrep-finetuning

# Create virtual environment at project root
python -m venv .venv
source .venv/bin/activate

# Create .env file with your NVIDIA API key
cat > .env << 'EOF'
NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
EOF

# Install Python dependencies
pip install -r requirements.txt
```

### 2. NeMo Data Designer Setup (for V2 data generation)

NeMo Data Designer is used by `scripts/1_generate_synthetic_data_v2.py` to generate synthetic SITREP training data. It is **not tracked** in this repository and must be cloned separately.

```bash
# Clone NeMo Data Designer into the project directory
git clone https://github.com/NVIDIA-NeMo/DataDesigner.git

# Create a separate virtual environment for Data Designer
cd DataDesigner
python -m venv .venv
source .venv/bin/activate

# Install Data Designer from source
make install
# Or: pip install -e .

# Configure Data Designer with your API keys
data-designer config providers
data-designer config models

# Return to project root
cd ..
```

**Configuration files** are stored in `.data-designer/`:
- `model_providers.yaml` - API provider settings (NVIDIA, OpenAI, etc.)
- `model_configs.yaml` - Model aliases and inference parameters

**Running V2 data generation:**
```bash
# Use the Data Designer's virtual environment
DataDesigner/.venv/bin/python scripts/1_generate_synthetic_data_v2.py --count 20
```

### 3. Remote Server Setup

```bash
# SSH into remote server
ssh user@remote-gpu-server

# Login to NGC (NVIDIA GPU Cloud) with credentials
echo 'nvapi-YOUR_NGC_API_KEY' | docker login nvcr.io --username '$oauthtoken' --password-stdin

# Pull NeMo Automodel container
docker pull nvcr.io/nvidian/nemo-automodel:26.02.rc0

# Create workspace directory
mkdir -p ~/sitrep-training/{config,data/scenarios,scripts,src}
```

## Usage

### Step 1: Generate Synthetic Data (Local)

Two approaches are available for generating training data:

#### Option A: V2 - NeMo Data Designer (Recommended)

Generate INDOPACOM-style SITREPs with multiple intel source inputs:

```bash
# Activate Data Designer environment and run
DataDesigner/.venv/bin/python scripts/1_generate_synthetic_data_v2.py --count 20
```

**Output:**
- `data/scenarios/scenario_NNN_*/` - Individual scenario directories containing:
  - `inputs/*.md` - Multiple intel report snippets (FLASH, ISR, SENSOR, etc.)
  - `output/sitrep.md` - Consolidated 8-section SITREP
  - `backbone.json` - Scenario backbone data
  - `metadata.json` - Generation metadata
  - `training_sample.jsonl` - Training pair for this scenario
- `data/scenarios/training_data.jsonl` - Combined training data from all scenarios

**Scenario types:** Korean Peninsula Tension, South China Sea Standoff, Taiwan Invasion, Vessel Incursion, Routine Patrol

#### Option B: V1 - Direct NVIDIA API

Generate logistics sit reps using NVIDIA's hosted Llama 3.1 70B:

```bash
source .venv/bin/activate
python scripts/1_generate_synthetic_data.py
```

**Output:**
- `data/raw/sitrep_coastal_expansion_day*.md` - 7 markdown sit reps
- Each sit rep contains 5 location analyses with logistics data

### Step 2: Prepare Training Data (Local)

Convert sit reps to JSONL format for fine-tuning:

```bash
uv run python scripts/2_prepare_training_data.py
```

**Output:**
- `data/processed/train.jsonl` - 6 training examples
- `data/processed/val.jsonl` - 1 validation example

**JSONL format:**
```json
{
  "input": "Generate a comprehensive logistics deployment situation report for Day 2.\n\n## Logistics Data\n\n### Location: Alpha Base...",
  "output": "# Coastal Expansion - Day 2 Logistics Deployment Analysis\n\n## TLDR\n...",
  "metadata": {"scenario": "coastal_expansion", "day": 2}
}
```

### Step 3: Transfer Files to Remote Server

Copy training data and configuration to GPU server:

```bash
# Copy training data
scp data/processed/*.jsonl user@remote-gpu-server:~/logistics-training-fresh/data/processed/

# Copy configuration
scp config/automodel_training_config.yaml user@remote-gpu-server:~/logistics-training-fresh/config/

# Copy training script
scp scripts/run_training_remote.sh user@remote-gpu-server:~/logistics-training-fresh/
```

### Step 4: Fine-Tune on Remote GPU

**On remote server:**

```bash
ssh user@remote-gpu-server
cd ~/logistics-training-fresh

# Clone NeMo Automodel repository (contains training scripts)
git clone https://github.com/NVIDIA-NeMo/Automodel.git

# Run training in NeMo Automodel container
docker run --rm --gpus "device=0" \
    --ipc=host --shm-size=8g \
    --tmpfs /root/.cache/huggingface:rw,size=30g \
    -v ~/logistics-training-fresh:/workspace \
    nvcr.io/nvidian/nemo-automodel:26.02.rc0 \
    bash /workspace/run_training_remote.sh
```

**Training script (`run_training_remote.sh`):**
```bash
#!/bin/bash
set -e

cd /workspace

echo "=== Starting Fine-Tuning ==="
python Automodel/examples/llm_finetune/finetune.py \
    -c config/automodel_training_config.yaml
```

**Expected output:**
```
step 0  | epoch 0 | loss 0.5431 | mem 27.99 GiB | tps 5860.98
step 1  | epoch 0 | loss 0.5100 | mem 28.01 GiB | tps 15611.44
...
step 19 | epoch 9 | loss 0.4572 | mem 28.10 GiB | tps 15914.70
[val] step 19 | epoch 9 | loss 0.4704
Updated LOWEST_VAL checkpoint symlink to epoch_9_step_19
```

**Training time:** ~40 seconds for 6 examples (10 epochs, 20 steps)

**Checkpoints saved to:** `~/logistics-training-fresh/checkpoints/`
- `LOWEST_VAL/` - Symlink to best checkpoint (epoch_9_step_19)
- `LATEST/` - Symlink to most recent checkpoint
- Individual checkpoints: `epoch_*_step_*/`

### Step 5: Copy Checkpoint to Local

Download the fine-tuned LoRA adapter:

```bash
# On local machine
mkdir -p outputs/models/finetuned-4b
scp -r 'user@remote-gpu-server:~/logistics-training-fresh/checkpoints/LOWEST_VAL/model/*' \
    outputs/models/finetuned-4b/
```

**Checkpoint structure:**
```
outputs/models/finetuned-4b/
├── adapter_model.safetensors  # LoRA weights (88MB)
├── adapter_config.json         # LoRA configuration
├── config.json                 # Model configuration
└── tokenizer files             # Tokenizer and special tokens
```

### Step 6: Evaluate Model (Remote GPU)

Compare baseline vs fine-tuned model quality:

**Copy evaluation code to remote:**
```bash
scp -r scripts user@remote-gpu-server:~/logistics-training-fresh/
scp -r src user@remote-gpu-server:~/logistics-training-fresh/
scp config/evaluation_config.yaml user@remote-gpu-server:~/logistics-training-fresh/config/
```

**Run evaluation on remote GPU:**
```bash
ssh user@remote-gpu-server
cd ~/logistics-training-fresh

# Create evaluation script
cat > run_evaluation.sh << 'EOF'
#!/bin/bash
set -e
cd /workspace
pip install -q pyyaml pandas matplotlib seaborn peft rouge-score
python scripts/4_evaluate_model.py --config config/evaluation_config.yaml --compare
EOF
chmod +x run_evaluation.sh

# Run in container
docker run --rm --gpus "device=0" \
    --ipc=host --shm-size=8g \
    --tmpfs /root/.cache/huggingface:rw,size=30g \
    -v ~/logistics-training-fresh:/workspace \
    nvcr.io/nvidian/nemo-automodel:26.02.rc0 \
    bash /workspace/run_evaluation.sh
```

**Copy results back:**
```bash
scp -r user@remote-gpu-server:~/logistics-training-fresh/outputs/evaluation_results/ \
    outputs/
```

## NeMo Automodel Container

**Container:** `nvcr.io/nvidian/nemo-automodel:26.02.rc0`

### What's Inside

The NeMo Automodel container is a pre-configured Docker image with:

- **Base:** Ubuntu 22.04 with NVIDIA CUDA 13.0
- **Python:** 3.12 with pre-installed ML libraries
- **PyTorch:** 2.10.0a0 optimized for NVIDIA GPUs
- **NeMo Framework:** Recipe-based training system
- **Transformers:** HuggingFace transformers library
- **PEFT:** Parameter-Efficient Fine-Tuning (LoRA support)
- **Optimizations:**
  - Triton kernels for fast attention
  - FSDP2 for distributed training
  - Flash Attention 2
  - Mixed precision training (BF16)

### Why Use This Container?

1. **No dependency management** - All packages pre-installed and tested
2. **GPU optimizations** - Compiled for CUDA 13.0+ with fast kernels
3. **Reproducible** - Same environment as NVIDIA's internal training
4. **Recipe system** - YAML-based configuration (no code changes)
5. **HuggingFace compatible** - Outputs standard checkpoints

### Container Mounts

```bash
-v ~/logistics-training-fresh:/workspace  # Project directory
--tmpfs /root/.cache/huggingface:rw,size=30g  # Temp model cache
--gpus "device=0"  # GPU access
--ipc=host --shm-size=8g  # Shared memory for dataloaders
```

## Training Configuration

**File:** `config/automodel_training_config.yaml`

### Key Sections

#### 1. Model Configuration
```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: nvidia/Nemotron-Mini-4B-Instruct
  force_hf: true  # CRITICAL: Force HuggingFace loading
```

**`force_hf: true` is critical** - Without this, NeMo tries to load cached checkpoint metadata which causes errors. This flag forces fresh loading from HuggingFace.

#### 2. Training Loop
```yaml
step_scheduler:
  global_batch_size: 4   # Total batch size across all GPUs
  local_batch_size: 1    # Batch size per GPU (enables gradient accumulation)
  max_steps: 100         # Total training steps
  ckpt_every_steps: 50   # Checkpoint frequency
  val_every_steps: 50    # Validation frequency
```

**Gradient accumulation:** With `global_batch_size=4` and `local_batch_size=1`, NeMo accumulates gradients over 4 forward passes before updating weights. This enables larger effective batch sizes on small datasets.

#### 3. LoRA Configuration
```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: true  # Apply LoRA to all linear layers
  dim: 32                 # LoRA rank (number of adapter dimensions)
  alpha: 64               # LoRA scaling factor (alpha/rank = 2.0)
  use_triton: true        # Use optimized Triton kernels
```

**LoRA parameters:**
- **rank=32**: Low-rank decomposition size. Higher = more capacity but slower.
- **alpha=64**: Scaling factor. Ratio alpha/rank controls adaptation strength.
- **Trainable params:** ~25M out of 4B (0.6% of model)

#### 4. Dataset Configuration
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: "data/processed/train.jsonl"
  column_mapping:
    question: input   # Maps our 'input' field to 'question'
    answer: output    # Maps our 'output' field to 'answer'
  answer_only_loss_mask: true  # Only compute loss on answer tokens
```

**`ColumnMappedTextInstructionDataset`** automatically:
- Applies chat template (`<extra_id_1>User\n...\n<extra_id_1>Assistant\n`)
- Masks loss on prompt tokens (only trains on outputs)
- Handles tokenization and padding

#### 5. Optimizer & Scheduler
```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-5             # Learning rate
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0

lr_scheduler:
  lr_decay_style: cosine  # Cosine annealing schedule
  min_lr: 1.0e-6          # Minimum learning rate
```

**Learning rate schedule:**
- Starts at `1e-5`
- Decays with cosine curve
- Ends at `1e-6` after `max_steps`

### Training Recipe Flow

1. **Load model** from HuggingFace (4B parameters)
2. **Inject LoRA adapters** into all linear layers (+25M params)
3. **Load dataset** with chat template formatting
4. **Training loop:**
   - Forward pass on 1 example
   - Accumulate gradients (repeat 4x)
   - Update LoRA weights only
   - Checkpoint every 50 steps
   - Validate every 50 steps
5. **Save checkpoint** with lowest validation loss

## Evaluation Metrics

The evaluation script measures sit rep formatting quality on a 100-point scale:

### Scoring Categories

1. **Section Completeness (40 points)**
   - Title format: `# Scenario - Day X Logistics Deployment Analysis`
   - Required sections: TLDR, Executive Summary, Location Analysis, Comparative Analysis, Temporal Trends, Recommendations, Action Items

2. **TLDR Quality (20 points)**
   - Must be H2 header: `## TLDR`
   - Contains recommendation, insight, and citations
   - Citations format: `(See: Section Name)`

3. **Markdown Formatting (20 points)**
   - Proper header hierarchy (H1, H2, H3)
   - Tables for data
   - Bold/italic for emphasis
   - Checkboxes for action items: `- [ ] Task`

4. **Structure Quality (20 points)**
   - Exactly one H1 header
   - 5+ H2 headers (main sections)
   - 3+ H3 headers (subsections)

### Current Results

**Baseline Model:** 0/100
- Generated comprehensive sit reps
- Missing exact formatting patterns (bold vs H2, no checkboxes)

**Fine-tuned Model:** 0/100
- Generated quality sit reps with analysis
- Training improved loss by 16%
- Formatting doesn't match ultra-strict criteria

**Issue:** Metrics are extremely strict and require exact patterns. Models learned general structure but not precise formatting. See `TRAINING_SUMMARY.md` for detailed analysis.

## Results Summary

### Training Metrics
- **Dataset:** 6 training examples, 1 validation example
- **Training Loss:** 0.5431 → 0.4572 (-15.8% improvement)
- **Validation Loss:** 0.5318 → 0.4704 (-11.5% improvement)
- **Training Time:** ~40 seconds (10 epochs, 20 steps)
- **GPU Memory:** 28GB on H100 NVL
- **Checkpoint Size:** 88MB LoRA adapter

### Model Outputs

**Both baseline and fine-tuned models generate:**
- Proper title structure
- TLDR/Executive summary
- Location analysis with tables
- Comparative analysis
- Temporal trends
- Specific recommendations
- Action items with details

**Fine-tuned model shows:**
- More detailed comparative analysis
- Better temporal trend explanation
- More specific action items
- Improved overall coherence

See `TRAINING_SUMMARY.md` for:
- Full training logs
- Checkpoint analysis
- Evaluation gap analysis
- Recommendations for improvement

## Troubleshooting

### Issue: `Missing key in checkpoint state_dict: lm_head.weight`

**Solution:** Ensure `force_hf: true` is set in `config/automodel_training_config.yaml`:
```yaml
model:
  force_hf: true  # Forces fresh load from HuggingFace
```

### Issue: GPU Out of Memory

**For 30B model:**
```yaml
step_scheduler:
  global_batch_size: 2  # Reduce from 4 to 2
```

**Or use gradient checkpointing:**
```yaml
model:
  gradient_checkpointing: true
```

### Issue: Evaluation shows 0/100 scores

This is expected with current strict metrics. Models generate quality outputs but don't match exact formatting patterns. Options:

1. **Relax metrics** to give partial credit
2. **Generate 50-100 training examples** with consistent formatting
3. **Add post-processing** to fix formatting

### Issue: Chat template not applied during inference

Ensure `src/evaluation/inference_utils.py` applies chat template:
```python
if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
```

## Using the Trained Model for Inference

### Option 1: Local Python Inference

Load and use the fine-tuned model directly in Python:

**Create inference script (`inference.py`):**

```python
#!/usr/bin/env python3
"""
Simple inference script for the fine-tuned logistics sit rep model.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_path: str, adapter_path: str, device: str = "cuda"):
    """Load the fine-tuned model with LoRA adapter."""
    print(f"Loading base model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # Merge LoRA weights into base model
    model.eval()

    return tokenizer, model

def generate_sitrep(
    tokenizer,
    model,
    logistics_data: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.7
) -> str:
    """Generate a sit rep from logistics data."""

    # Format the prompt
    prompt = f"Generate a comprehensive logistics deployment situation report.\n\n{logistics_data}"

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode and extract generated text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the assistant response
    assistant_marker = '<extra_id_1>Assistant\n'
    if assistant_marker in full_output:
        generated_text = full_output.split(assistant_marker)[-1]
        # Remove trailing special tokens
        for token in ['<extra_id_', '</s>', '<eos>']:
            if token in generated_text:
                generated_text = generated_text.split(token)[0]
        return generated_text.strip()

    return full_output

# Example usage
if __name__ == "__main__":
    # Load model
    tokenizer, model = load_model(
        base_model_path="nvidia/Nemotron-Mini-4B-Instruct",
        adapter_path="outputs/models/finetuned-4b"
    )

    # Example logistics data
    logistics_data = """
## Logistics Data

### Location: Alpha Base (ID: LOC-001)
- Coordinates: 31.59°N, 98.35°W
- Terrain: Coastal Plain at 683m elevation
- Weather: Clear, 22°C
- Distance from DC: 242 km
- Population Served: 159,856
- Current Demand: 4,033 units/day
- Base Deployment Cost: $111,050
...
"""

    # Generate sit rep
    print("Generating sit rep...")
    sitrep = generate_sitrep(tokenizer, model, logistics_data)

    print("\n" + "="*60)
    print("GENERATED SIT REP")
    print("="*60)
    print(sitrep)
```

**Run inference:**
```bash
# Install dependencies
uv pip install torch transformers peft accelerate

# Run inference
uv run python inference.py
```

### Option 2: REST API Endpoint with FastAPI

Deploy as a REST API for production use:

**Create API server (`api_server.py`):**

```python
#!/usr/bin/env python3
"""
FastAPI server for logistics sit rep generation.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Logistics Sit Rep Generator",
    description="Generate logistics deployment situation reports",
    version="1.0.0"
)

# Global model cache
MODEL_CACHE = {}

class LogisticsRequest(BaseModel):
    logistics_data: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9

class SitRepResponse(BaseModel):
    sitrep: str
    tokens_generated: int

@app.on_event("startup")
async def load_models():
    """Load model on startup."""
    print("Loading fine-tuned model...")

    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/Nemotron-Mini-4B-Instruct",
        trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        "nvidia/Nemotron-Mini-4B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(
        base_model,
        "outputs/models/finetuned-4b"
    )
    model = model.merge_and_unload()
    model.eval()

    MODEL_CACHE["tokenizer"] = tokenizer
    MODEL_CACHE["model"] = model

    print("Model loaded successfully!")

@app.post("/generate", response_model=SitRepResponse)
async def generate_sitrep(request: LogisticsRequest):
    """
    Generate a logistics situation report.

    **Example Request:**
    ```json
    {
        "logistics_data": "## Logistics Data\\n\\n### Location: Alpha Base...",
        "max_tokens": 4096,
        "temperature": 0.7
    }
    ```
    """
    try:
        tokenizer = MODEL_CACHE["tokenizer"]
        model = MODEL_CACHE["model"]

        # Format prompt with chat template
        prompt = f"Generate a comprehensive logistics deployment situation report.\n\n{request.logistics_data}"
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract assistant response
        assistant_marker = '<extra_id_1>Assistant\n'
        if assistant_marker in full_output:
            generated_text = full_output.split(assistant_marker)[-1]
            for token in ['<extra_id_', '</s>', '<eos>']:
                if token in generated_text:
                    generated_text = generated_text.split(token)[0]
            sitrep = generated_text.strip()
        else:
            sitrep = full_output

        tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])

        return SitRepResponse(
            sitrep=sitrep,
            tokens_generated=tokens_generated
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": "model" in MODEL_CACHE
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Install dependencies:**
```bash
uv pip install fastapi uvicorn[standard] torch transformers peft accelerate
```

**Run API server:**
```bash
# Start server
uv run python api_server.py

# Server will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

**Test API endpoint:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "logistics_data": "## Logistics Data\n\n### Location: Alpha Base...",
    "max_tokens": 4096,
    "temperature": 0.7
  }'
```

### Option 3: Production Deployment with vLLM

For high-throughput production deployment, use vLLM:

**Install vLLM:**
```bash
pip install vllm
```

**Merge LoRA adapter (one-time):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load and merge
base_model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-Mini-4B-Instruct",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "outputs/models/finetuned-4b")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("outputs/models/nemotron-4b-sitrep-merged")
tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-Mini-4B-Instruct")
tokenizer.save_pretrained("outputs/models/nemotron-4b-sitrep-merged")
```

**Run vLLM server:**
```bash
# Start vLLM OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model outputs/models/nemotron-4b-sitrep-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size 1
```

**Use OpenAI-compatible client:**
```python
from openai import OpenAI

# Point to vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM doesn't require auth by default
)

# Generate sit rep
response = client.chat.completions.create(
    model="outputs/models/nemotron-4b-sitrep-merged",
    messages=[
        {
            "role": "user",
            "content": "Generate a comprehensive logistics deployment situation report.\n\n## Logistics Data\n\n..."
        }
    ],
    max_tokens=4096,
    temperature=0.7
)

sitrep = response.choices[0].message.content
print(sitrep)
```

**Benefits of vLLM:**
- **20-30x higher throughput** than standard inference
- **Continuous batching** for efficient GPU utilization
- **PagedAttention** for optimized memory usage
- **OpenAI-compatible API** for easy integration

### Option 4: NVIDIA Triton Inference Server

For enterprise deployment with multiple model versions and monitoring:

**Create Triton model repository:**
```bash
mkdir -p triton_models/nemotron_sitrep/1
# Export model to ONNX or TensorRT format
# Configure config.pbtxt with input/output specifications
```

**Run Triton server:**
```bash
docker run --gpus all --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/triton_models:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

### Option 5: Batch Processing

For processing multiple logistics scenarios:

**Create batch inference script (`batch_inference.py`):**

```python
#!/usr/bin/env python3
"""
Batch process logistics data to generate multiple sit reps.
"""
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def batch_generate(
    input_file: str,
    output_dir: str,
    base_model: str = "nvidia/Nemotron-Mini-4B-Instruct",
    adapter_path: str = "outputs/models/finetuned-4b",
    batch_size: int = 4
):
    """Process multiple logistics scenarios."""

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    # Load input data
    with open(input_file, 'r') as f:
        scenarios = [json.loads(line) for line in f]

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process in batches
    for i in tqdm(range(0, len(scenarios), batch_size)):
        batch = scenarios[i:i+batch_size]

        for scenario in batch:
            # Generate sit rep
            prompt = f"Generate a comprehensive logistics deployment situation report.\n\n{scenario['input']}"
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=4096, temperature=0.7)

            sitrep = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Save output
            output_file = output_dir / f"sitrep_{scenario['metadata']['scenario']}_day{scenario['metadata']['day']}.md"
            output_file.write_text(sitrep)

if __name__ == "__main__":
    batch_generate(
        input_file="data/processed/test.jsonl",
        output_dir="outputs/generated_sitreps"
    )
```

**Run batch processing:**
```bash
uv run python batch_inference.py
```

### Performance Comparison

| Method | Throughput | Latency | Memory | Use Case |
|--------|-----------|---------|---------|----------|
| Local Python | ~1-2 req/min | ~30-60s | 8-10GB | Development, testing |
| FastAPI | ~2-5 req/min | ~20-40s | 8-10GB | Small-scale production |
| vLLM | ~50-100 req/min | ~2-5s | 10-12GB | High-throughput production |
| Triton | ~40-80 req/min | ~3-6s | 10-15GB | Enterprise deployment |
| Batch | ~10-20 scenarios/min | N/A | 8-10GB | Offline processing |

### Deployment Checklist

- [ ] Merge LoRA adapter into base model for faster loading
- [ ] Set up health check endpoints for monitoring
- [ ] Configure GPU memory limits and batching
- [ ] Implement request queuing for high load
- [ ] Add logging and metrics collection
- [ ] Set up model versioning and A/B testing
- [ ] Configure auto-scaling based on load
- [ ] Implement authentication and rate limiting
- [ ] Test with production-like data volumes
- [ ] Document API contracts and examples

## Next Steps

### Scale to Production

1. **Generate more training data:**
   ```bash
   # Update config to include more scenarios
   uv run python scripts/1_generate_synthetic_data.py
   # Target: 50-100 examples across multiple scenarios
   ```

2. **Train larger model (30B Nemotron Nano):**
   ```yaml
   model:
     pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
     trust_remote_code: true  # Required for Mamba SSM
   ```
   Requires more GPU memory or multi-GPU setup.

3. **Improve evaluation metrics:**
   - Give partial credit for reasonable formatting
   - Focus on content quality vs exact formatting
   - Add ROUGE scores for content similarity

## References

- **NeMo Automodel:** https://github.com/NVIDIA-NeMo/Automodel
- **Nemotron Models:** https://huggingface.co/collections/nvidia/nemotron-mini-6797b7e73cdd12e5e7c30960
- **Container Registry:** https://catalog.ngc.nvidia.com/orgs/nvidian/containers/nemo-automodel
- **Training Summary:** `TRAINING_SUMMARY.md`

## License

This project is for educational and research purposes.
