---
name: ndd-docs
description: Fetch and explain NeMo Data Designer documentation from NVIDIA. Use when the user asks about Data Designer columns, samplers, LLM text generation, structured output, expressions, validation, seed datasets, or synthetic data generation workflows. Also review DataDesigner directory that has actual working code examples.
---

# NeMo Data Designer Documentation

## Overview

NeMo Data Designer is NVIDIA's framework for generating high-quality synthetic data. It uses a declarative approach where you specify columns and the framework handles execution order, batching, and parallelization.

**Documentation**: https://nvidia-nemo.github.io/DataDesigner/latest/
**NOTE** : The documentation does have some bugs, so also refer to DataDesigner directory for actual working examples!!!

## Quick Start Pattern

```python
from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SamplerType,
)

# Initialize
data_designer = DataDesigner()
config_builder = DataDesignerConfigBuilder()

# Add columns
config_builder.add_column(SamplerColumnConfig(...))
config_builder.add_column(LLMTextColumnConfig(...))

# Preview and iterate
preview_results = data_designer.preview(config_builder=config_builder)
preview_results.display_sample_record()
```

## Column Types

### 🎲 Sampler Columns (`SamplerColumnConfig`)

Fast, deterministic data generation using statistical distributions.

**Sampler Types** (`SamplerType` enum):
- `UUID` - Unique identifiers
- `CATEGORY` - Categorical values with optional weights
- `SUBCATEGORY` - Hierarchical categories (state within country)
- `UNIFORM` - Even distribution (int or float)
- `GAUSSIAN` - Normal distribution (mean, std)
- `BERNOULLI` - Binary outcomes
- `BERNOULLI_MIXTURE` - Multi-component binary
- `BINOMIAL` - Success counts in trials
- `POISSON` - Event counts
- `SCIPY` - Any scipy.stats distribution
- `PERSON` - Synthetic individuals (name, demographics)
- `DATETIME` - Timestamps in range
- `TIMEDELTA` - Duration values

**Example**:
```python
SamplerColumnConfig(
    name="language",
    sampler_type=SamplerType.CATEGORY,
    params=CategorySamplerParams(
        values=["English", "Spanish", "French"],
        weights=[0.5, 0.3, 0.2]  # optional
    ),
)
```

**Conditional Sampling** - Change behavior based on other columns:
```python
SamplerColumnConfig(
    name="income",
    sampler_type=SamplerType.GAUSSIAN,
    params=GaussianSamplerParams(mean=50000, std=15000),
    conditional_params={
        "occupation == 'Engineer'": GaussianSamplerParams(mean=95000, std=20000),
        "occupation == 'Manager'": GaussianSamplerParams(mean=120000, std=30000),
    }
)
```

### 📝 LLM-Text Columns (`LLMTextColumnConfig`)

Generate natural language using LLMs with Jinja2 templating.

**Key Fields**:
- `name` - Column name
- `model_alias` - Model to use (e.g., `"nvidia-text"`)
- `prompt` - Jinja2 template referencing other columns
- `system_prompt` - Optional system instructions

**Example**:
```python
LLMTextColumnConfig(
    name="review",
    model_alias="nvidia-text",
    system_prompt="You are a product reviewer.",
    prompt="""Write a {{ sentiment }} review for {{ product_name }}.
Rating: {{ rating }}/5 stars.""",
)
```

**Reasoning Traces**: Models with extended thinking create `{column_name}__reasoning_trace` automatically.

### 💻 LLM-Code Columns (`LLMCodeColumnConfig`)

Generate code with automatic extraction from markdown blocks.

**Supported Languages**: `python`, `javascript`, `typescript`, `java`, `kotlin`, `go`, `rust`, `ruby`, `scala`, `swift`

**SQL Dialects**: `sql:sqlite`, `sql:postgres`, `sql:mysql`, `sql:tsql`, `sql:bigquery`, `sql:ansi`

```python
LLMCodeColumnConfig(
    name="solution",
    model_alias="nvidia-text",
    code_lang="python",
    prompt="Write a Python function that {{ task_description }}",
)
```

### 🗂️ LLM-Structured Columns (`LLMStructuredColumnConfig`)

Generate JSON with guaranteed schema using Pydantic models.

```python
from pydantic import BaseModel

class ProductSpec(BaseModel):
    name: str
    price: float
    features: list[str]
    in_stock: bool

LLMStructuredColumnConfig(
    name="product_info",
    model_alias="nvidia-text",
    output_format=ProductSpec,
    prompt="Generate product specifications for {{ category }}",
)
```

### ⚖️ LLM-Judge Columns (`LLMJudgeColumnConfig`)

Score content across quality dimensions.

```python
from data_designer.essentials import Score

LLMJudgeColumnConfig(
    name="quality_score",
    model_alias="nvidia-reasoning",
    prompt="Evaluate the following response: {{ response }}",
    scores=[
        Score(
            name="accuracy",
            rubric="How factually accurate is the response?",
            options={1: "Incorrect", 3: "Partially correct", 5: "Fully accurate"}
        ),
        Score(
            name="clarity",
            rubric="How clear and well-structured is the response?",
            options={1: "Confusing", 3: "Adequate", 5: "Crystal clear"}
        ),
    ]
)
```

### 🧬 Embedding Columns (`EmbeddingColumnConfig`)

Generate vector embeddings for text.

```python
EmbeddingColumnConfig(
    name="doc_embedding",
    target_column="document_text",
    model_alias="nvidia-embedding",
)
```

### 🧩 Expression Columns (`ExpressionColumnConfig`)

Transform data with Jinja2 templates (no LLM).

```python
ExpressionColumnConfig(
    name="full_name",
    expression="{{ first_name }} {{ last_name }}",
)

ExpressionColumnConfig(
    name="total_price",
    expression="{{ quantity * unit_price }}",
    dtype="float",  # int, float, str, bool
)
```

### 🔍 Validation Columns (`ValidationColumnConfig`)

Check generated content against rules.

**Validator Types**:
- `code` - Python/SQL linting
- `local_callable` - Custom Python function
- `remote` - HTTP endpoint

```python
ValidationColumnConfig(
    name="code_validation",
    target_columns=["solution"],
    validator_type="code",
    params=CodeValidatorParams(code_lang="python"),
)
```

### 🌱 Seed Dataset Columns (`SeedDatasetColumnConfig`)

Bootstrap from existing data. Use `config_builder.add_seed_data()`:

```python
import pandas as pd

seed_df = pd.DataFrame({
    "product_name": ["Widget A", "Gadget B"],
    "category": ["Electronics", "Home"]
})

config_builder.add_seed_data(seed_df)
# Now "product_name" and "category" are available for Jinja2 references
```

## Default Model Aliases

Set API keys: `NVIDIA_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`

| Alias | Use Case |
|-------|----------|
| `nvidia-text` | General text generation |
| `nvidia-reasoning` | Analysis and reasoning |
| `nvidia-vision` | Image understanding |
| `nvidia-embedding` | Text embeddings |
| `openai-text` | General text (GPT) |
| `openai-reasoning` | Reasoning (GPT) |

## Common Properties

All columns inherit:
- `name` - Unique identifier
- `drop` - If `True`, generate but exclude from output (default: `False`)
- `required_columns` - Auto-computed dependencies from Jinja2 templates

## Workflow Pattern

1. **Configure** model providers
2. **Design** columns iteratively
3. **Preview** with `data_designer.preview()`
4. **Create** full dataset with `data_designer.create()`

## Additional Resources

For detailed API reference, fetch:
- Columns: https://nvidia-nemo.github.io/DataDesigner/latest/concepts/columns/
- Models: https://nvidia-nemo.github.io/DataDesigner/latest/concepts/models/default-model-settings/
- Column configs: https://nvidia-nemo.github.io/DataDesigner/latest/code_reference/column_configs/
