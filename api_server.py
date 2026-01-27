#!/usr/bin/env python3
"""
FastAPI server for logistics sit rep generation.

Usage:
    python api_server.py
    # Server runs on http://localhost:8000
    # API docs at http://localhost:8000/docs

Requirements:
    pip install fastapi uvicorn[standard] torch transformers peft accelerate
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Logistics Sit Rep Generator",
    description="Generate logistics deployment situation reports using fine-tuned Nemotron-Mini-4B",
    version="1.0.0"
)

# Global model cache
MODEL_CACHE = {}

class LogisticsRequest(BaseModel):
    logistics_data: str = Field(
        ...,
        description="Logistics data in markdown format with location details",
        example="## Logistics Data\n\n### Location: Alpha Base (ID: LOC-001)\n- Coordinates: 31.59°N, 98.35°W\n..."
    )
    max_tokens: int = Field(
        default=4096,
        ge=512,
        le=8192,
        description="Maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (higher = more creative)"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold"
    )

class SitRepResponse(BaseModel):
    sitrep: str = Field(..., description="Generated situation report in markdown format")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    model_version: str = Field(..., description="Model version used")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

@app.on_event("startup")
async def load_models():
    """Load model on startup."""
    print("="*60)
    print("Loading fine-tuned model...")
    print("="*60)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "nvidia/Nemotron-Mini-4B-Instruct",
            trust_remote_code=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            "nvidia/Nemotron-Mini-4B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
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
        MODEL_CACHE["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"✓ Model loaded successfully on {MODEL_CACHE['device']}!")
        print("="*60)

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

@app.post("/generate", response_model=SitRepResponse)
async def generate_sitrep(request: LogisticsRequest):
    """
    Generate a logistics situation report.

    **Example Request:**
    ```json
    {
        "logistics_data": "## Logistics Data\\n\\n### Location: Alpha Base (ID: LOC-001)\\n- Coordinates: 31.59°N, 98.35°W\\n- Terrain: Coastal Plain at 683m elevation\\n- Weather: Clear, 22°C\\n- Distance from DC: 242 km\\n- Population Served: 159,856\\n- Current Demand: 4,033 units/day\\n- Base Deployment Cost: $111,050",
        "max_tokens": 4096,
        "temperature": 0.7
    }
    ```

    **Example Response:**
    ```json
    {
        "sitrep": "# Logistics Deployment Situation Report\\n\\n## TLDR\\n...",
        "tokens_generated": 1523,
        "model_version": "nemotron-mini-4b-sitrep-v1"
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
            tokens_generated=tokens_generated,
            model_version="nemotron-mini-4b-sitrep-v1"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded="model" in MODEL_CACHE,
        device=MODEL_CACHE.get("device", "unknown")
    )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Logistics Sit Rep Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/generate (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        },
        "model": "nvidia/Nemotron-Mini-4B-Instruct (fine-tuned)"
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Logistics Sit Rep Generator API")
    print("="*60)
    print("Server: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
