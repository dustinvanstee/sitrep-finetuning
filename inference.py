#!/usr/bin/env python3
"""
Simple inference script for the fine-tuned logistics sit rep model.

Usage:
    python inference.py
    python inference.py --adapter outputs/models/finetuned-4b --device cuda
"""
import argparse
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
    print("Generating sit rep...")
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

def main():
    parser = argparse.ArgumentParser(description="Generate logistics sit reps")
    parser.add_argument(
        "--base-model",
        default="nvidia/Nemotron-Mini-4B-Instruct",
        help="Base model path"
    )
    parser.add_argument(
        "--adapter",
        default="outputs/models/finetuned-4b",
        help="LoRA adapter path"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    args = parser.parse_args()

    # Load model
    tokenizer, model = load_model(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        device=args.device
    )

    # Example logistics data
    logistics_data = """## Logistics Data

### Location: Alpha Base (ID: LOC-001)
- Coordinates: 31.59°N, 98.35°W
- Terrain: Coastal Plain at 683m elevation
- Weather: Clear, 22°C
- Wind Speed: 10.0 km/h
- Distance from DC: 242.2 km
- Road Quality: Excellent
- Population Served: 159,856
- Current Demand: 4,033 units/day
- Demand Growth: 11.7%
- Competitors: 6
- Base Deployment Cost: $111,050
- Transport Cost per Unit: $29.22
- Labor Cost: $16.30/hour
- Political Stability: 5.3/10
- Disaster Risk: 3.5/10
- Supply Chain Risk: 7.2/10
- Infrastructure: Airport=False, Seaport=False, Rail=True

### Location: Bravo Station (ID: LOC-002)
- Coordinates: 31.50°N, 95.73°W
- Terrain: Beach at 110m elevation
- Weather: Rain, 18°C
- Wind Speed: 15.0 km/h
- Distance from DC: 324.3 km
- Road Quality: Poor
- Population Served: 86,383
- Current Demand: 4,157 units/day
- Demand Growth: 6.1%
- Competitors: 4
- Base Deployment Cost: $103,530
- Transport Cost per Unit: $37.43
- Labor Cost: $31.06/hour
- Political Stability: 6.0/10
- Disaster Risk: 6.9/10
- Supply Chain Risk: 7.8/10
- Infrastructure: Airport=False, Seaport=False, Rail=True

### Location: Charlie Point (ID: LOC-003)
- Coordinates: 31.60°N, 97.20°W
- Terrain: Low Hills at 122m elevation
- Weather: Partly Cloudy, 25°C
- Wind Speed: 5.0 km/h
- Distance from DC: 315.9 km
- Road Quality: Fair
- Population Served: 282,988
- Current Demand: 5,963 units/day
- Demand Growth: 17.1%
- Competitors: 5
- Base Deployment Cost: $107,810
- Transport Cost per Unit: $36.59
- Labor Cost: $25.83/hour
- Political Stability: 7.5/10
- Disaster Risk: 3.4/10
- Supply Chain Risk: 4.0/10
- Infrastructure: Airport=True, Seaport=False, Rail=True
"""

    # Generate sit rep
    sitrep = generate_sitrep(
        tokenizer,
        model,
        logistics_data,
        temperature=args.temperature
    )

    print("\n" + "="*60)
    print("GENERATED SIT REP")
    print("="*60)
    print(sitrep)
    print("="*60)

if __name__ == "__main__":
    main()
