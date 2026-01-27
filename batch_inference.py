#!/usr/bin/env python3
"""
Batch process logistics data to generate multiple sit reps.

Usage:
    python batch_inference.py --input data/processed/test.jsonl --output outputs/generated_sitreps
    python batch_inference.py --input scenarios.jsonl --output results/ --batch-size 4
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(
    base_model: str = "nvidia/Nemotron-Mini-4B-Instruct",
    adapter_path: str = "outputs/models/finetuned-4b",
    device: str = "cuda"
):
    """Load model with LoRA adapter."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model_obj, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    return tokenizer, model

def generate_sitrep(tokenizer, model, logistics_data: str, max_tokens: int = 4096, temperature: float = 0.7):
    """Generate a single sit rep."""
    prompt = f"Generate a comprehensive logistics deployment situation report.\n\n{logistics_data}"

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract assistant response
    assistant_marker = '<extra_id_1>Assistant\n'
    if assistant_marker in full_output:
        generated_text = full_output.split(assistant_marker)[-1]
        for token in ['<extra_id_', '</s>', '<eos>']:
            if token in generated_text:
                generated_text = generated_text.split(token)[0]
        return generated_text.strip()

    return full_output

def batch_generate(
    input_file: str,
    output_dir: str,
    base_model: str = "nvidia/Nemotron-Mini-4B-Instruct",
    adapter_path: str = "outputs/models/finetuned-4b",
    device: str = "cuda",
    max_tokens: int = 4096,
    temperature: float = 0.7
):
    """Process multiple logistics scenarios."""

    # Load model
    tokenizer, model = load_model(base_model, adapter_path, device)

    # Load input data
    print(f"Loading scenarios from {input_file}")
    with open(input_file, 'r') as f:
        scenarios = [json.loads(line) for line in f]

    print(f"Found {len(scenarios)} scenarios to process")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process scenarios
    results = []
    for scenario in tqdm(scenarios, desc="Generating sit reps"):
        try:
            # Generate sit rep
            sitrep = generate_sitrep(
                tokenizer,
                model,
                scenario['input'],
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Save output
            metadata = scenario.get('metadata', {})
            scenario_name = metadata.get('scenario', 'unknown')
            day = metadata.get('day', 'unknown')
            output_file = output_dir / f"sitrep_{scenario_name}_day{day}.md"

            output_file.write_text(sitrep)

            results.append({
                'scenario': scenario_name,
                'day': day,
                'output_file': str(output_file),
                'status': 'success'
            })

        except Exception as e:
            print(f"Error processing scenario: {e}")
            results.append({
                'scenario': metadata.get('scenario', 'unknown'),
                'day': metadata.get('day', 'unknown'),
                'error': str(e),
                'status': 'failed'
            })

    # Save results summary
    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_scenarios': len(scenarios),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'results': results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("Batch Generation Complete")
    print(f"{'='*60}")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_file}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Batch generate logistics sit reps")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with logistics scenarios"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for generated sit reps"
    )
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
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate per sit rep"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    batch_generate(
        input_file=args.input,
        output_dir=args.output,
        base_model=args.base_model,
        adapter_path=args.adapter,
        device=args.device,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()
