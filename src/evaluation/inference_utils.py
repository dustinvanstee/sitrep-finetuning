"""
Inference utilities for generating sit reps from models.
Handles model loading and batch generation.
"""

import os
import logging
from typing import List, Dict, Optional, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel

logger = logging.getLogger(__name__)


class SitRepGenerator:
    """Generate sit reps using a language model."""

    def __init__(
        self,
        model_path: str,
        lora_weights_path: Optional[str] = None,
        device: str = "auto",
        load_in_8bit: bool = False
    ):
        """
        Initialize the generator.

        Args:
            model_path: Path to base model or fine-tuned model
            lora_weights_path: Optional path to LoRA weights to apply
            device: Device to load model on
            load_in_8bit: Whether to load model in 8-bit mode
        """
        self.model_path = model_path
        self.lora_weights_path = lora_weights_path
        self.device = device

        logger.info(f"Loading model from {model_path}")
        self.tokenizer, self.model = self._load_model(
            model_path=model_path,
            lora_weights_path=lora_weights_path,
            device=device,
            load_in_8bit=load_in_8bit
        )

    def _load_model(
        self,
        model_path: str,
        lora_weights_path: Optional[str],
        device: str,
        load_in_8bit: bool
    ):
        """Load model and tokenizer."""
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": device,
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )

        # Apply LoRA weights if provided
        if lora_weights_path:
            logger.info(f"Loading LoRA weights from {lora_weights_path}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path
            )
            model = model.merge_and_unload()

        model.eval()
        return tokenizer, model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """
        Generate a sit rep from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        # Apply chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Format as chat messages
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # Leave room for output
        ).to(self.model.device)

        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )

        # Decode (keep special tokens initially to find boundaries)
        full_output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=False
        )

        # Extract only the generated part after the chat template
        import re

        # Look for the Assistant response marker
        assistant_marker = '<extra_id_1>Assistant\n'
        if assistant_marker in full_output:
            # Split on assistant marker and take everything after
            parts = full_output.split(assistant_marker)
            if len(parts) > 1:
                generated_text = parts[-1].strip()
            else:
                generated_text = full_output
        else:
            # Fallback: decode without special tokens and try to extract
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Remove the original prompt if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

        # Remove any trailing special tokens
        for token in ['<extra_id_', '</s>', '<eos>']:
            if token in generated_text:
                generated_text = generated_text.split(token)[0]

        return generated_text.strip()

    def generate_batch(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> List[str]:
        """
        Generate sit reps for a batch of prompts.

        Args:
            prompts: List of input prompts
            **generation_kwargs: Arguments passed to generate()

        Returns:
            List of generated texts
        """
        results = []

        for i, prompt in enumerate(prompts):
            logger.info(f"Generating {i+1}/{len(prompts)}")
            try:
                generated = self.generate(prompt, **generation_kwargs)
                results.append(generated)
            except Exception as e:
                logger.error(f"Error generating for prompt {i+1}: {e}")
                results.append("")

        return results


def load_test_data(test_file: str) -> List[Dict[str, Any]]:
    """
    Load test data from JSONL file.

    Args:
        test_file: Path to test JSONL file

    Returns:
        List of test examples
    """
    import json

    examples = []
    with open(test_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    logger.info(f"Loaded {len(examples)} test examples from {test_file}")
    return examples


def create_generators(
    baseline_model_path: str,
    finetuned_model_path: str,
    device: str = "auto"
) -> Dict[str, SitRepGenerator]:
    """
    Create generators for baseline and fine-tuned models.

    Args:
        baseline_model_path: Path to baseline model
        finetuned_model_path: Path to fine-tuned model (with LoRA)
        device: Device to load models on

    Returns:
        Dictionary with 'baseline' and 'finetuned' generators
    """
    generators = {}

    logger.info("Loading baseline model")
    generators['baseline'] = SitRepGenerator(
        model_path=baseline_model_path,
        device=device
    )

    logger.info("Loading fine-tuned model")
    # Check if finetuned_model_path is a LoRA adapter
    if os.path.exists(os.path.join(finetuned_model_path, 'adapter_config.json')):
        # It's a LoRA adapter, load on top of baseline
        generators['finetuned'] = SitRepGenerator(
            model_path=baseline_model_path,
            lora_weights_path=finetuned_model_path,
            device=device
        )
    else:
        # It's a full model
        generators['finetuned'] = SitRepGenerator(
            model_path=finetuned_model_path,
            device=device
        )

    return generators


def generate_comparison(
    test_examples: List[Dict[str, Any]],
    baseline_generator: SitRepGenerator,
    finetuned_generator: SitRepGenerator,
    generation_config: dict
) -> Dict[str, List[str]]:
    """
    Generate sit reps from both models for comparison.

    Args:
        test_examples: List of test examples with 'input' field
        baseline_generator: Baseline model generator
        finetuned_generator: Fine-tuned model generator
        generation_config: Generation parameters

    Returns:
        Dictionary with 'baseline', 'finetuned', and 'reference' lists
    """
    prompts = [ex['input'] for ex in test_examples]
    references = [ex['output'] for ex in test_examples]

    logger.info("Generating with baseline model")
    baseline_outputs = baseline_generator.generate_batch(
        prompts,
        **generation_config
    )

    logger.info("Generating with fine-tuned model")
    finetuned_outputs = finetuned_generator.generate_batch(
        prompts,
        **generation_config
    )

    return {
        'baseline': baseline_outputs,
        'finetuned': finetuned_outputs,
        'reference': references,
        'inputs': prompts
    }


def save_generations(
    outputs: Dict[str, List[str]],
    output_dir: str,
    test_examples: List[Dict[str, Any]]
):
    """
    Save generated outputs to files.

    Args:
        outputs: Dictionary with generated outputs
        output_dir: Directory to save outputs
        test_examples: Original test examples for metadata
    """
    import json

    os.makedirs(output_dir, exist_ok=True)

    # Save each model's outputs
    for model_type in ['baseline', 'finetuned', 'reference']:
        if model_type not in outputs:
            continue

        model_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)

        # Save as individual markdown files
        for i, (text, example) in enumerate(zip(outputs[model_type], test_examples)):
            filename = f"sitrep_{i+1}_{example['metadata']['scenario']}_day{example['metadata']['day']}.md"
            filepath = os.path.join(model_dir, filename)

            with open(filepath, 'w') as f:
                f.write(text)

        # Save as combined JSONL
        jsonl_path = os.path.join(output_dir, f"{model_type}_outputs.jsonl")
        with open(jsonl_path, 'w') as f:
            for i, text in enumerate(outputs[model_type]):
                entry = {
                    'output': text,
                    'metadata': test_examples[i]['metadata']
                }
                f.write(json.dumps(entry) + '\n')

        logger.info(f"Saved {model_type} outputs to {model_dir}")

    # Save combined comparison
    comparison_path = os.path.join(output_dir, 'comparison.jsonl')
    with open(comparison_path, 'w') as f:
        for i in range(len(test_examples)):
            entry = {
                'input': outputs['inputs'][i],
                'baseline': outputs['baseline'][i],
                'finetuned': outputs['finetuned'][i],
                'reference': outputs['reference'][i],
                'metadata': test_examples[i]['metadata']
            }
            f.write(json.dumps(entry) + '\n')

    logger.info(f"Saved comparison to {comparison_path}")
