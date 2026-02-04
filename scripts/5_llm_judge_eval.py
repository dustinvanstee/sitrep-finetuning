#!/usr/bin/env python3
"""
Script 5: Standalone LLM-as-a-Judge Evaluation

Evaluates previously generated SITREP outputs using LLM judge.
Runs OUTSIDE docker container (no GPU needed, just API access).

Prerequisites:
    - Run generation first (via docker): ./scripts/run_evaluation_4gpu.sh --no-llm-judge
    - Outputs exist in: outputs/evaluation_results_v2/

Usage:
    # Activate venv and run
    source .venv/bin/activate
    
    # Basic evaluation (uses existing outputs)
    python scripts/5_llm_judge_eval.py
    
    # Limit samples
    python scripts/5_llm_judge_eval.py --num-samples 5
    
    # Use different judge model
    python scripts/5_llm_judge_eval.py --judge-model deepseek-ai/deepseek-v3.2
    
    # Skip cloud baseline evaluation
    python scripts/5_llm_judge_eval.py --no-cloud-baseline
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Local imports
from src.evaluation.llm_judge import (
    LLMJudge,
    JudgeConfig,
    compute_judge_statistics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "output_dir": "outputs/evaluation_results_v2",
    "judge": {
        "provider": "nvidia",
        "model": "moonshotai/kimi-k2-thinking",
        "temperature": 0.3,
        "max_tokens": 4096,
    }
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_outputs(output_dir: str, num_samples: Optional[int] = None) -> Dict[str, List[str]]:
    """Load previously generated outputs from JSONL files."""
    outputs = {}
    
    # Load baseline outputs
    baseline_path = os.path.join(output_dir, 'baseline_outputs.jsonl')
    if os.path.exists(baseline_path):
        outputs['baseline'] = []
        with open(baseline_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                outputs['baseline'].append(data.get('output', ''))
        logger.info(f"Loaded {len(outputs['baseline'])} baseline outputs")
    else:
        raise FileNotFoundError(f"Baseline outputs not found: {baseline_path}")
    
    # Load fine-tuned outputs
    finetuned_path = os.path.join(output_dir, 'finetuned_outputs.jsonl')
    if os.path.exists(finetuned_path):
        outputs['finetuned'] = []
        with open(finetuned_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                outputs['finetuned'].append(data.get('output', ''))
        logger.info(f"Loaded {len(outputs['finetuned'])} fine-tuned outputs")
    else:
        raise FileNotFoundError(f"Fine-tuned outputs not found: {finetuned_path}")
    
    # Load cloud baseline outputs (optional)
    cloud_path = os.path.join(output_dir, 'cloud_baseline_outputs.jsonl')
    if os.path.exists(cloud_path):
        outputs['cloud_baseline'] = []
        with open(cloud_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                outputs['cloud_baseline'].append(data.get('output', ''))
        logger.info(f"Loaded {len(outputs['cloud_baseline'])} cloud baseline outputs")
    
    # Load comparison data for inputs and references
    comparison_path = os.path.join(output_dir, 'comparison.jsonl')
    if os.path.exists(comparison_path):
        outputs['inputs'] = []
        outputs['reference'] = []
        with open(comparison_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                outputs['inputs'].append(data.get('input', ''))
                outputs['reference'].append(data.get('reference', ''))
        logger.info(f"Loaded {len(outputs['inputs'])} inputs and references")
    
    # Limit samples if requested
    if num_samples:
        for key in outputs:
            outputs[key] = outputs[key][:num_samples]
        logger.info(f"Limited to {num_samples} samples")
    
    return outputs


# =============================================================================
# EVALUATION
# =============================================================================

def run_llm_judge(
    judge: LLMJudge,
    sitreps: List[str],
    input_intels: Optional[List[str]] = None,
    reference_sitreps: Optional[List[str]] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """Run LLM judge evaluation on a set of SITREPs."""
    logger.info(f"Evaluating {model_name} outputs ({len(sitreps)} samples)...")
    
    evaluations = judge.evaluate_batch(sitreps, input_intels, reference_sitreps)
    stats = compute_judge_statistics(evaluations)
    
    # Convert evaluations to serializable format
    eval_dicts = []
    for e in evaluations:
        if hasattr(e, 'model_dump'):
            eval_dicts.append(e.model_dump())
        elif hasattr(e, '__dict__'):
            eval_dicts.append(vars(e))
        else:
            eval_dicts.append(str(e))
    
    return {
        'stats': stats,
        'evaluations': eval_dicts
    }


def compute_improvements(baseline_stats: Dict, finetuned_stats: Dict, cloud_stats: Optional[Dict] = None) -> Dict:
    """Compute improvement metrics."""
    improvement = {}
    
    if baseline_stats.get("overall") and finetuned_stats.get("overall"):
        baseline_mean = baseline_stats["overall"]["mean"]
        finetuned_mean = finetuned_stats["overall"]["mean"]
        
        improvement["overall_improvement"] = finetuned_mean - baseline_mean
        improvement["overall_improvement_pct"] = (
            (finetuned_mean - baseline_mean) / baseline_mean * 100
            if baseline_mean > 0 else 0
        )
        
        # Per-dimension improvements
        for dim in ["section_completeness", "factual_accuracy", "analytical_quality",
                    "formatting_consistency", "actionability"]:
            if dim in baseline_stats and dim in finetuned_stats:
                improvement[f"{dim}_improvement"] = (
                    finetuned_stats[dim]["mean"] - baseline_stats[dim]["mean"]
                )
    
    # Cloud comparison
    if cloud_stats and cloud_stats.get("overall") and finetuned_stats.get("overall"):
        cloud_mean = cloud_stats["overall"]["mean"]
        finetuned_mean = finetuned_stats["overall"]["mean"]
        if cloud_mean > 0:
            improvement['vs_cloud_gap'] = cloud_mean - finetuned_mean
            improvement['vs_cloud_pct'] = (finetuned_mean / cloud_mean) * 100
    
    return improvement


# =============================================================================
# OUTPUT
# =============================================================================

def print_summary(results: Dict, cloud_name: str = "Nemotron-Super-49B"):
    """Print evaluation summary to console."""
    print("\n" + "=" * 70)
    print("LLM-AS-A-JUDGE EVALUATION RESULTS")
    print("=" * 70)
    
    baseline_stats = results.get('baseline', {}).get('stats', {})
    finetuned_stats = results.get('finetuned', {}).get('stats', {})
    cloud_stats = results.get('cloud_baseline', {}).get('stats', {})
    improvement = results.get('improvement', {})
    
    has_cloud = bool(cloud_stats and cloud_stats.get('overall'))
    
    print(f"\nJudge Model: {results.get('judge_model', 'unknown')}")
    print(f"Samples Evaluated: {results.get('num_samples', 'unknown')}")
    
    print("\n" + "-" * 50)
    print("OVERALL SCORES")
    print("-" * 50)
    
    if baseline_stats.get('overall'):
        print(f"  Baseline (4B):       {baseline_stats['overall']['mean']:>6.1f} / 100")
    if finetuned_stats.get('overall'):
        print(f"  Fine-tuned (4B+LoRA): {finetuned_stats['overall']['mean']:>6.1f} / 100")
    if has_cloud:
        print(f"  {cloud_name}: {cloud_stats['overall']['mean']:>6.1f} / 100")
    
    if improvement:
        print(f"\n  Improvement vs Baseline: {improvement.get('overall_improvement', 0):>+6.1f} ({improvement.get('overall_improvement_pct', 0):+.1f}%)")
        if has_cloud and 'vs_cloud_pct' in improvement:
            print(f"  Quality vs Cloud:        {improvement['vs_cloud_pct']:>6.0f}% of {cloud_name}")
            print(f"  Gap to Cloud:            {improvement['vs_cloud_gap']:>6.1f} points")
    
    print("\n" + "-" * 50)
    print("DIMENSION SCORES (Fine-tuned)")
    print("-" * 50)
    
    if finetuned_stats:
        for dim in ['section_completeness', 'factual_accuracy', 'analytical_quality',
                    'formatting_consistency', 'actionability']:
            score = finetuned_stats.get(dim, {}).get('mean', 0)
            baseline_score = baseline_stats.get(dim, {}).get('mean', 0)
            diff = score - baseline_score
            print(f"  {dim.replace('_', ' ').title():<25} {score:.1f}/5  ({diff:+.1f} vs baseline)")
    
    # Show common issues/strengths if available
    if finetuned_stats.get('common_issues'):
        print("\n" + "-" * 50)
        print("COMMON ISSUES (Fine-tuned)")
        print("-" * 50)
        for issue, count in list(finetuned_stats['common_issues'].items())[:5]:
            print(f"  - {issue} ({count}x)")
    
    if finetuned_stats.get('common_strengths'):
        print("\n" + "-" * 50)
        print("COMMON STRENGTHS (Fine-tuned)")
        print("-" * 50)
        for strength, count in list(finetuned_stats['common_strengths'].items())[:5]:
            print(f"  - {strength} ({count}x)")
    
    print("\n" + "=" * 70)


def save_results(results: Dict, output_dir: str):
    """Save evaluation results to timestamped files in llm_judge/ subdirectory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create llm_judge subdirectory
    llm_judge_dir = os.path.join(output_dir, 'llm_judge')
    os.makedirs(llm_judge_dir, exist_ok=True)
    
    # Save summary JSON (with timestamp)
    summary_path = os.path.join(llm_judge_dir, f'results_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'judge_model': results.get('judge_model'),
            'num_samples': results.get('num_samples'),
            'baseline_stats': results.get('baseline', {}).get('stats'),
            'finetuned_stats': results.get('finetuned', {}).get('stats'),
            'cloud_baseline_stats': results.get('cloud_baseline', {}).get('stats'),
            'improvement': results.get('improvement'),
        }, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")
    
    # Save detailed evaluations (with timestamp)
    detailed_path = os.path.join(llm_judge_dir, f'detailed_{timestamp}.json')
    with open(detailed_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'judge_model': results.get('judge_model'),
            'num_samples': results.get('num_samples'),
            'baseline_evaluations': results.get('baseline', {}).get('evaluations', []),
            'finetuned_evaluations': results.get('finetuned', {}).get('evaluations', []),
            'cloud_baseline_evaluations': results.get('cloud_baseline', {}).get('evaluations', []),
        }, f, indent=2)
    logger.info(f"Saved detailed evaluations to {detailed_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone LLM-as-a-Judge Evaluation (runs outside docker)"
    )
    
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation_results_v2',
                        help='Directory with generated outputs')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Limit number of samples to evaluate')
    
    # Judge configuration
    parser.add_argument('--judge-provider', type=str, default='nvidia',
                        choices=['nvidia', 'astra', 'openai', 'local-nim'],
                        help='LLM provider for judge (astra uses inference-api.nvidia.com)')
    parser.add_argument('--judge-model', type=str, default='moonshotai/kimi-k2-thinking',
                        help='Model name for judge')
    parser.add_argument('--judge-temperature', type=float, default=0.3,
                        help='Temperature for judge model')
    
    # Options
    parser.add_argument('--no-cloud-baseline', action='store_true',
                        help='Skip evaluating cloud baseline outputs')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Only evaluate baseline model')
    parser.add_argument('--finetuned-only', action='store_true',
                        help='Only evaluate fine-tuned model')
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Check for API key based on provider
    if args.judge_provider == 'astra':
        api_key = os.environ.get("ASTRA_API_KEY")
        if not api_key:
            logger.error("ASTRA_API_KEY not set. Required for astra provider.")
            sys.exit(1)
    else:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            logger.error("NVIDIA_API_KEY not set. Please set it in .env or environment.")
            sys.exit(1)
    
    logger.info(f"Using API key: {api_key[:12]}...{api_key[-4:]}")
    
    # Load outputs
    try:
        outputs = load_outputs(args.output_dir, args.num_samples)
    except FileNotFoundError as e:
        logger.error(f"Output files not found: {e}")
        logger.info("Run generation first: ./scripts/run_evaluation_4gpu.sh --no-llm-judge")
        sys.exit(1)
    
    # Initialize judge
    judge_config = JudgeConfig(
        provider=args.judge_provider,
        model=args.judge_model,
        temperature=args.judge_temperature,
    )
    judge = LLMJudge(judge_config)
    
    logger.info(f"Initialized judge: {args.judge_model}")
    
    # Results container
    results = {
        'judge_model': args.judge_model,
        'num_samples': len(outputs['baseline']),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Evaluate baseline
    if not args.finetuned_only:
        results['baseline'] = run_llm_judge(
            judge,
            outputs['baseline'],
            outputs.get('inputs'),
            outputs.get('reference'),
            "Baseline (4B)"
        )
    
    # Evaluate fine-tuned
    if not args.baseline_only:
        results['finetuned'] = run_llm_judge(
            judge,
            outputs['finetuned'],
            outputs.get('inputs'),
            outputs.get('reference'),
            "Fine-tuned (4B+LoRA)"
        )
    
    # Evaluate cloud baseline
    if not args.no_cloud_baseline and 'cloud_baseline' in outputs and not args.baseline_only and not args.finetuned_only:
        results['cloud_baseline'] = run_llm_judge(
            judge,
            outputs['cloud_baseline'],
            outputs.get('inputs'),
            outputs.get('reference'),
            "Cloud Baseline (49B)"
        )
    
    # Compute improvements
    if 'baseline' in results and 'finetuned' in results:
        results['improvement'] = compute_improvements(
            results['baseline']['stats'],
            results['finetuned']['stats'],
            results.get('cloud_baseline', {}).get('stats')
        )
    
    # Print and save results
    print_summary(results)
    save_results(results, args.output_dir)
    
    logger.info("LLM Judge evaluation complete!")


if __name__ == '__main__':
    main()
