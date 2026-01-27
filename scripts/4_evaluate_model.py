#!/usr/bin/env python3
"""
Script 4: Evaluate Model

Evaluates the fine-tuned model against baseline on formatting metrics.
Generates sit reps and compares quality.

Usage:
    python scripts/4_evaluate_model.py --compare --output outputs/evaluation_results/
    python scripts/4_evaluate_model.py --baseline-only
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from src.evaluation.inference_utils import (
    load_test_data,
    create_generators,
    generate_comparison,
    save_generations
)
from src.evaluation.formatting_metrics import (
    SitRepFormattingEvaluator,
    evaluate_batch,
    compare_models
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load evaluation configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_comparison_visualizations(
    comparison_stats: dict,
    output_dir: str
):
    """
    Create visualization plots comparing models.

    Args:
        comparison_stats: Comparison statistics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # 1. Overall score comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['Baseline', 'Fine-tuned']
    scores = [
        comparison_stats['baseline']['mean_total'],
        comparison_stats['finetuned']['mean_total']
    ]

    bars = ax.bar(models, scores, color=['#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Average Formatting Score (out of 100)', fontsize=12)
    ax.set_title('Model Comparison: Overall Formatting Quality', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add improvement annotation
    improvement = comparison_stats['improvement']['total_score_improvement']
    improvement_pct = comparison_stats['improvement']['total_score_improvement_pct']
    ax.text(0.5, max(scores) + 5,
            f'Improvement: +{improvement:.1f} points ({improvement_pct:.1f}%)',
            ha='center', fontsize=11, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300)
    plt.close()

    # 2. Category breakdown
    fig, ax = plt.subplots(figsize=(12, 6))
    categories = ['Section\nCompleteness', 'TLDR\nQuality', 'Markdown\nFormatting', 'Structure\nQuality']
    baseline_vals = [
        comparison_stats['baseline']['mean_section_completeness'],
        comparison_stats['baseline']['mean_tldr_quality'],
        comparison_stats['baseline']['mean_markdown_formatting'],
        comparison_stats['baseline']['mean_structure_quality']
    ]
    finetuned_vals = [
        comparison_stats['finetuned']['mean_section_completeness'],
        comparison_stats['finetuned']['mean_tldr_quality'],
        comparison_stats['finetuned']['mean_markdown_formatting'],
        comparison_stats['finetuned']['mean_structure_quality']
    ]

    x = range(len(categories))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], baseline_vals, width, label='Baseline', color='#ff7f0e')
    bars2 = ax.bar([i + width/2 for i in x], finetuned_vals, width, label='Fine-tuned', color='#2ca02c')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison: Score Breakdown by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim([0, max(max(baseline_vals), max(finetuned_vals)) * 1.2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_breakdown.png'), dpi=300)
    plt.close()

    # 3. Improvement heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    improvements = pd.DataFrame({
        'Category': categories,
        'Improvement': [
            comparison_stats['improvement']['section_completeness_improvement'],
            comparison_stats['improvement']['tldr_quality_improvement'],
            comparison_stats['improvement']['markdown_formatting_improvement'],
            comparison_stats['improvement']['structure_quality_improvement']
        ]
    })

    colors = ['green' if x > 0 else 'red' for x in improvements['Improvement']]
    bars = ax.barh(improvements['Category'], improvements['Improvement'], color=colors)

    ax.set_xlabel('Improvement (points)', fontsize=12)
    ax.set_title('Fine-tuning Impact by Category', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements['Improvement'])):
        ax.text(val + 0.5 if val > 0 else val - 0.5, i,
                f'{val:+.1f}',
                va='center', ha='left' if val > 0 else 'right',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_heatmap.png'), dpi=300)
    plt.close()

    logger.info(f"Saved visualizations to {output_dir}")


def generate_markdown_report(
    comparison_stats: dict,
    test_examples: List[dict],
    outputs: Dict[str, List[str]],
    output_path: str
):
    """
    Generate comprehensive markdown evaluation report.

    Args:
        comparison_stats: Comparison statistics
        test_examples: Test examples
        outputs: Generated outputs
        output_path: Path to save report
    """
    report = f"""# Logistics Sit Rep Fine-Tuning Evaluation Report

## Executive Summary

This report presents the evaluation results of fine-tuning Nemotron Nano 3 on logistics situation report generation.

### Key Findings

- **Overall Improvement**: {comparison_stats['improvement']['total_score_improvement']:+.1f} points ({comparison_stats['improvement']['total_score_improvement_pct']:+.1f}%)
- **Baseline Score**: {comparison_stats['baseline']['mean_total']:.1f}/100
- **Fine-tuned Score**: {comparison_stats['finetuned']['mean_total']:.1f}/100
- **Test Set Size**: {len(test_examples)} sit reps

### Performance by Category

| Category | Baseline | Fine-tuned | Improvement |
|----------|----------|------------|-------------|
| Section Completeness | {comparison_stats['baseline']['mean_section_completeness']:.1f}/40 | {comparison_stats['finetuned']['mean_section_completeness']:.1f}/40 | {comparison_stats['improvement']['section_completeness_improvement']:+.1f} |
| TLDR Quality | {comparison_stats['baseline']['mean_tldr_quality']:.1f}/20 | {comparison_stats['finetuned']['mean_tldr_quality']:.1f}/20 | {comparison_stats['improvement']['tldr_quality_improvement']:+.1f} |
| Markdown Formatting | {comparison_stats['baseline']['mean_markdown_formatting']:.1f}/20 | {comparison_stats['finetuned']['mean_markdown_formatting']:.1f}/20 | {comparison_stats['improvement']['markdown_formatting_improvement']:+.1f} |
| Structure Quality | {comparison_stats['baseline']['mean_structure_quality']:.1f}/20 | {comparison_stats['finetuned']['mean_structure_quality']:.1f}/20 | {comparison_stats['improvement']['structure_quality_improvement']:+.1f} |

## Detailed Analysis

### Success Criteria Assessment

**Minimum Viable Product (Target: >60/100 with >10% improvement)**
- Total Score: {comparison_stats['finetuned']['mean_total']:.1f}/100 ({'✓ PASS' if comparison_stats['finetuned']['mean_total'] > 60 else '✗ FAIL'})
- Improvement: {comparison_stats['improvement']['total_score_improvement_pct']:+.1f}% ({'✓ PASS' if comparison_stats['improvement']['total_score_improvement_pct'] > 10 else '✗ FAIL'})

**Production Ready (Target: >80/100 with >20% improvement)**
- Total Score: {comparison_stats['finetuned']['mean_total']:.1f}/100 ({'✓ PASS' if comparison_stats['finetuned']['mean_total'] > 80 else '✗ FAIL'})
- Improvement: {comparison_stats['improvement']['total_score_improvement_pct']:+.1f}% ({'✓ PASS' if comparison_stats['improvement']['total_score_improvement_pct'] > 20 else '✗ FAIL'})

### Score Distribution

**Baseline Model:**
- Mean: {comparison_stats['baseline']['mean_total']:.1f}
- Range: {comparison_stats['baseline']['min_total']:.1f} - {comparison_stats['baseline']['max_total']:.1f}

**Fine-tuned Model:**
- Mean: {comparison_stats['finetuned']['mean_total']:.1f}
- Range: {comparison_stats['finetuned']['min_total']:.1f} - {comparison_stats['finetuned']['max_total']:.1f}

## Visualizations

Generated visualizations:
- `overall_comparison.png`: Overall score comparison
- `category_breakdown.png`: Detailed breakdown by category
- `improvement_heatmap.png`: Improvement analysis

## Output Files

- `baseline/`: Baseline model outputs
- `finetuned/`: Fine-tuned model outputs
- `reference/`: Reference (ground truth) outputs
- `comparison.jsonl`: Side-by-side comparison data

## Methodology

### Evaluation Metrics

The evaluation uses a 100-point scoring system across four categories:

1. **Section Completeness (40 points)**: Presence of required sections (TLDR, Executive Summary, Location Analysis, etc.)
2. **TLDR Quality (20 points)**: Quality of TLDR section with recommendation, insight, and citations
3. **Markdown Formatting (20 points)**: Use of headers, bold, tables, checkboxes, bullets, and links
4. **Structure Quality (20 points)**: Document structure with appropriate header hierarchy

### Test Set

- Number of examples: {len(test_examples)}
- Scenarios covered: {', '.join(set(ex['metadata']['scenario'] for ex in test_examples))}

---

*Report generated by Logistics Sit Rep Fine-Tuning Pipeline*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Saved evaluation report to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model against baseline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/evaluation_config.yaml',
        help='Path to evaluation configuration file'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        help='Override test data file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Override output directory'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run comparison between baseline and fine-tuned'
    )
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Evaluate baseline model only'
    )
    parser.add_argument(
        '--finetuned-only',
        action='store_true',
        help='Evaluate fine-tuned model only'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip generation, use existing outputs'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Get paths
    test_file = args.test_file or config['test_data']['path']
    output_dir = args.output or config['output']['save_dir']

    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    logger.info(f"Loading test data from {test_file}")
    test_examples = load_test_data(test_file)

    if not args.skip_generation:
        # Create generators
        if args.compare or not (args.baseline_only or args.finetuned_only):
            logger.info("Creating both baseline and fine-tuned generators")
            generators = create_generators(
                baseline_model_path=config['models']['baseline']['path'],
                finetuned_model_path=config['models']['finetuned']['path'],
                device=config['processing'].get('device', 'auto')
            )

            # Generate outputs
            logger.info("Generating sit reps from both models")
            outputs = generate_comparison(
                test_examples=test_examples,
                baseline_generator=generators['baseline'],
                finetuned_generator=generators['finetuned'],
                generation_config=config['generation']
            )

            # Save generations
            save_generations(outputs, output_dir, test_examples)

        elif args.baseline_only:
            logger.info("Evaluating baseline only")
            # Implementation for baseline only
            pass

        elif args.finetuned_only:
            logger.info("Evaluating fine-tuned only")
            # Implementation for finetuned only
            pass

    else:
        # Load existing outputs
        logger.info("Loading existing outputs")
        outputs = {
            'baseline': [],
            'finetuned': [],
            'reference': [ex['output'] for ex in test_examples],
            'inputs': [ex['input'] for ex in test_examples]
        }

        # Load baseline
        baseline_jsonl = os.path.join(output_dir, 'baseline_outputs.jsonl')
        with open(baseline_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                outputs['baseline'].append(data['output'])

        # Load finetuned
        finetuned_jsonl = os.path.join(output_dir, 'finetuned_outputs.jsonl')
        with open(finetuned_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                outputs['finetuned'].append(data['output'])

    # Evaluate formatting
    logger.info("Evaluating formatting metrics")
    comparison_stats = compare_models(
        baseline_sitreps=outputs['baseline'],
        finetuned_sitreps=outputs['finetuned']
    )

    # Save detailed scores
    scores_path = os.path.join(output_dir, 'detailed_scores.json')
    with open(scores_path, 'w') as f:
        json.dump(comparison_stats, f, indent=2)
    logger.info(f"Saved detailed scores to {scores_path}")

    # Create visualizations
    if config['output'].get('save_visualizations', True):
        logger.info("Creating visualizations")
        viz_dir = os.path.join(output_dir, 'visualizations')
        create_comparison_visualizations(comparison_stats, viz_dir)

    # Generate report
    if config['output'].get('save_comparison_report', True):
        logger.info("Generating evaluation report")
        report_path = os.path.join(output_dir, 'evaluation_report.md')
        generate_markdown_report(
            comparison_stats=comparison_stats,
            test_examples=test_examples,
            outputs=outputs,
            output_path=report_path
        )

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Test Examples: {len(test_examples)}")
    print(f"\nBaseline Score: {comparison_stats['baseline']['mean_total']:.1f}/100")
    print(f"Fine-tuned Score: {comparison_stats['finetuned']['mean_total']:.1f}/100")
    print(f"Improvement: {comparison_stats['improvement']['total_score_improvement']:+.1f} points ({comparison_stats['improvement']['total_score_improvement_pct']:+.1f}%)")
    print(f"\nResults saved to: {output_dir}")
    print("="*60 + "\n")

    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
