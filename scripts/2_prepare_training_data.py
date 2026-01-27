#!/usr/bin/env python3
"""
Script 2: Prepare Training Data

Converts raw logistics data and sit reps into JSONL format for model training.

Usage:
    python scripts/2_prepare_training_data.py --raw_dir data/raw --output_dir data/processed
    python scripts/2_prepare_training_data.py --scenarios coastal_expansion mountain_crisis
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import yaml
from tqdm import tqdm

from src.training.data_preprocessor import (
    LogisticsDataPreprocessor,
    validate_training_data,
    TrainingExample
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def discover_scenarios(raw_data_dir: str) -> dict:
    """
    Discover scenarios and their day counts from raw data directory.

    Args:
        raw_data_dir: Directory with raw data files

    Returns:
        Dictionary mapping scenario names to number of days
    """
    scenarios = {}

    for filename in os.listdir(raw_data_dir):
        if filename.startswith('logistics_data_') and filename.endswith('.csv'):
            # Parse filename: logistics_data_{scenario}_day{N}.csv
            parts = filename.replace('logistics_data_', '').replace('.csv', '').split('_day')
            if len(parts) == 2:
                scenario_name = parts[0]
                day_number = int(parts[1])

                if scenario_name not in scenarios:
                    scenarios[scenario_name] = 0
                scenarios[scenario_name] = max(scenarios[scenario_name], day_number)

    logger.info(f"Discovered {len(scenarios)} scenarios: {list(scenarios.keys())}")
    return scenarios


def process_all_scenarios(
    raw_data_dir: str,
    scenarios: dict,
    preprocessor: LogisticsDataPreprocessor,
    filter_scenarios: List[str] = None
) -> List[TrainingExample]:
    """
    Process all scenarios into training examples.

    Args:
        raw_data_dir: Directory with raw data
        scenarios: Dictionary of scenario names to day counts
        preprocessor: Data preprocessor instance
        filter_scenarios: Optional list of scenario names to process

    Returns:
        List of all training examples
    """
    all_examples = []

    for scenario_name, num_days in tqdm(scenarios.items(), desc="Processing scenarios"):
        # Filter if specified
        if filter_scenarios and scenario_name not in filter_scenarios:
            logger.info(f"Skipping {scenario_name} (not in filter)")
            continue

        logger.info(f"Processing {scenario_name} ({num_days} days)")

        examples = preprocessor.process_scenario_files(
            raw_data_dir=raw_data_dir,
            scenario_name=scenario_name,
            days=num_days
        )

        all_examples.extend(examples)
        logger.info(f"Generated {len(examples)} examples from {scenario_name}")

    return all_examples


def print_data_summary(train_examples: List[TrainingExample], val_examples: List[TrainingExample]):
    """Print summary of processed data."""
    print("\n" + "=" * 60)
    print("TRAINING DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total Examples: {len(train_examples) + len(val_examples)}")
    print(f"Training Examples: {len(train_examples)}")
    print(f"Validation Examples: {len(val_examples)}")
    print(f"Train/Val Split: {len(train_examples)/(len(train_examples)+len(val_examples))*100:.1f}% / {len(val_examples)/(len(train_examples)+len(val_examples))*100:.1f}%")

    # Count by scenario
    train_scenarios = {}
    val_scenarios = {}

    for ex in train_examples:
        scenario = ex.metadata['scenario']
        train_scenarios[scenario] = train_scenarios.get(scenario, 0) + 1

    for ex in val_examples:
        scenario = ex.metadata['scenario']
        val_scenarios[scenario] = val_scenarios.get(scenario, 0) + 1

    print("\nExamples per Scenario:")
    print("-" * 60)
    all_scenarios = set(train_scenarios.keys()) | set(val_scenarios.keys())
    for scenario in sorted(all_scenarios):
        train_count = train_scenarios.get(scenario, 0)
        val_count = val_scenarios.get(scenario, 0)
        print(f"{scenario}:")
        print(f"  Train: {train_count}, Val: {val_count}, Total: {train_count + val_count}")

    print("\n" + "=" * 60)


def validate_and_report(train_path: str, val_path: str):
    """Validate training data and print report."""
    print("\n" + "=" * 60)
    print("TRAINING DATA VALIDATION")
    print("=" * 60)

    print("\nTraining Set:")
    train_stats = validate_training_data(train_path)
    print(f"  Examples: {train_stats['num_examples']}")
    print(f"  Avg Input Length: {train_stats['avg_input_length']:.0f} chars")
    print(f"  Avg Output Length: {train_stats['avg_output_length']:.0f} chars")
    print(f"  Max Input Length: {train_stats['max_input_length']} chars")
    print(f"  Max Output Length: {train_stats['max_output_length']} chars")
    print(f"  Scenarios: {', '.join(train_stats['scenarios'])}")

    print("\nValidation Set:")
    val_stats = validate_training_data(val_path)
    print(f"  Examples: {val_stats['num_examples']}")
    print(f"  Avg Input Length: {val_stats['avg_input_length']:.0f} chars")
    print(f"  Avg Output Length: {val_stats['avg_output_length']:.0f} chars")
    print(f"  Max Input Length: {val_stats['max_input_length']} chars")
    print(f"  Max Output Length: {val_stats['max_output_length']} chars")
    print(f"  Scenarios: {', '.join(val_stats['scenarios'])}")

    # Estimate tokens (rough approximation: 4 chars per token)
    print("\nToken Estimates (approx 4 chars/token):")
    print(f"  Train avg: {train_stats['avg_input_length']/4:.0f} input + {train_stats['avg_output_length']/4:.0f} output = {(train_stats['avg_input_length'] + train_stats['avg_output_length'])/4:.0f} total")
    print(f"  Val avg: {val_stats['avg_input_length']/4:.0f} input + {val_stats['avg_output_length']/4:.0f} output = {(val_stats['avg_input_length'] + val_stats['avg_output_length'])/4:.0f} total")

    print("\n" + "=" * 60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Prepare training data from raw logistics data and sit reps"
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw',
        help='Directory with raw data files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed JSONL files'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        help='Specific scenarios to process (if not specified, processes all)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Fraction of data for validation (default: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation on existing JSONL files'
    )

    args = parser.parse_args()

    # If validate mode, just validate and exit
    if args.validate:
        train_path = os.path.join(args.output_dir, 'train.jsonl')
        val_path = os.path.join(args.output_dir, 'val.jsonl')

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            logger.error("Training files not found. Run without --validate first.")
            return

        validate_and_report(train_path, val_path)
        return

    # Load environment variables
    load_dotenv()

    # Discover scenarios
    logger.info(f"Scanning raw data directory: {args.raw_dir}")
    scenarios = discover_scenarios(args.raw_dir)

    if not scenarios:
        logger.error(f"No scenarios found in {args.raw_dir}")
        return

    # Create preprocessor
    preprocessor = LogisticsDataPreprocessor()

    # Process all scenarios
    logger.info("Processing scenarios into training examples")
    all_examples = process_all_scenarios(
        raw_data_dir=args.raw_dir,
        scenarios=scenarios,
        preprocessor=preprocessor,
        filter_scenarios=args.scenarios
    )

    if not all_examples:
        logger.error("No training examples generated")
        return

    logger.info(f"Generated {len(all_examples)} total examples")

    # Split train/val
    logger.info(f"Splitting data with {args.val_split:.0%} for validation")
    train_examples, val_examples = preprocessor.split_train_val(
        examples=all_examples,
        val_split=args.val_split,
        random_seed=args.seed
    )

    # Print summary
    print_data_summary(train_examples, val_examples)

    # Save to JSONL
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, 'train.jsonl')
    val_path = os.path.join(args.output_dir, 'val.jsonl')

    logger.info(f"Saving training data to {train_path}")
    preprocessor.save_examples_to_jsonl(train_examples, train_path)

    logger.info(f"Saving validation data to {val_path}")
    preprocessor.save_examples_to_jsonl(val_examples, val_path)

    # Validate and report
    validate_and_report(train_path, val_path)

    logger.info("Data preparation complete!")


if __name__ == '__main__':
    main()
