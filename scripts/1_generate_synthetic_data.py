#!/usr/bin/env python3
"""
Script 1: Generate Synthetic Logistics Data

Generates synthetic logistics location data and corresponding sit reps
using NVIDIA Data Designer and LLM generation.

Usage:
    python scripts/1_generate_synthetic_data.py --scenarios coastal_expansion mountain_crisis
    python scripts/1_generate_synthetic_data.py --config config/data_designer_config.yaml
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

from src.data_generation.logistics_schema import ScenarioConfig
from src.data_generation.data_designer_client import (
    LogisticsDataGenerator,
    create_generator_from_env
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_scenario_from_config(scenario_config: dict, location_params: dict) -> ScenarioConfig:
    """Create ScenarioConfig from configuration dictionary."""
    name = scenario_config['name']

    # Get location parameters for this scenario
    lat_range = location_params.get(name, {}).get('lat_range', [30.0, 45.0])
    lon_range = location_params.get(name, {}).get('lon_range', [-120.0, -80.0])

    return ScenarioConfig(
        name=name,
        description=scenario_config.get('description', ''),
        days=scenario_config['days'],
        locations_per_day=scenario_config['locations_per_day'],
        weather_patterns=scenario_config.get('weather_patterns', []),
        terrain_types=scenario_config.get('terrain_types', []),
        lat_range=tuple(lat_range),
        lon_range=tuple(lon_range)
    )


def generate_scenarios(
    scenarios: List[ScenarioConfig],
    generator: LogisticsDataGenerator,
    output_dir: str
) -> dict:
    """
    Generate data for all scenarios.

    Args:
        scenarios: List of scenario configurations
        generator: Data generator instance
        output_dir: Output directory for generated data

    Returns:
        Dictionary with generation statistics
    """
    stats = {
        'total_scenarios': len(scenarios),
        'total_days': 0,
        'total_sitreps': 0,
        'total_locations': 0,
        'scenarios': {}
    }

    for scenario in tqdm(scenarios, desc="Generating scenarios"):
        logger.info(f"Processing scenario: {scenario.name}")

        try:
            results = generator.generate_scenario_data(
                scenario=scenario,
                output_dir=output_dir
            )

            num_sitreps = len(results)
            num_locations = sum(len(df) for df, _ in results)

            stats['total_days'] += scenario.days
            stats['total_sitreps'] += num_sitreps
            stats['total_locations'] += num_locations
            stats['scenarios'][scenario.name] = {
                'days': scenario.days,
                'sitreps_generated': num_sitreps,
                'locations_generated': num_locations,
                'success': True
            }

            logger.info(
                f"Completed {scenario.name}: {num_sitreps} sit reps, "
                f"{num_locations} locations"
            )

        except Exception as e:
            logger.error(f"Error processing scenario {scenario.name}: {e}")
            stats['scenarios'][scenario.name] = {
                'days': scenario.days,
                'sitreps_generated': 0,
                'locations_generated': 0,
                'success': False,
                'error': str(e)
            }

    return stats


def print_summary(stats: dict):
    """Print generation summary."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Scenarios: {stats['total_scenarios']}")
    print(f"Total Days: {stats['total_days']}")
    print(f"Total Sit Reps: {stats['total_sitreps']}")
    print(f"Total Locations: {stats['total_locations']}")
    print("\nPer-Scenario Breakdown:")
    print("-" * 60)

    for scenario_name, scenario_stats in stats['scenarios'].items():
        status = "✓" if scenario_stats['success'] else "✗"
        print(f"\n{status} {scenario_name}:")
        print(f"  Days: {scenario_stats['days']}")
        print(f"  Sit Reps: {scenario_stats['sitreps_generated']}")
        print(f"  Locations: {scenario_stats['locations_generated']}")
        if not scenario_stats['success']:
            print(f"  Error: {scenario_stats.get('error', 'Unknown')}")

    print("\n" + "=" * 60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic logistics data using NVIDIA Data Designer"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/data_designer_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        help='Specific scenarios to generate (if not specified, generates all from config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration without generating data'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Create data generator
    logger.info("Initializing NVIDIA Data Designer client")
    generator = create_generator_from_env()

    # Create scenario configs
    scenarios_to_generate = []
    location_params = config.get('location_params', {})

    for scenario_config in config['scenarios']:
        scenario_name = scenario_config['name']

        # Filter by requested scenarios
        if args.scenarios and scenario_name not in args.scenarios:
            continue

        scenario = create_scenario_from_config(scenario_config, location_params)
        scenarios_to_generate.append(scenario)

    logger.info(f"Configured {len(scenarios_to_generate)} scenarios")

    if args.dry_run:
        print("\n=== DRY RUN - Configuration Summary ===")
        for scenario in scenarios_to_generate:
            print(f"\nScenario: {scenario.name}")
            print(f"  Days: {scenario.days}")
            print(f"  Locations per day: {scenario.locations_per_day}")
            print(f"  Total sit reps: {scenario.days}")
            print(f"  Total locations: {scenario.days * scenario.locations_per_day}")
        print("\nTotal sit reps across all scenarios:", sum(s.days for s in scenarios_to_generate))
        print("Total locations:", sum(s.days * s.locations_per_day for s in scenarios_to_generate))
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate data
    logger.info("Starting data generation")
    stats = generate_scenarios(
        scenarios=scenarios_to_generate,
        generator=generator,
        output_dir=args.output_dir
    )

    # Print summary
    print_summary(stats)

    # Save stats to file
    stats_path = os.path.join(args.output_dir, 'generation_stats.yaml')
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    logger.info(f"Saved generation statistics to {stats_path}")

    logger.info("Data generation complete!")


if __name__ == '__main__':
    main()
