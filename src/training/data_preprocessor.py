"""
Data preprocessing utilities for training data preparation.
Converts raw logistics data and sit reps into training format.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example for instruction fine-tuning."""
    input: str
    output: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSONL export."""
        return {
            'input': self.input,
            'output': self.output,
            'metadata': self.metadata
        }


class LogisticsDataPreprocessor:
    """Preprocess logistics data for model training."""

    def __init__(self, input_template: str = None):
        """
        Initialize preprocessor.

        Args:
            input_template: Custom template for formatting input prompts
        """
        self.input_template = input_template or self._default_input_template()

    @staticmethod
    def _default_input_template() -> str:
        """Default template for input prompts."""
        return """Generate a comprehensive logistics deployment situation report for Day {day_number}.

## Logistics Data

{locations_data}

Generate a well-formatted markdown sit rep with all required sections: TLDR, Executive Summary, Location Analysis, Comparative Analysis, Temporal Trends, Recommendations, and Action Items."""

    def format_location_data(self, location: Dict[str, Any]) -> str:
        """
        Format a single location's data for the prompt.

        Args:
            location: Dictionary with location data

        Returns:
            Formatted string
        """
        return f"""### Location: {location.get('location_name', 'Unknown')} (ID: {location.get('location_id', 'N/A')})
- Coordinates: {location.get('latitude', 0):.2f}°N, {abs(location.get('longitude', 0)):.2f}°W
- Terrain: {location.get('terrain_type', 'Unknown')} at {location.get('elevation_m', 0)}m elevation
- Weather: {location.get('weather_condition', 'Unknown')}, {location.get('temperature_c', 0):.1f}°C
- Wind Speed: {location.get('wind_speed_kmh', 0):.1f} km/h
- Distance from DC: {location.get('distance_from_dc_km', 0):.1f} km
- Road Quality: {location.get('road_quality', 'Unknown')}
- Population Served: {location.get('population_served', 0):,}
- Current Demand: {location.get('current_demand_units', 0):,} units/day
- Demand Growth: {location.get('demand_growth_projection_pct', 0):.1f}%
- Competitors: {location.get('num_competitors', 0)}
- Base Deployment Cost: ${location.get('base_deployment_cost_usd', 0):,.0f}
- Transport Cost per Unit: ${location.get('transportation_cost_per_unit_usd', 0):.2f}
- Labor Cost: ${location.get('labor_cost_per_hour_usd', 0):.2f}/hour
- Political Stability: {location.get('political_stability_score', 0):.1f}/10
- Disaster Risk: {location.get('natural_disaster_risk_score', 0):.1f}/10
- Supply Chain Risk: {location.get('supply_chain_risk_score', 0):.1f}/10
- Infrastructure: Airport={location.get('has_airport', False)}, Seaport={location.get('has_seaport', False)}, Rail={location.get('has_rail_access', False)}
"""

    def create_training_example(
        self,
        locations_df: pd.DataFrame,
        sitrep_content: str,
        scenario_name: str,
        day_number: int
    ) -> TrainingExample:
        """
        Create a training example from location data and sit rep.

        Args:
            locations_df: DataFrame with location data
            sitrep_content: Generated sit rep markdown
            scenario_name: Name of the scenario
            day_number: Day number

        Returns:
            TrainingExample instance
        """
        # Format all locations
        locations_data = "\n".join([
            self.format_location_data(row)
            for _, row in locations_df.iterrows()
        ])

        # Create input prompt
        input_text = self.input_template.format(
            day_number=day_number,
            locations_data=locations_data
        )

        # Create training example
        return TrainingExample(
            input=input_text,
            output=sitrep_content,
            metadata={
                'scenario': scenario_name,
                'day': day_number,
                'num_locations': len(locations_df)
            }
        )

    def process_scenario_files(
        self,
        raw_data_dir: str,
        scenario_name: str,
        days: int
    ) -> List[TrainingExample]:
        """
        Process all files for a scenario.

        Args:
            raw_data_dir: Directory with raw data files
            scenario_name: Scenario name
            days: Number of days in scenario

        Returns:
            List of training examples
        """
        examples = []

        for day in range(1, days + 1):
            # Load CSV
            csv_path = os.path.join(
                raw_data_dir,
                f"logistics_data_{scenario_name}_day{day}.csv"
            )

            # Load markdown
            md_path = os.path.join(
                raw_data_dir,
                f"sitrep_{scenario_name}_day{day}.md"
            )

            if not os.path.exists(csv_path):
                logger.warning(f"CSV not found: {csv_path}")
                continue

            if not os.path.exists(md_path):
                logger.warning(f"Markdown not found: {md_path}")
                continue

            try:
                # Read data
                locations_df = pd.read_csv(csv_path)

                with open(md_path, 'r') as f:
                    sitrep_content = f.read()

                # Create example
                example = self.create_training_example(
                    locations_df=locations_df,
                    sitrep_content=sitrep_content,
                    scenario_name=scenario_name,
                    day_number=day
                )

                examples.append(example)
                logger.info(f"Processed {scenario_name} Day {day}")

            except Exception as e:
                logger.error(f"Error processing {scenario_name} Day {day}: {e}")
                continue

        return examples

    def save_examples_to_jsonl(
        self,
        examples: List[TrainingExample],
        output_path: str
    ):
        """
        Save training examples to JSONL file.

        Args:
            examples: List of training examples
            output_path: Path to output JSONL file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            for example in examples:
                json_line = json.dumps(example.to_dict())
                f.write(json_line + '\n')

        logger.info(f"Saved {len(examples)} examples to {output_path}")

    def split_train_val(
        self,
        examples: List[TrainingExample],
        val_split: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """
        Split examples into train and validation sets.

        Args:
            examples: List of training examples
            val_split: Fraction of data for validation
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_examples, val_examples)
        """
        import random
        random.seed(random_seed)

        # Group by scenario to ensure scenario diversity in both sets
        scenarios = {}
        for example in examples:
            scenario = example.metadata['scenario']
            if scenario not in scenarios:
                scenarios[scenario] = []
            scenarios[scenario].append(example)

        train_examples = []
        val_examples = []

        # Split each scenario
        for scenario, scenario_examples in scenarios.items():
            random.shuffle(scenario_examples)
            n_val = max(1, int(len(scenario_examples) * val_split))

            val_examples.extend(scenario_examples[:n_val])
            train_examples.extend(scenario_examples[n_val:])

        logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} val")
        return train_examples, val_examples


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load training examples from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of example dictionaries
    """
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def validate_training_data(file_path: str) -> dict:
    """
    Validate training data file.

    Args:
        file_path: Path to JSONL file

    Returns:
        Dictionary with validation statistics
    """
    examples = load_jsonl(file_path)

    stats = {
        'num_examples': len(examples),
        'avg_input_length': 0,
        'avg_output_length': 0,
        'max_input_length': 0,
        'max_output_length': 0,
        'scenarios': set(),
        'days_per_scenario': {}
    }

    for example in examples:
        input_len = len(example['input'])
        output_len = len(example['output'])

        stats['avg_input_length'] += input_len
        stats['avg_output_length'] += output_len
        stats['max_input_length'] = max(stats['max_input_length'], input_len)
        stats['max_output_length'] = max(stats['max_output_length'], output_len)

        scenario = example['metadata']['scenario']
        stats['scenarios'].add(scenario)

        if scenario not in stats['days_per_scenario']:
            stats['days_per_scenario'][scenario] = set()
        stats['days_per_scenario'][scenario].add(example['metadata']['day'])

    if examples:
        stats['avg_input_length'] /= len(examples)
        stats['avg_output_length'] /= len(examples)

    stats['scenarios'] = list(stats['scenarios'])
    stats['days_per_scenario'] = {
        k: len(v) for k, v in stats['days_per_scenario'].items()
    }

    return stats
