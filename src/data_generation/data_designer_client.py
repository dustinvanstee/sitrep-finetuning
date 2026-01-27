"""
Logistics data generator - generates synthetic data locally and uses NVIDIA API for sit reps.
Simplified version that doesn't require NVIDIA Data Designer library.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from openai import OpenAI

from .logistics_schema import (
    LogisticsLocation, ScenarioConfig,
    TerrainType, WeatherCondition, RoadQuality, Season
)
from .sitrep_templates import format_locations_for_prompt

logger = logging.getLogger(__name__)


class LogisticsDataGenerator:
    """Generate synthetic logistics data locally and sit reps via NVIDIA API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "meta/llama-3.1-70b-instruct"
    ):
        """
        Initialize the data generator.

        Args:
            api_key: NVIDIA API key
            base_url: Base URL for NVIDIA API
            model: LLM model to use for sit rep generation
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        np.random.seed(42)  # For reproducibility

    def generate_locations(
        self,
        scenario: ScenarioConfig,
        day: int,
        num_locations: int
    ) -> pd.DataFrame:
        """
        Generate synthetic location data locally.

        Args:
            scenario: Scenario configuration
            day: Day number in the scenario
            num_locations: Number of locations to generate

        Returns:
            DataFrame with location data
        """
        logger.info(f"Generating {num_locations} locations for {scenario.name} Day {day}")

        locations = []
        location_names = [
            "Alpha Base", "Bravo Station", "Charlie Point", "Delta Hub",
            "Echo Terminal", "Foxtrot Depot", "Golf Center", "Hotel Junction",
            "India Port", "Juliet Field", "Kilo Plaza", "Lima Crossing"
        ]

        for i in range(num_locations):
            # Generate base metrics
            lat = np.random.uniform(scenario.lat_range[0], scenario.lat_range[1])
            lon = np.random.uniform(scenario.lon_range[0], scenario.lon_range[1])
            elevation = int(np.random.uniform(0, 3000))
            distance_km = np.random.uniform(50, 500)
            num_competitors = int(np.random.uniform(1, 8))

            # Calculate costs
            base_cost = 50000 + (distance_km * 100) + (num_competitors * 5000) + (elevation * 10)
            transport_cost = 5.0 + (distance_km * 0.1)

            location = {
                'location_id': f'LOC-{day:02d}{i+1:02d}',
                'location_name': location_names[i % len(location_names)],
                'latitude': round(lat, 2),
                'longitude': round(lon, 2),
                'terrain_type': np.random.choice(scenario.terrain_types),
                'elevation_m': elevation,
                'weather_condition': np.random.choice(scenario.weather_patterns),
                'temperature_c': round(np.random.normal(20, 8), 1),
                'wind_speed_kmh': round(abs(np.random.normal(15, 8)), 1),
                'precipitation_mm': round(abs(np.random.normal(5, 10)), 1),
                'distance_from_dc_km': round(distance_km, 1),
                'road_quality': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor']),
                'has_airport': bool(np.random.random() < 0.4),
                'has_seaport': bool(np.random.random() < 0.3),
                'has_rail_access': bool(np.random.random() < 0.6),
                'num_competitors': num_competitors,
                'population_served': int(np.random.uniform(50000, 500000)),
                'current_demand_units': int(np.random.normal(5000, 2000)),
                'demand_growth_projection_pct': round(np.random.uniform(-5, 25), 1),
                'market_saturation_pct': round(np.random.uniform(20, 90), 1),
                'base_deployment_cost_usd': round(base_cost, 2),
                'transportation_cost_per_unit_usd': round(transport_cost, 2),
                'labor_cost_per_hour_usd': round(np.random.uniform(15, 35), 2),
                'facility_rental_monthly_usd': round(np.random.normal(25000, 10000), 2),
                'political_stability_score': round(np.random.uniform(4, 9), 1),
                'natural_disaster_risk_score': round(np.random.uniform(2, 8), 1),
                'supply_chain_risk_score': round(np.random.uniform(3, 9), 1),
                'day_number': day,
                'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'])
            }
            locations.append(location)

        return pd.DataFrame(locations)

    def generate_day_data(
        self,
        scenario: ScenarioConfig,
        day: int,
        output_dir: str
    ) -> tuple:
        """
        Generate data for a single day in a scenario.

        Args:
            scenario: Scenario configuration
            day: Day number
            output_dir: Directory to save outputs

        Returns:
            Tuple of (logistics_dataframe, sitrep_markdown)
        """
        logger.info(f"Generating data for {scenario.name} Day {day}")

        # Generate location data locally
        locations_df = self.generate_locations(
            scenario=scenario,
            day=day,
            num_locations=scenario.locations_per_day
        )

        # Convert to list of dicts for template
        locations_list = locations_df.to_dict('records')

        # Generate sit rep using NVIDIA API
        prompt = format_locations_for_prompt(
            locations=locations_list,
            scenario_name=scenario.name.replace('_', ' ').title(),
            day_number=day,
            total_days=scenario.days
        )

        logger.info("Generating sit rep with NVIDIA API (Llama 3.1 70B)")
        sitrep_content = self._generate_sitrep_with_llm(prompt)

        # Save outputs immediately
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(
            output_dir,
            f"logistics_data_{scenario.name}_day{day}.csv"
        )
        locations_df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved location data to {csv_path}")

        sitrep_path = os.path.join(
            output_dir,
            f"sitrep_{scenario.name}_day{day}.md"
        )
        with open(sitrep_path, 'w') as f:
            f.write(sitrep_content)
        logger.info(f"✓ Saved sit rep to {sitrep_path}")

        return locations_df, sitrep_content

    def _generate_sitrep_with_llm(self, prompt: str) -> str:
        """
        Generate sit rep content using NVIDIA API.

        Args:
            prompt: Formatted prompt with location data

        Returns:
            Generated sit rep markdown
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a logistics analyst generating comprehensive situation reports."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=8192
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating sit rep with NVIDIA API: {e}")
            logger.warning("Returning fallback sit rep")
            return self._generate_fallback_sitrep()

    def _generate_fallback_sitrep(self) -> str:
        """Generate a basic fallback sit rep if LLM fails."""
        return """
# Logistics Deployment Analysis

## TLDR
**Recommended Deployment Site:** Analysis pending
**Key Insight:** Data generation in progress

## Executive Summary
This is a placeholder sit rep generated due to API limitations.

## Location Analysis
Location analysis pending.

## Recommendations
Further analysis required.

## Action Items
- [ ] Review location data
- [ ] Complete analysis
"""

    def generate_scenario_data(
        self,
        scenario: ScenarioConfig,
        output_dir: str
    ) -> List[tuple]:
        """
        Generate complete dataset for a scenario across all days.

        Args:
            scenario: Scenario configuration
            output_dir: Directory to save outputs

        Returns:
            List of (dataframe, sitrep) tuples for each day
        """
        results = []

        for day in range(1, scenario.days + 1):
            # Check if this day already exists (resume capability)
            csv_path = os.path.join(output_dir, f"logistics_data_{scenario.name}_day{day}.csv")
            sitrep_path = os.path.join(output_dir, f"sitrep_{scenario.name}_day{day}.md")

            if os.path.exists(csv_path) and os.path.exists(sitrep_path):
                logger.info(f"✓ Skipping {scenario.name} Day {day} - already exists")
                # Load existing data for results
                try:
                    existing_df = pd.read_csv(csv_path)
                    with open(sitrep_path, 'r') as f:
                        existing_sitrep = f.read()
                    results.append((existing_df, existing_sitrep))
                    continue
                except Exception as e:
                    logger.warning(f"Could not load existing day {day}: {e}. Regenerating...")

            try:
                day_data = self.generate_day_data(
                    scenario=scenario,
                    day=day,
                    output_dir=output_dir
                )
                results.append(day_data)
            except Exception as e:
                logger.error(f"✗ Error generating day {day} data: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        # Combine all days into single CSV
        if results:
            all_locations = pd.concat([df for df, _ in results], ignore_index=True)
            combined_path = os.path.join(
                output_dir,
                f"logistics_data_{scenario.name}_complete.csv"
            )
            all_locations.to_csv(combined_path, index=False)
            logger.info(f"✓ Saved combined data to {combined_path}")

        return results


def create_generator_from_env() -> LogisticsDataGenerator:
    """
    Create data generator from environment variables.

    Returns:
        Configured LogisticsDataGenerator
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NEMO_MICROSERVICES_BASE_URL", "https://integrate.api.nvidia.com/v1")

    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable not set")

    return LogisticsDataGenerator(api_key=api_key, base_url=base_url)
