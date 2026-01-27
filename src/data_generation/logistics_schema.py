"""
Logistics data schema definitions for synthetic data generation.
Defines the structure and parameters for location data.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class TerrainType(Enum):
    """Terrain types for logistics locations."""
    COASTAL_PLAIN = "Coastal Plain"
    BEACH = "Beach"
    LOW_HILLS = "Low Hills"
    MOUNTAIN = "Mountain"
    HIGHLAND = "Highland"
    VALLEY = "Valley"
    URBAN = "Urban"
    SUBURBAN = "Suburban"
    INDUSTRIAL = "Industrial"
    DESERT = "Desert"
    FOREST = "Forest"
    PLAINS = "Plains"


class WeatherCondition(Enum):
    """Weather conditions."""
    CLEAR = "Clear"
    PARTLY_CLOUDY = "Partly Cloudy"
    CLOUDY = "Cloudy"
    OVERCAST = "Overcast"
    RAIN = "Rain"
    HEAVY_RAIN = "Heavy Rain"
    SNOW = "Snow"
    FOG = "Fog"
    STORMY = "Stormy"


class RoadQuality(Enum):
    """Road infrastructure quality."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"


class Season(Enum):
    """Seasonal categorization."""
    SPRING = "Spring"
    SUMMER = "Summer"
    FALL = "Fall"
    WINTER = "Winter"


@dataclass
class LogisticsLocation:
    """Complete logistics location data structure."""

    # Identification
    location_id: str
    location_name: str

    # Geography
    latitude: float
    longitude: float
    terrain_type: str
    elevation_m: int

    # Weather
    weather_condition: str
    temperature_c: float
    wind_speed_kmh: float
    precipitation_mm: float

    # Infrastructure
    distance_from_dc_km: float
    road_quality: str
    has_airport: bool
    has_seaport: bool
    has_rail_access: bool

    # Market
    num_competitors: int
    population_served: int
    current_demand_units: int
    demand_growth_projection_pct: float
    market_saturation_pct: float

    # Costs (USD)
    base_deployment_cost_usd: float
    transportation_cost_per_unit_usd: float
    labor_cost_per_hour_usd: float
    facility_rental_monthly_usd: float

    # Risk Scores (0-10)
    political_stability_score: float
    natural_disaster_risk_score: float
    supply_chain_risk_score: float

    # Temporal
    day_number: int
    season: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/DataFrame export."""
        return {
            'location_id': self.location_id,
            'location_name': self.location_name,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'terrain_type': self.terrain_type,
            'elevation_m': self.elevation_m,
            'weather_condition': self.weather_condition,
            'temperature_c': self.temperature_c,
            'wind_speed_kmh': self.wind_speed_kmh,
            'precipitation_mm': self.precipitation_mm,
            'distance_from_dc_km': self.distance_from_dc_km,
            'road_quality': self.road_quality,
            'has_airport': self.has_airport,
            'has_seaport': self.has_seaport,
            'has_rail_access': self.has_rail_access,
            'num_competitors': self.num_competitors,
            'population_served': self.population_served,
            'current_demand_units': self.current_demand_units,
            'demand_growth_projection_pct': self.demand_growth_projection_pct,
            'market_saturation_pct': self.market_saturation_pct,
            'base_deployment_cost_usd': self.base_deployment_cost_usd,
            'transportation_cost_per_unit_usd': self.transportation_cost_per_unit_usd,
            'labor_cost_per_hour_usd': self.labor_cost_per_hour_usd,
            'facility_rental_monthly_usd': self.facility_rental_monthly_usd,
            'political_stability_score': self.political_stability_score,
            'natural_disaster_risk_score': self.natural_disaster_risk_score,
            'supply_chain_risk_score': self.supply_chain_risk_score,
            'day_number': self.day_number,
            'season': self.season
        }


@dataclass
class ScenarioConfig:
    """Configuration for a deployment scenario."""
    name: str
    description: str
    days: int
    locations_per_day: int
    weather_patterns: List[str]
    terrain_types: List[str]
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]

    def __post_init__(self):
        """Validate configuration."""
        assert self.days > 0, "Days must be positive"
        assert self.locations_per_day > 0, "Locations per day must be positive"
        assert len(self.weather_patterns) > 0, "Must have at least one weather pattern"
        assert len(self.terrain_types) > 0, "Must have at least one terrain type"


# Column definitions for Data Designer
LOGISTICS_COLUMNS = {
    # Categorical columns
    "location_name": {
        "type": "sampler",
        "sampler_type": "category",
        "values": [
            "Alpha Base", "Bravo Station", "Charlie Point", "Delta Hub",
            "Echo Terminal", "Foxtrot Depot", "Golf Center", "Hotel Junction",
            "India Port", "Juliet Field", "Kilo Plaza", "Lima Crossing",
            "Mike Summit", "November Bay", "Oscar Valley", "Papa Ridge"
        ]
    },
    "terrain_type": {
        "type": "sampler",
        "sampler_type": "category",
        "values": [t.value for t in TerrainType]
    },
    "weather_condition": {
        "type": "sampler",
        "sampler_type": "category",
        "values": [w.value for w in WeatherCondition]
    },
    "road_quality": {
        "type": "sampler",
        "sampler_type": "category",
        "values": [r.value for r in RoadQuality]
    },
    "season": {
        "type": "sampler",
        "sampler_type": "category",
        "values": [s.value for s in Season]
    },

    # Numeric columns with ranges
    "latitude": {"type": "uniform", "min": 30.0, "max": 45.0},
    "longitude": {"type": "uniform", "min": -120.0, "max": -80.0},
    "elevation_m": {"type": "uniform_int", "min": 0, "max": 3000},
    "temperature_c": {"type": "normal", "mean": 20.0, "std": 8.0},
    "wind_speed_kmh": {"type": "lognormal", "mean": 15.0, "std": 8.0},
    "precipitation_mm": {"type": "lognormal", "mean": 5.0, "std": 10.0},
    "distance_from_dc_km": {"type": "uniform", "min": 50.0, "max": 500.0},
    "population_served": {"type": "uniform_int", "min": 50000, "max": 500000},
    "current_demand_units": {"type": "normal_int", "mean": 5000, "std": 2000},
    "demand_growth_projection_pct": {"type": "uniform", "min": -5.0, "max": 25.0},
    "market_saturation_pct": {"type": "uniform", "min": 20.0, "max": 90.0},
    "num_competitors": {"type": "uniform_int", "min": 1, "max": 8},
    "labor_cost_per_hour_usd": {"type": "uniform", "min": 15.0, "max": 35.0},
    "facility_rental_monthly_usd": {"type": "normal", "mean": 25000, "std": 10000},
    "political_stability_score": {"type": "uniform", "min": 4.0, "max": 9.0},
    "natural_disaster_risk_score": {"type": "uniform", "min": 2.0, "max": 8.0},
    "supply_chain_risk_score": {"type": "uniform", "min": 3.0, "max": 9.0},

    # Boolean columns
    "has_airport": {"type": "boolean", "probability": 0.4},
    "has_seaport": {"type": "boolean", "probability": 0.3},
    "has_rail_access": {"type": "boolean", "probability": 0.6},
}


# Expression templates for computed columns
EXPRESSION_TEMPLATES = {
    "base_deployment_cost_usd": "50000 + ({{ distance_from_dc_km }} * 100) + ({{ num_competitors }} * 5000) + ({{ elevation_m }} * 10)",
    "transportation_cost_per_unit_usd": "5.0 + ({{ distance_from_dc_km }} * 0.1) + ({{ road_quality == 'Poor' }} * 3.0)",
    "location_id": "'LOC-' + str({{ row_index }}).zfill(4)",
}
