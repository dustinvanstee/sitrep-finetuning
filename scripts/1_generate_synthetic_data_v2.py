#!/usr/bin/env python3
"""
Generate synthetic INDOPACOM SITREP training data using NeMo Data Designer.

For each scenario:
- Generates variable number of input intel report snippets (markdown files)
- Generates consolidated 8-section SITREP output

Usage:
    cd /home/dvanstee/projects/2026-01-nt3-sitrep
    DataDesigner/.venv/bin/python scripts/1_generate_synthetic_data_v2.py --count 1
    DataDesigner/.venv/bin/python scripts/1_generate_synthetic_data_v2.py --count 25
"""
import os
import json
import argparse
import uuid
import random
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import litellm
from pydantic import BaseModel, Field

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"Loaded environment from: {ENV_FILE}")
else:
    # Try config/.env as fallback
    ALT_ENV = PROJECT_ROOT / "config" / ".env"
    if ALT_ENV.exists():
        load_dotenv(ALT_ENV)
        print(f"Loaded environment from: {ALT_ENV}")

# Read NIM configuration from environment
NIM_BASE_URL = os.environ.get("NEMO_MICROSERVICES_BASE_URL", "http://localhost:8000/v1")
NIM_MODEL = os.environ.get("NIM_MODEL", "nvidia/nemotron-3-nano")

DATA_DESIGNER_DIR = PROJECT_ROOT / ".data-designer"
os.environ.setdefault("DATA_DESIGNER_HOME", str(DATA_DESIGNER_DIR))

# Import Data Designer after env setup
from data_designer.interface.data_designer import DataDesigner
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig, ModelProvider, ChatCompletionInferenceParams
from data_designer.config.column_configs import (
    SamplerColumnConfig, 
    LLMStructuredColumnConfig,
)
from data_designer.config.seed_source import DataFrameSeedSource
from data_designer.config.sampler_params import CategorySamplerParams
from data_designer.engine.secret_resolver import PlaintextResolver

OUTPUT_DIR = PROJECT_ROOT / "data" / "scenarios"


# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================

@dataclass
class ScenarioParams:
    """Parameters for a generated scenario."""
    scenario_type: str
    theater_location: str
    escalation_level: str
    reporting_unit: str
    dtg: str
    reporting_period_start: str
    reporting_period_end: str
    num_flash_reports: int
    num_isr_reports: int
    num_sensor_reports: int
    include_missile_warning: bool
    include_cyber_report: bool
    include_partner_report: bool
    include_narrative: bool


# Scenario types with their report density profiles
SCENARIO_PROFILES = {
    "taiwan_invasion": {
        "description": "Full-scale amphibious invasion of Taiwan",
        "min_reports": 8, "max_reports": 12,
        "missile_prob": 0.95, "cyber_prob": 0.90, "partner_prob": 0.85,
    },
    "south_china_sea_standoff": {
        "description": "Naval standoff near disputed islands",
        "min_reports": 5, "max_reports": 8,
        "missile_prob": 0.30, "cyber_prob": 0.50, "partner_prob": 0.70,
    },
    "vessel_incursion": {
        "description": "Single vessel territorial incursion",
        "min_reports": 3, "max_reports": 5,
        "missile_prob": 0.05, "cyber_prob": 0.20, "partner_prob": 0.40,
    },
    "korean_peninsula_tension": {
        "description": "Elevated tension on Korean peninsula",
        "min_reports": 6, "max_reports": 10,
        "missile_prob": 0.70, "cyber_prob": 0.65, "partner_prob": 0.80,
    },
    "routine_patrol": {
        "description": "Routine patrol operations with minor activity",
        "min_reports": 2, "max_reports": 4,
        "missile_prob": 0.02, "cyber_prob": 0.15, "partner_prob": 0.30,
    },
    "exercise_poseidon": {
        "description": "Joint allied exercise in Philippine Sea",
        "min_reports": 4, "max_reports": 7,
        "missile_prob": 0.10, "cyber_prob": 0.25, "partner_prob": 0.95,
    },
}

THEATER_LOCATIONS = [
    "Taiwan Strait",
    "South China Sea - Spratly Islands",
    "South China Sea - Paracel Islands",
    "Philippine Sea",
    "East China Sea - Senkaku Islands",
    "Korean Peninsula - DMZ",
    "Korean Peninsula - West Sea",
    "Luzon Strait",
    "Miyako Strait",
]

REPORTING_UNITS = [
    "MARFORPAC",
    "PACFLT",
    "7th Fleet",
    "III MEF",
    "INDOPACOM J2",
    "CTF 70",
    "CTF 76",
]

ESCALATION_LEVELS = ["LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"]


class BLUFSection(BaseModel):
    """Bottom Line Up Front section of SITREP."""
    bullet_points: list[str] = Field(description="2-3 critical summary bullet points", min_length=2, max_length=4)
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(description="Overall confidence level")
    probability_assessment: str = Field(description="Percentage breakdown of likely scenarios")


class FriendlyForcesSection(BaseModel):
    """Section 2: Friendly Forces status."""
    unit_status: list[str] = Field(description="Status of each major unit (2-4 items)", min_length=2, max_length=5)
    assessment: str = Field(description="Overall friendly forces assessment")


class AdversaryActivitySection(BaseModel):
    """Section 3: Adversary activity summary."""
    observations: list[str] = Field(description="Key adversary observations (2-4 items)", min_length=2, max_length=5)
    assessment: str = Field(description="Threat assessment summary")


class OperationsSection(BaseModel):
    """Section 4: Current operations."""
    current_ops: list[str] = Field(description="Current operations summary (2-3 items)", min_length=1, max_length=4)
    planned_ops: str = Field(description="Planned operations or next actions")


class LogisticsSection(BaseModel):
    """Section 5: Logistics status."""
    supply_status: str = Field(description="Current supply/logistics status")
    shortfalls: str = Field(description="Any logistics shortfalls or concerns")
    resupply_eta: str = Field(description="Expected resupply timeline if applicable")


class IntelligenceAssessmentSection(BaseModel):
    """Section 6: Intelligence assessment."""
    key_findings: list[str] = Field(description="Key intelligence findings (2-4 items)", min_length=2, max_length=5)
    information_gaps: str = Field(description="Known information gaps")
    analyst_judgment: str = Field(description="Analyst recommendation")


class RisksSection(BaseModel):
    """Section 7: Risks and watch items."""
    risks: list[str] = Field(description="Key risks and watch items (2-4 bullet points)", min_length=2, max_length=5)


class CommanderCommentsSection(BaseModel):
    """Section 8: Commander's comments."""
    guidance: str = Field(description="Brief commander guidance (1-2 sentences)")
    priority_focus: str = Field(description="Priority focus area")


class SITREPStructured(BaseModel):
    """
    Pydantic model for the 8-section SITREP output.
    Uses LLMStructuredColumnConfig for guaranteed schema compliance.
    """
    bluf: BLUFSection = Field(description="Section 1: Bottom Line Up Front")
    friendly_forces: FriendlyForcesSection = Field(description="Section 2: Friendly Forces")
    adversary_activity: AdversaryActivitySection = Field(description="Section 3: Adversary Activity")
    operations: OperationsSection = Field(description="Section 4: Operations")
    logistics: LogisticsSection = Field(description="Section 5: Logistics")
    intelligence_assessment: IntelligenceAssessmentSection = Field(description="Section 6: Intelligence Assessment")
    risks: RisksSection = Field(description="Section 7: Risks / Watch Items")
    commander_comments: CommanderCommentsSection = Field(description="Section 8: Commander's Comments")


class ScenarioBackbone(BaseModel):
    """
    Pydantic model for consistent scenario facts - generated in Pass 1.
    Uses LLMStructuredColumnConfig for guaranteed schema compliance.
    All input reports reference these facts for correlation.
    """
    # Timeline (all timestamps within the reporting period)
    primary_event_dtg: str = Field(description="Primary event timestamp in DDHHMMz format")
    secondary_event_dtg: str = Field(description="Secondary event timestamp in DDHHMMz format")
    isr_observation_dtg: str = Field(description="ISR collection timestamp in DDHHMMz format")
    
    # Adversary details
    adversary_unit: str = Field(description="Specific unit designation, e.g., 'PLA Navy East Sea Fleet'")
    adversary_vessel_count: int = Field(description="Number of vessels observed", ge=1, le=50)
    adversary_aircraft_count: int = Field(description="Number of aircraft observed", ge=0, le=100)
    adversary_formation_location: str = Field(description="Specific location, e.g., '15nm east of Senkaku Islands'")
    adversary_grid_ref: str = Field(description="Military grid reference, e.g., '25.5°N 122.3°E'")
    adversary_activity: str = Field(description="Brief description of adversary actions")
    
    # Missile details
    missile_type: Literal["SRBM", "MRBM", "IRBM", "cruise", "none"] = Field(description="Missile class or 'none'")
    missile_launch_location: str = Field(description="Launch site description")
    missile_trajectory: str = Field(description="Direction, e.g., 'Northeast toward Taiwan Strait'")
    missile_count: int = Field(description="Number of missiles, 0 if none", ge=0, le=20)
    
    # Friendly force details
    friendly_arg_name: str = Field(description="ARG designation, e.g., 'ARG-71', 'ESG-7'")
    friendly_air_wing: str = Field(description="Air wing, e.g., 'CVW-5', 'MAG-12'")
    friendly_readiness_pct: int = Field(description="Readiness percentage", ge=70, le=100)
    friendly_munitions_status: str = Field(description="e.g., '98% stockpiles; 2% shortfall in PGMs'")
    friendly_personnel_count: int = Field(description="Total personnel count", ge=500, le=50000)
    friendly_personnel_status: str = Field(description="e.g., '94% strength; 850 replacements en route'")
    
    # Cyber/EW details
    cyber_target: str = Field(description="Target of cyber ops")
    cyber_method: str = Field(description="e.g., 'DDoS attacks, GPS spoofing'")
    ew_effect: str = Field(description="EW impact description")
    
    # Partner details
    partner_nation: Literal["Japan", "South Korea", "Australia", "Philippines", "Taiwan"] = Field(description="Partner nation")
    partner_request: str = Field(description="What partner is requesting")
    
    # Environmental
    weather_visibility: Literal["GOOD", "FAIR", "POOR"] = Field(description="Visibility conditions")
    sea_state: int = Field(description="Sea state 1-6", ge=1, le=6)
    weather_forecast: str = Field(description="Next 12-24 hour forecast")
    
    # Key intelligence
    primary_assessment: str = Field(description="1-2 sentence situation assessment")
    threat_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Threat level")
    confidence_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Intelligence confidence")


# =============================================================================
# DATE-TIME GROUP UTILITIES
# =============================================================================

def generate_dtg(base_time: datetime) -> str:
    """Generate military Date-Time Group (DDHHMMz MON YY)."""
    return base_time.strftime("%d%H%MZ %b %y").upper()


def generate_reporting_period(dtg_time: datetime) -> tuple[str, str]:
    """Generate 24-hour reporting period ending at DTG."""
    end_time = dtg_time
    start_time = end_time - timedelta(hours=24)
    return (
        start_time.strftime("%d%H%MZ").upper(),
        end_time.strftime("%d%H%MZ").upper()
    )


# =============================================================================
# DATA DESIGNER SETUP
# =============================================================================

def create_designer(use_nvidia_api: bool = False) -> tuple:
    """
    Create DataDesigner instance.
    
    Args:
        use_nvidia_api: If True, use NVIDIA build API. If False, use local NIM (default).
    """
    api_key = os.environ.get("NVIDIA_API_KEY", "not-needed")
    
    if use_nvidia_api and api_key not in ("", "USING_LOCAL_NIM_SERVICE", "not-needed"):
        # NVIDIA Build API (only if we have a real API key)
        base_url = "https://integrate.api.nvidia.com/v1"
        model_name = "nvidia/llama-3.1-nemotron-70b-instruct"
        provider_name = "nvidia"
        
        model_provider = ModelProvider(
            name=provider_name,
            endpoint=base_url,
            provider_type="openai",
            api_key=api_key,
        )
    else:
        # Local NIM (default) - use environment variables
        base_url = NIM_BASE_URL
        model_name = NIM_MODEL
        provider_name = "local-nim"
        
        model_provider = ModelProvider(
            name=provider_name,
            endpoint=base_url,
            provider_type="openai",
            api_key="not-needed",
        )

    model_config = ModelConfig(
        alias="intel-gen",
        model=model_name,
        provider=provider_name,
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.8,
            max_tokens=4096,
        ),
    )

    designer = DataDesigner(
        model_providers=[model_provider],
        secret_resolver=PlaintextResolver(),
    )

    return designer, model_config, base_url, model_name


def llm_call(base_url: str, model: str, prompt: str, api_key: str = "", max_tokens: int = 4096) -> str:
    """Direct LLM call via litellm."""
    response = litellm.completion(
        model=f"openai/{model}",
        api_base=base_url,
        api_key=api_key or "not-needed",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# =============================================================================
# PASS 1: SCENARIO BACKBONE GENERATION (using LLMStructuredColumnConfig)
# =============================================================================

def generate_scenario_backbone(designer: DataDesigner, model_config: ModelConfig,
                                params: ScenarioParams) -> ScenarioBackbone:
    """
    PASS 1: Generate consistent scenario facts using LLMStructuredColumnConfig.
    This uses Data Designer's structured output to guarantee schema compliance.
    All input reports reference these facts for correlation.
    """
    
    # Build config with model
    builder = DataDesignerConfigBuilder(model_configs=[model_config])
    
    # Create seed data with scenario context (columns available for Jinja2 templates)
    seed_df = pd.DataFrame([{
        "scenario_type": params.scenario_type,
        "scenario_description": SCENARIO_PROFILES[params.scenario_type]['description'],
        "theater_location": params.theater_location,
        "escalation_level": params.escalation_level,
        "reporting_unit": params.reporting_unit,
        "dtg": params.dtg,
        "reporting_period": f"{params.reporting_period_start} to {params.reporting_period_end}",
        "include_missiles": str(params.include_missile_warning),
    }])
    builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
    
    # Add structured backbone generation column
    builder.add_column(LLMStructuredColumnConfig(
        name="backbone",
        model_alias=model_config.alias,
        output_format=ScenarioBackbone,
        prompt="""Generate consistent facts for a military intelligence scenario in the INDOPACOM theater.
All subsequent intelligence reports will reference these exact facts for consistency.

Scenario Context:
- Type: {{ scenario_type }} - {{ scenario_description }}
- Location: {{ theater_location }}
- Escalation Level: {{ escalation_level }}
- Reporting Unit: {{ reporting_unit }}
- Report DTG: {{ dtg }}
- Reporting Period: {{ reporting_period }}
- Include missiles: {{ include_missiles }}

Generate realistic military values appropriate for this scenario type and escalation level.
For timestamps, use DDHHMMz format (e.g., 221530z).
For grid references, use degrees format (e.g., 25.5°N 122.3°E).
If include_missiles is False, set missile_type to 'none' and missile_count to 0.""",
    ))
    
    # Generate using Data Designer
    try:
        result = designer.preview(builder, num_records=1)
        row = result.dataset.iloc[0]
        
        # Extract the structured backbone - it should already be a dict or ScenarioBackbone
        backbone_data = row['backbone']
        
        # Handle different return types from Data Designer
        if isinstance(backbone_data, ScenarioBackbone):
            return backbone_data
        elif isinstance(backbone_data, dict):
            return ScenarioBackbone(**backbone_data)
        elif isinstance(backbone_data, str):
            # Parse JSON string if needed
            data = json.loads(backbone_data)
            return ScenarioBackbone(**data)
        else:
            raise ValueError(f"Unexpected backbone type: {type(backbone_data)}")
            
    except Exception as e:
        print(f"    Warning: LLMStructuredColumnConfig failed ({e}), using defaults")
        # Return defaults if structured generation fails
        return ScenarioBackbone(
            primary_event_dtg=params.dtg.split()[0],
            secondary_event_dtg=params.dtg.split()[0],
            isr_observation_dtg=params.dtg.split()[0],
            adversary_unit="PLA Navy East Sea Fleet",
            adversary_vessel_count=8,
            adversary_aircraft_count=12,
            adversary_formation_location=f"15nm east of {params.theater_location}",
            adversary_grid_ref="25.0°N 122.0°E",
            adversary_activity="conducting coordinated maritime operations",
            missile_type="MRBM" if params.include_missile_warning else "none",
            missile_launch_location="coastal launch sites",
            missile_trajectory="eastbound toward Taiwan Strait",
            missile_count=4 if params.include_missile_warning else 0,
            friendly_arg_name="ARG-71",
            friendly_air_wing="CVW-5",
            friendly_readiness_pct=95,
            friendly_munitions_status="98% stockpiles; 2% shortfall in PGMs",
            friendly_personnel_count=15000,
            friendly_personnel_status="94% strength",
            cyber_target="regional communications",
            cyber_method="DDoS and GPS spoofing",
            ew_effect="GPS interference in coastal AO",
            partner_nation="Japan",
            partner_request="ISR sharing and coordination",
            weather_visibility="FAIR",
            sea_state=3,
            weather_forecast="degrading in 12 hours",
            primary_assessment="Adversary activity elevated; monitoring recommended",
            threat_level="MEDIUM",
            confidence_level="MEDIUM",
        )


# =============================================================================
# SCENARIO PARAMETER GENERATION (all NDD samplers)
# =============================================================================

def generate_scenario_params(designer: DataDesigner, model_config: ModelConfig) -> ScenarioParams:
    """
    Generate scenario parameters using Data Designer samplers.
    
    Uses basic samplers for categorical values, then applies scenario-dependent
    logic in Python to avoid conditional_params complexity issues.
    """
    
    builder = DataDesignerConfigBuilder(model_configs=[model_config])
    
    # Scenario type with weighted probabilities
    builder.add_column(SamplerColumnConfig(
        name="scenario_type",
        sampler_type="category",
        params=CategorySamplerParams(
            values=list(SCENARIO_PROFILES.keys()),
            weights=[0.20, 0.25, 0.15, 0.15, 0.15, 0.10],
        ),
    ))
    
    # Theater location
    builder.add_column(SamplerColumnConfig(
        name="theater_location",
        sampler_type="category",
        params=CategorySamplerParams(values=THEATER_LOCATIONS),
    ))
    
    # Escalation level (weighted toward moderate)
    builder.add_column(SamplerColumnConfig(
        name="escalation_level",
        sampler_type="category",
        params=CategorySamplerParams(
            values=ESCALATION_LEVELS,
            weights=[0.20, 0.30, 0.25, 0.15, 0.10],
        ),
    ))
    
    # Reporting unit
    builder.add_column(SamplerColumnConfig(
        name="reporting_unit",
        sampler_type="category",
        params=CategorySamplerParams(values=REPORTING_UNITS),
    ))
    
    # Sample one record for base parameters
    result = designer.preview(builder, num_records=1)
    row = result.dataset.iloc[0]
    
    scenario_type = row['scenario_type']
    profile = SCENARIO_PROFILES[scenario_type]
    
    # Generate time info using Python random (simpler than DatetimeSampler)
    base_time = datetime.now() - timedelta(days=random.randint(0, 30))
    dtg = generate_dtg(base_time)
    period_start, period_end = generate_reporting_period(base_time)
    
    # Determine report counts based on scenario profile (Python logic)
    total_reports = random.randint(profile["min_reports"], profile["max_reports"])
    num_flash = random.randint(1, min(3, total_reports))
    remaining = max(1, total_reports - num_flash)
    num_isr = random.randint(1, min(3, remaining))
    remaining_after_isr = max(1, remaining - num_isr)
    num_sensor = max(1, min(remaining_after_isr, 4))
    
    return ScenarioParams(
        scenario_type=scenario_type,
        theater_location=row['theater_location'],
        escalation_level=row['escalation_level'],
        reporting_unit=row['reporting_unit'],
        dtg=dtg,
        reporting_period_start=period_start,
        reporting_period_end=period_end,
        num_flash_reports=num_flash,
        num_isr_reports=num_isr,
        num_sensor_reports=num_sensor,
        include_missile_warning=random.random() < profile["missile_prob"],
        include_cyber_report=random.random() < profile["cyber_prob"],
        include_partner_report=random.random() < profile["partner_prob"],
        include_narrative=random.random() < 0.7,
    )


# =============================================================================
# PASS 2: INPUT REPORT GENERATORS (using backbone facts)
# =============================================================================

def generate_flash_report(base_url: str, model: str, api_key: str, 
                          params: ScenarioParams, backbone: ScenarioBackbone, index: int) -> str:
    """Generate FLASH/ALERT TRAFFIC report using backbone facts."""
    prompt = f"""Generate a realistic military FLASH/ALERT TRAFFIC report for an INDOPACOM scenario.

CRITICAL: Use these EXACT facts from the scenario (do not invent different values):
- Adversary Unit: {backbone.adversary_unit}
- Location: {backbone.adversary_formation_location}
- Grid Reference: {backbone.adversary_grid_ref}
- Activity: {backbone.adversary_activity}
- Vessel Count: {backbone.adversary_vessel_count}
- Threat Level: {backbone.threat_level}
- Assessment: {backbone.primary_assessment}

Context:
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- Theater: {params.theater_location}
- Escalation: {params.escalation_level}
- This is flash report {index + 1} of {params.num_flash_reports}

Generate ONLY the report in this exact format (use facts above):

=== FLASH / ALERT TRAFFIC ===
FLASH REPORT
DTG: {backbone.primary_event_dtg if index == 0 else backbone.secondary_event_dtg}
EVENT: [describe event using the adversary facts above]
ASSESSMENT: [use the assessment from above]
IMPACT: {params.theater_location}
CONFIDENCE: {backbone.confidence_level}

---"""
    
    return llm_call(base_url, model, prompt, api_key)


def generate_missile_warning(base_url: str, model: str, api_key: str,
                             params: ScenarioParams, backbone: ScenarioBackbone) -> str:
    """Generate SPACE/MISSILE WARNING report using backbone facts."""
    prompt = f"""Generate a realistic military SPACE/MISSILE WARNING report.

CRITICAL: Use these EXACT facts (do not invent different values):
- Missile Type: {backbone.missile_type}
- Launch Location: {backbone.missile_launch_location}
- Trajectory: {backbone.missile_trajectory}
- Missile Count: {backbone.missile_count}
- Grid Reference: {backbone.adversary_grid_ref}

Context:
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- Theater: {params.theater_location}

Generate ONLY the report in this exact format:

=== SPACE / MISSILE WARNING ===
MISSILE WARNING
Time: {backbone.primary_event_dtg}
Launch Points: {backbone.missile_launch_location} near {backbone.adversary_grid_ref}
Trajectory: {backbone.missile_trajectory}
Estimate: {backbone.missile_type}
Count: {backbone.missile_count}
Sensor: [appropriate sensor - SBIRS/Overhead IR/radar]
CONFIDENCE: {backbone.confidence_level}

---"""
    
    return llm_call(base_url, model, prompt, api_key)


def generate_isr_report(base_url: str, model: str, api_key: str,
                        params: ScenarioParams, backbone: ScenarioBackbone, index: int) -> str:
    """Generate ISR/IMAGERY ANALYST NOTE using backbone facts."""
    prompt = f"""Generate a realistic military ISR/IMAGERY ANALYST NOTE.

CRITICAL: Use these EXACT facts (do not invent different values):
- Adversary Unit: {backbone.adversary_unit}
- Formation Location: {backbone.adversary_formation_location}
- Vessel Count: {backbone.adversary_vessel_count} vessels
- Aircraft Count: {backbone.adversary_aircraft_count} aircraft
- Activity: {backbone.adversary_activity}
- Weather Visibility: {backbone.weather_visibility}
- Sea State: {backbone.sea_state}

Context:
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- Theater: {params.theater_location}
- This is ISR report {index + 1} of {params.num_isr_reports}

Generate ONLY the report in this exact format:

=== ISR / IMAGERY ANALYST NOTE ===
IMAGERY SUMMARY
DTG: {backbone.isr_observation_dtg}
[2-4 lines describing observations of the {backbone.adversary_vessel_count} vessels and {backbone.adversary_aircraft_count} aircraft at {backbone.adversary_formation_location}]
Weather: Visibility {backbone.weather_visibility}, Sea State {backbone.sea_state}
Source: [Commercial satellite + national imagery]
CONFIDENCE: {backbone.confidence_level}

---"""
    
    return llm_call(base_url, model, prompt, api_key)


SENSOR_PLATFORMS = ["MQ-9", "MQ-4C Triton", "P-8A", "EP-3", "RQ-4"]


def sample_sensor_platform(designer: DataDesigner, model_config: ModelConfig) -> str:
    """Sample a sensor platform using NDD CategorySampler."""
    builder = DataDesignerConfigBuilder(model_configs=[model_config])
    builder.add_column(SamplerColumnConfig(
        name="platform",
        sampler_type="category",
        params=CategorySamplerParams(values=SENSOR_PLATFORMS),
    ))
    result = designer.preview(builder, num_records=1)
    return result.dataset.iloc[0]['platform']


def generate_sensor_report(designer: DataDesigner, model_config: ModelConfig,
                           base_url: str, model: str, api_key: str,
                           params: ScenarioParams, backbone: ScenarioBackbone, index: int) -> str:
    """Generate UAV/SENSOR EVENT LOG using backbone facts."""
    # Use NDD sampler for platform selection
    platform = sample_sensor_platform(designer, model_config)
    
    prompt = f"""Generate a realistic military UAV/SENSOR EVENT LOG.

CRITICAL: Use these EXACT facts (do not invent different values):
- Observed Vessels: {backbone.adversary_vessel_count}
- Observed Aircraft: {backbone.adversary_aircraft_count}
- Location: {backbone.adversary_formation_location}
- Adversary Activity: {backbone.adversary_activity}
- Threat Level: {backbone.threat_level}

Context:
- Platform: {platform}
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- This is sensor report {index + 1} of {params.num_sensor_reports}

Generate ONLY the report in this exact format:

=== UAV / SENSOR EVENT LOG ===
SENSOR EVENT LOG
Platform: {platform}
Time: {backbone.isr_observation_dtg}–{backbone.secondary_event_dtg}
Event: Maritime surface tracks at {backbone.adversary_formation_location}
Count: {backbone.adversary_vessel_count} vessels, {backbone.adversary_aircraft_count} aircraft
Behavior: {backbone.adversary_activity}
Threat Level: {backbone.threat_level}

---"""
    
    return llm_call(base_url, model, prompt, api_key)


def generate_air_track_summary(base_url: str, model: str, api_key: str,
                               params: ScenarioParams, backbone: ScenarioBackbone) -> str:
    """Generate AIR DOMAIN TRACK SUMMARY using backbone facts."""
    prompt = f"""Generate a realistic military AIR DOMAIN TRACK SUMMARY.

CRITICAL: Use these EXACT facts (do not invent different values):
- Aircraft Count: {backbone.adversary_aircraft_count} aircraft observed
- Adversary Unit: {backbone.adversary_unit}
- Location: {backbone.adversary_formation_location}
- Activity Pattern: {backbone.adversary_activity}
- Escalation Level: {params.escalation_level}

Context:
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- Theater: {params.theater_location}

Generate ONLY the report in this exact format:

=== AIR DOMAIN TRACK SUMMARY ===
AIR TRACK SUMMARY
{backbone.adversary_aircraft_count} aircraft from {backbone.adversary_unit} operating near {backbone.adversary_formation_location}.
[2-3 lines describing patterns - CAP, escort, or patrol based on the activity]
Track density: [elevated/normal/reduced] vs baseline for {params.escalation_level} escalation.

---"""
    
    return llm_call(base_url, model, prompt, api_key)


def generate_cyber_report(base_url: str, model: str, api_key: str,
                          params: ScenarioParams, backbone: ScenarioBackbone) -> str:
    """Generate CYBER/EW STATUS report using backbone facts."""
    prompt = f"""Generate a realistic military CYBER/EW STATUS report.

CRITICAL: Use these EXACT facts (do not invent different values):
- Cyber Target: {backbone.cyber_target}
- Cyber Method: {backbone.cyber_method}
- EW Effect: {backbone.ew_effect}
- Location: {params.theater_location}

Context:
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- Escalation: {params.escalation_level}

Generate ONLY the report in this exact format (no extra text):

=== CYBER / EW STATUS ===
CYBER STATUS
[2-4 lines describing cyber activity, DDoS, GPS interference, EW events]
CONFIDENCE: [LOW/MED/HIGH]

---"""
    
    return llm_call(base_url, model, prompt, api_key)


def generate_friendly_force_status(base_url: str, model: str, api_key: str,
                                   params: ScenarioParams, backbone: ScenarioBackbone) -> str:
    """Generate FRIENDLY FORCE STATUS report using backbone facts."""
    prompt = f"""Generate a realistic military FRIENDLY FORCE STATUS report.

CRITICAL: Use these EXACT facts (do not invent different values):
- ARG Designation: {backbone.friendly_arg_name}
- Air Wing: {backbone.friendly_air_wing}
- Readiness: {backbone.friendly_readiness_pct}%
- Munitions Status: {backbone.friendly_munitions_status}
- Personnel Count: {backbone.friendly_personnel_count}
- Personnel Status: {backbone.friendly_personnel_status}
- Reporting Unit: {params.reporting_unit}

Context:
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- Escalation: {params.escalation_level}

Generate ONLY the report in this exact format:

=== FRIENDLY FORCE STATUS ===
FORCE STATUS
{backbone.friendly_arg_name}: Ready
{backbone.friendly_air_wing}: {backbone.friendly_readiness_pct}% readiness
Munitions: {backbone.friendly_munitions_status}
Personnel: {backbone.friendly_personnel_count} troops; {backbone.friendly_personnel_status}

---"""
    
    return llm_call(base_url, model, prompt, api_key)


def generate_partner_report(base_url: str, model: str, api_key: str,
                            params: ScenarioParams, backbone: ScenarioBackbone) -> str:
    """Generate PARTNER/ALLIED REPORT using backbone facts."""
    prompt = f"""Generate a realistic military PARTNER/ALLIED REPORT (LNO REPORT).

CRITICAL: Use these EXACT facts (do not invent different values):
- Partner Nation: {backbone.partner_nation}
- Partner Request: {backbone.partner_request}
- Adversary Activity: {backbone.adversary_activity}
- Location: {backbone.adversary_formation_location}

Context:
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- Theater: {params.theater_location}
- Escalation: {params.escalation_level}

Generate ONLY the report in this exact format:

=== PARTNER / ALLIED REPORT ===
LNO REPORT
Partner: {backbone.partner_nation}
{backbone.partner_nation} reports observing {backbone.adversary_activity} near {backbone.adversary_formation_location}.
Request: {backbone.partner_request}
[1-2 additional lines of coordination details]

---"""
    
    return llm_call(base_url, model, prompt, api_key)


def generate_metoc_report(base_url: str, model: str, api_key: str,
                          params: ScenarioParams, backbone: ScenarioBackbone) -> str:
    """Generate METOC/ENVIRONMENT report using backbone facts."""
    # Determine operational impact based on weather and sea state
    if backbone.sea_state >= 5 or backbone.weather_visibility == "POOR":
        op_impact = "SIGNIFICANT"
    elif backbone.sea_state >= 3 or backbone.weather_visibility == "FAIR":
        op_impact = "MODERATE"
    else:
        op_impact = "MINIMAL"
    
    prompt = f"""Generate a realistic military METOC/ENVIRONMENT report.

CRITICAL: Use these EXACT facts (do not invent different values):
- Visibility: {backbone.weather_visibility}
- Sea State: {backbone.sea_state}
- Forecast: {backbone.weather_forecast}
- Location: {params.theater_location}

Generate ONLY the report in this exact format:

=== METOC / ENVIRONMENT ===
METOC UPDATE
Visibility: {backbone.weather_visibility}
Sea State: {backbone.sea_state}
Forecast: {backbone.weather_forecast}
Operational Impact: {op_impact}

---"""
    
    return llm_call(base_url, model, prompt, api_key)


def generate_osint_report(base_url: str, model: str, api_key: str,
                          params: ScenarioParams, backbone: ScenarioBackbone) -> str:
    """Generate OSINT/CIVIL SIGNALS report using backbone facts."""
    prompt = f"""Generate a realistic military OSINT/CIVIL SIGNALS report.

CRITICAL: Reference these facts in OSINT context (social media, news, shipping):
- Adversary Activity: {backbone.adversary_activity}
- Location: {backbone.adversary_formation_location}
- Vessel Count: {backbone.adversary_vessel_count} vessels
- Theater: {params.theater_location}
- Cyber Activity: {backbone.cyber_method}

Context:
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- Escalation: {params.escalation_level}

Generate ONLY the report in this exact format:

=== OSINT / CIVIL SIGNALS ===
OSINT
Social media reports [activity consistent with {backbone.adversary_activity}] near {params.theater_location}.
Commercial shipping AIS shows {backbone.adversary_vessel_count} vessels in unusual patterns near {backbone.adversary_formation_location}.
[1-2 additional OSINT indicators]
Assessment: Context only; not confirmed by classified sources.

---"""
    
    return llm_call(base_url, model, prompt, api_key)


def generate_narrative_summary(base_url: str, model: str, api_key: str,
                               params: ScenarioParams, backbone: ScenarioBackbone, 
                               all_reports: list[str]) -> str:
    """Generate longer narrative intelligence summary using backbone facts."""
    reports_text = "\n\n".join(all_reports[:5])  # Use first 5 reports as context
    
    prompt = f"""You are a senior intelligence analyst. Write a 2-3 paragraph narrative intelligence summary.

CRITICAL: Ensure your narrative references these consistent facts:
- Adversary Unit: {backbone.adversary_unit}
- Adversary Location: {backbone.adversary_formation_location}
- Vessel Count: {backbone.adversary_vessel_count}
- Aircraft Count: {backbone.adversary_aircraft_count}
- Activity: {backbone.adversary_activity}
- Threat Level: {backbone.threat_level}
- Primary Assessment: {backbone.primary_assessment}
- Friendly Forces: {backbone.friendly_arg_name}, {backbone.friendly_air_wing}

Context:
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- Location: {params.theater_location}
- Escalation Level: {params.escalation_level}
- Reporting Period: {params.reporting_period_start} to {params.reporting_period_end}

Recent Reports:
{reports_text}

Write a professional intelligence narrative (2-3 paragraphs) that:
1. Synthesizes the key events and patterns
2. Provides analytical assessment
3. Notes information gaps or uncertainties

Output ONLY the narrative text in markdown format, starting with:

=== INTELLIGENCE NARRATIVE SUMMARY ===
"""
    
    return llm_call(base_url, model, prompt, api_key, max_tokens=2048)


# =============================================================================
# SITREP OUTPUT GENERATOR (using LLMStructuredColumnConfig)
# =============================================================================

def format_sitrep_to_markdown(sitrep: SITREPStructured, params: ScenarioParams) -> str:
    """Format structured SITREP to markdown output."""
    
    # Format bullet points with proper markdown
    bluf_bullets = "\n".join(f"- {bullet}" for bullet in sitrep.bluf.bullet_points)
    friendly_status = "\n".join(f"- {status}" for status in sitrep.friendly_forces.unit_status)
    adversary_obs = "\n".join(f"- {obs}" for obs in sitrep.adversary_activity.observations)
    current_ops = "\n".join(f"- {op}" for op in sitrep.operations.current_ops)
    intel_findings = "\n".join(f"- {finding}" for finding in sitrep.intelligence_assessment.key_findings)
    risk_bullets = "\n".join(f"- {risk}" for risk in sitrep.risks.risks)
    
    return f"""SITUATIONAL REPORT (SITREP)

Unit: {params.reporting_unit}
Report Type: Daily SITREP
DTG: {params.dtg}
Classification: SECRET//NOFORN
Reporting Period: {params.reporting_period_start}–{params.reporting_period_end}

---

1. BLUF (Bottom Line Up Front)
{bluf_bullets}

Confidence: {sitrep.bluf.confidence}
Probability assessment: {sitrep.bluf.probability_assessment}

---

2. FRIENDLY FORCES
{friendly_status}

Assessment: {sitrep.friendly_forces.assessment}

---

3. ADVERSARY ACTIVITY
{adversary_obs}

Assessment: {sitrep.adversary_activity.assessment}

---

4. OPERATIONS
{current_ops}

Planned: {sitrep.operations.planned_ops}

---

5. LOGISTICS
Supply Status: {sitrep.logistics.supply_status}
Shortfalls: {sitrep.logistics.shortfalls}
Resupply ETA: {sitrep.logistics.resupply_eta}

---

6. INTELLIGENCE ASSESSMENT
{intel_findings}

Information Gaps: {sitrep.intelligence_assessment.information_gaps}

Analyst judgment: {sitrep.intelligence_assessment.analyst_judgment}

---

7. RISKS / WATCH ITEMS
{risk_bullets}

---

8. COMMANDER'S COMMENTS
{sitrep.commander_comments.guidance}

Priority Focus: {sitrep.commander_comments.priority_focus}

---"""


def generate_sitrep(designer: DataDesigner, model_config: ModelConfig,
                    params: ScenarioParams, backbone: ScenarioBackbone,
                    all_inputs: list[str]) -> str:
    """
    Generate consolidated 8-section SITREP using LLMStructuredColumnConfig.
    Returns formatted markdown string.
    """
    
    inputs_text = "\n\n".join(all_inputs)
    
    # Build config with model
    builder = DataDesignerConfigBuilder(model_configs=[model_config])
    
    # Create seed data with all context (columns available for Jinja2 templates)
    seed_df = pd.DataFrame([{
        "reporting_unit": params.reporting_unit,
        "dtg": params.dtg,
        "reporting_period": f"{params.reporting_period_start} to {params.reporting_period_end}",
        "theater_location": params.theater_location,
        "escalation_level": params.escalation_level,
        "scenario_type": params.scenario_type,
        # Backbone facts
        "adversary_unit": backbone.adversary_unit,
        "adversary_location": backbone.adversary_formation_location,
        "vessel_count": str(backbone.adversary_vessel_count),
        "aircraft_count": str(backbone.adversary_aircraft_count),
        "threat_level": backbone.threat_level,
        "friendly_arg": backbone.friendly_arg_name,
        "friendly_air_wing": backbone.friendly_air_wing,
        "readiness_pct": str(backbone.friendly_readiness_pct),
        "munitions_status": backbone.friendly_munitions_status,
        "intelligence_inputs": inputs_text,
    }])
    builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
    
    # Add structured SITREP generation column
    builder.add_column(LLMStructuredColumnConfig(
        name="sitrep",
        model_alias=model_config.alias,
        output_format=SITREPStructured,
        prompt="""You are a military operations officer preparing a Situation Report (SITREP).

Based on the intelligence inputs and scenario context, generate a structured SITREP.

=== SCENARIO CONTEXT ===
- Reporting Unit: {{ reporting_unit }}
- DTG: {{ dtg }}
- Reporting Period: {{ reporting_period }}
- Theater: {{ theater_location }}
- Escalation Level: {{ escalation_level }}

=== KEY FACTS (from scenario backbone) ===
- Adversary: {{ adversary_unit }} at {{ adversary_location }}
- Vessels: {{ vessel_count }}, Aircraft: {{ aircraft_count }}
- Threat Level: {{ threat_level }}
- Friendly Forces: {{ friendly_arg }}, {{ friendly_air_wing }}
- Readiness: {{ readiness_pct }}%
- Munitions: {{ munitions_status }}

=== INTELLIGENCE INPUTS ===
{{ intelligence_inputs }}

Generate a comprehensive SITREP with all 8 sections. Be specific and reference the facts above.
For bullet points, provide 2-4 concise items per section.
Ensure assessments are consistent with the threat level and escalation.""",
    ))
    
    # Generate using Data Designer
    try:
        result = designer.preview(builder, num_records=1)
        row = result.dataset.iloc[0]
        
        # Extract the structured SITREP
        sitrep_data = row['sitrep']
        
        # Handle different return types
        if isinstance(sitrep_data, SITREPStructured):
            sitrep = sitrep_data
        elif isinstance(sitrep_data, dict):
            sitrep = SITREPStructured(**sitrep_data)
        elif isinstance(sitrep_data, str):
            data = json.loads(sitrep_data)
            sitrep = SITREPStructured(**data)
        else:
            raise ValueError(f"Unexpected SITREP type: {type(sitrep_data)}")
        
        # Format to markdown
        return format_sitrep_to_markdown(sitrep, params)
        
    except Exception as e:
        print(f"    Warning: LLMStructuredColumnConfig failed ({e}), falling back to raw LLM")
        # Fallback to raw LLM call if structured generation fails
        return generate_sitrep_fallback(params, all_inputs)


def generate_sitrep_fallback(params: ScenarioParams, all_inputs: list[str]) -> str:
    """Fallback SITREP generation using raw LLM call."""
    from litellm import completion
    
    inputs_text = "\n\n".join(all_inputs)
    
    prompt = f"""You are a military operations officer preparing a Situation Report (SITREP).

Based on the following intelligence inputs, generate a consolidated SITREP following the exact template below.

=== INTELLIGENCE INPUTS ===
{inputs_text}

=== SITREP TEMPLATE (fill this out) ===

SITUATIONAL REPORT (SITREP)

Unit: {params.reporting_unit}
Report Type: Daily SITREP
DTG: {params.dtg}
Classification: SECRET//NOFORN
Reporting Period: {params.reporting_period_start}–{params.reporting_period_end}

---

1. BLUF (Bottom Line Up Front)
[Write 2-3 bullet points summarizing the most critical information]

Confidence: [High/Medium/Low]
Probability assessment: [percentage breakdown of likely scenarios]

---

2. FRIENDLY FORCES
[Summarize friendly force status from inputs]

Assessment: [Overall assessment]

---

3. ADVERSARY ACTIVITY
[Summarize adversary actions from inputs]

Assessment: [Threat assessment]

---

4. OPERATIONS
[Current operations summary]

---

5. LOGISTICS
[Logistics/supply status]

---

6. INTELLIGENCE ASSESSMENT
[Synthesized intelligence assessment]

Analyst judgment: [Recommendation]

---

7. RISKS / WATCH ITEMS
[2-4 bullet points of key risks]

---

8. COMMANDER'S COMMENTS
[Brief commander guidance]

---

Generate ONLY the filled SITREP, no additional commentary."""

    response = completion(
        model=f"openai/{NIM_MODEL}",
        api_base=NIM_BASE_URL,
        api_key="not-needed",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=4096,
    )
    return response.choices[0].message.content


# =============================================================================
# MAIN GENERATION PIPELINE (TWO-PASS DESIGN)
# =============================================================================

def generate_scenario(designer: DataDesigner, model_config: ModelConfig,
                      base_url: str, model: str, api_key: str,
                      scenario_num: int) -> Path:
    """
    Generate a complete scenario with inputs and output.
    
    TWO-PASS DESIGN:
    - Pass 1: Generate scenario backbone (consistent facts)
    - Pass 2: Generate all reports using those facts
    """
    
    print(f"\n{'='*60}")
    print(f"SCENARIO {scenario_num}")
    print('='*60)
    
    # Generate scenario parameters
    params = generate_scenario_params(designer, model_config)
    
    print(f"  Type: {params.scenario_type}")
    print(f"  Location: {params.theater_location}")
    print(f"  Escalation: {params.escalation_level}")
    print(f"  Unit: {params.reporting_unit}")
    print(f"  DTG: {params.dtg}")
    
    # Create scenario directory
    scenario_id = f"scenario_{scenario_num:03d}_{params.scenario_type}"
    scenario_dir = OUTPUT_DIR / scenario_id
    inputs_dir = scenario_dir / "inputs"
    output_dir = scenario_dir / "output"
    
    inputs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PASS 1: Generate scenario backbone (consistent facts)
    # =========================================================================
    print(f"\n  PASS 1: Generating scenario backbone...")
    backbone = generate_scenario_backbone(designer, model_config, params)
    print(f"    Adversary: {backbone.adversary_unit}")
    print(f"    Location: {backbone.adversary_formation_location}")
    print(f"    Vessels: {backbone.adversary_vessel_count}, Aircraft: {backbone.adversary_aircraft_count}")
    print(f"    Friendly: {backbone.friendly_arg_name}, {backbone.friendly_air_wing}")
    print(f"    Threat Level: {backbone.threat_level}")
    
    # Save backbone to file for reference
    backbone_data = {
        "primary_event_dtg": backbone.primary_event_dtg,
        "secondary_event_dtg": backbone.secondary_event_dtg,
        "adversary_unit": backbone.adversary_unit,
        "adversary_vessel_count": backbone.adversary_vessel_count,
        "adversary_aircraft_count": backbone.adversary_aircraft_count,
        "adversary_formation_location": backbone.adversary_formation_location,
        "adversary_grid_ref": backbone.adversary_grid_ref,
        "adversary_activity": backbone.adversary_activity,
        "missile_type": backbone.missile_type,
        "missile_count": backbone.missile_count,
        "friendly_arg_name": backbone.friendly_arg_name,
        "friendly_air_wing": backbone.friendly_air_wing,
        "friendly_readiness_pct": backbone.friendly_readiness_pct,
        "friendly_munitions_status": backbone.friendly_munitions_status,
        "friendly_personnel_count": backbone.friendly_personnel_count,
        "partner_nation": backbone.partner_nation,
        "weather_visibility": backbone.weather_visibility,
        "sea_state": backbone.sea_state,
        "threat_level": backbone.threat_level,
        "confidence_level": backbone.confidence_level,
        "primary_assessment": backbone.primary_assessment,
    }
    backbone_path = scenario_dir / "backbone.json"
    backbone_path.write_text(json.dumps(backbone_data, indent=2))
    
    # =========================================================================
    # PASS 2: Generate input reports using backbone facts
    # =========================================================================
    all_reports = []
    report_num = 1
    
    print(f"\n  PASS 2: Generating correlated inputs...")
    
    # FLASH reports
    for i in range(params.num_flash_reports):
        print(f"    [{report_num}] FLASH report {i+1}...", end=" ", flush=True)
        report = generate_flash_report(base_url, model, api_key, params, backbone, i)
        all_reports.append(report)
        filepath = inputs_dir / f"{report_num:02d}_flash_{i+1}.md"
        filepath.write_text(report)
        print("done")
        report_num += 1
    
    # Missile warning (if applicable)
    if params.include_missile_warning:
        print(f"    [{report_num}] Missile warning...", end=" ", flush=True)
        report = generate_missile_warning(base_url, model, api_key, params, backbone)
        all_reports.append(report)
        filepath = inputs_dir / f"{report_num:02d}_missile_warning.md"
        filepath.write_text(report)
        print("done")
        report_num += 1
    
    # ISR reports
    for i in range(params.num_isr_reports):
        print(f"    [{report_num}] ISR report {i+1}...", end=" ", flush=True)
        report = generate_isr_report(base_url, model, api_key, params, backbone, i)
        all_reports.append(report)
        filepath = inputs_dir / f"{report_num:02d}_isr_{i+1}.md"
        filepath.write_text(report)
        print("done")
        report_num += 1
    
    # Sensor reports
    for i in range(params.num_sensor_reports):
        print(f"    [{report_num}] Sensor report {i+1}...", end=" ", flush=True)
        report = generate_sensor_report(designer, model_config, base_url, model, api_key, params, backbone, i)
        all_reports.append(report)
        filepath = inputs_dir / f"{report_num:02d}_sensor_{i+1}.md"
        filepath.write_text(report)
        print("done")
        report_num += 1
    
    # Air track summary
    print(f"    [{report_num}] Air track summary...", end=" ", flush=True)
    report = generate_air_track_summary(base_url, model, api_key, params, backbone)
    all_reports.append(report)
    filepath = inputs_dir / f"{report_num:02d}_air_track.md"
    filepath.write_text(report)
    print("done")
    report_num += 1
    
    # Cyber report (if applicable)
    if params.include_cyber_report:
        print(f"    [{report_num}] Cyber/EW report...", end=" ", flush=True)
        report = generate_cyber_report(base_url, model, api_key, params, backbone)
        all_reports.append(report)
        filepath = inputs_dir / f"{report_num:02d}_cyber_ew.md"
        filepath.write_text(report)
        print("done")
        report_num += 1
    
    # Friendly force status
    print(f"    [{report_num}] Friendly force status...", end=" ", flush=True)
    report = generate_friendly_force_status(base_url, model, api_key, params, backbone)
    all_reports.append(report)
    filepath = inputs_dir / f"{report_num:02d}_friendly_force.md"
    filepath.write_text(report)
    print("done")
    report_num += 1
    
    # Partner report (if applicable)
    if params.include_partner_report:
        print(f"    [{report_num}] Partner/Allied report...", end=" ", flush=True)
        report = generate_partner_report(base_url, model, api_key, params, backbone)
        all_reports.append(report)
        filepath = inputs_dir / f"{report_num:02d}_partner.md"
        filepath.write_text(report)
        print("done")
        report_num += 1
    
    # METOC report
    print(f"    [{report_num}] METOC report...", end=" ", flush=True)
    report = generate_metoc_report(base_url, model, api_key, params, backbone)
    all_reports.append(report)
    filepath = inputs_dir / f"{report_num:02d}_metoc.md"
    filepath.write_text(report)
    print("done")
    report_num += 1
    
    # OSINT report
    print(f"    [{report_num}] OSINT report...", end=" ", flush=True)
    report = generate_osint_report(base_url, model, api_key, params, backbone)
    all_reports.append(report)
    filepath = inputs_dir / f"{report_num:02d}_osint.md"
    filepath.write_text(report)
    print("done")
    report_num += 1
    
    # Narrative summary (if applicable)
    if params.include_narrative:
        print(f"    [{report_num}] Narrative summary...", end=" ", flush=True)
        report = generate_narrative_summary(base_url, model, api_key, params, backbone, all_reports)
        all_reports.append(report)
        filepath = inputs_dir / f"{report_num:02d}_narrative.md"
        filepath.write_text(report)
        print("done")
        report_num += 1
    
    print(f"\n  Total input reports: {len(all_reports)}")
    
    # Generate consolidated SITREP using LLMStructuredColumnConfig
    print(f"\n  Generating consolidated SITREP (structured)...", end=" ", flush=True)
    sitrep = generate_sitrep(designer, model_config, params, backbone, all_reports)
    sitrep_path = output_dir / "sitrep.md"
    sitrep_path.write_text(sitrep)
    print("done")
    
    # =========================================================================
    # Save training data as JSONL (input/output format for fine-tuning)
    # =========================================================================
    training_record = {
        "input": "\n\n".join(all_reports),  # Concatenated intel inputs
        "output": sitrep,                     # The consolidated SITREP
        "scenario_id": scenario_id,
        "scenario_type": params.scenario_type,
        "theater_location": params.theater_location,
        "escalation_level": params.escalation_level,
    }
    
    # Save individual scenario training sample
    jsonl_path = scenario_dir / "training_sample.jsonl"
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps(training_record) + '\n')
    print(f"  Saved training JSONL: {jsonl_path.name}")
    
    # Save scenario metadata
    metadata = {
        "scenario_id": scenario_id,
        "scenario_type": params.scenario_type,
        "scenario_description": SCENARIO_PROFILES[params.scenario_type]['description'],
        "theater_location": params.theater_location,
        "escalation_level": params.escalation_level,
        "reporting_unit": params.reporting_unit,
        "dtg": params.dtg,
        "reporting_period": f"{params.reporting_period_start}–{params.reporting_period_end}",
        "num_input_reports": len(all_reports),
        "input_files": [f.name for f in sorted(inputs_dir.glob("*.md"))],
    }
    metadata_path = scenario_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    
    print(f"\n  Saved to: {scenario_dir}")
    
    return scenario_dir, training_record


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic INDOPACOM SITREP training data"
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of scenarios to generate (default: 1)"
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use NVIDIA cloud API instead of local NIM (requires valid NVIDIA_API_KEY)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("INDOPACOM SITREP Synthetic Data Generator v2")
    print("="*60)
    print(f"  Scenarios to generate: {args.count}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Determine which endpoint to use
    if args.cloud:
        print("  Using NVIDIA Cloud API")
    else:
        print(f"  Using local NIM: {NIM_BASE_URL}")
        print(f"  Model: {NIM_MODEL}")
    
    # Create Data Designer
    designer, model_config, base_url, model = create_designer(use_nvidia_api=args.cloud)
    api_key = os.environ.get("NVIDIA_API_KEY", "not-needed")
    
    print(f"  Model: {model}")
    print(f"  Endpoint: {base_url}")
    
    # Generate scenarios
    generated = []
    all_training_records = []
    
    for i in range(1, args.count + 1):
        try:
            scenario_dir, training_record = generate_scenario(
                designer, model_config, base_url, model, api_key, i
            )
            generated.append(scenario_dir)
            all_training_records.append(training_record)
        except Exception as e:
            print(f"\n  ERROR generating scenario {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save master training JSONL file with all scenarios
    if all_training_records:
        master_jsonl_path = OUTPUT_DIR / "training_data.jsonl"
        with open(master_jsonl_path, 'w') as f:
            for record in all_training_records:
                # Write only input/output for training (strip metadata)
                training_only = {
                    "input": record["input"],
                    "output": record["output"],
                }
                f.write(json.dumps(training_only) + '\n')
        print(f"\n  Master training file: {master_jsonl_path}")
        print(f"  Total training samples: {len(all_training_records)}")
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"  Successfully generated: {len(generated)}/{args.count} scenarios")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    if generated:
        print("\n  Generated scenarios:")
        for path in generated:
            print(f"    - {path.name}")
        print(f"\n  Training data: {OUTPUT_DIR / 'training_data.jsonl'}")


if __name__ == "__main__":
    main()
