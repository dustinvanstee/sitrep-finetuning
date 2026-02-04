#!/usr/bin/env python3
"""
Generate supplementary military document training data to augment SITREP training.

This script generates additional document types that share the same scenario backbone
structure as SITREPs, creating a diverse training mix for better generalization.

Document Types Generated:
- INTSUMs (Intelligence Summaries) - Close cousin to SITREPs
- SPOTREPs (Spot Reports) - Short tactical updates (SALUTE format)
- Contact Reports - Immediate enemy contact reports
- OPORDs (Operations Orders) - 5-paragraph format
- FRAGOs (Fragmentary Orders) - Updates to existing orders
- Watch Officer Logs - Chronological event tracking

Usage:
    cd /home/dvanstee/projects/2026-01-nt3-sitrep
    DataDesigner/.venv/bin/python scripts/1_generate_supplementary_data.py --count 50 --type intsum
    DataDesigner/.venv/bin/python scripts/1_generate_supplementary_data.py --count 30 --type spotrep
    DataDesigner/.venv/bin/python scripts/1_generate_supplementary_data.py --count 30 --type contact
    DataDesigner/.venv/bin/python scripts/1_generate_supplementary_data.py --count 20 --type opord
    DataDesigner/.venv/bin/python scripts/1_generate_supplementary_data.py --count 20 --type frago
    DataDesigner/.venv/bin/python scripts/1_generate_supplementary_data.py --count 20 --type watchlog
    DataDesigner/.venv/bin/python scripts/1_generate_supplementary_data.py --count 50 --type all  # generates mix
"""
import os
import json
import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Project setup - reuse existing infrastructure
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

NIM_BASE_URL = os.environ.get("NIM_LOCAL_BASE_URL", "http://localhost:8000/v1")
NIM_MODEL = os.environ.get("NIM_MODEL", "nvidia/nemotron-3-nano")

DATA_DESIGNER_DIR = PROJECT_ROOT / ".data-designer"
os.environ.setdefault("DATA_DESIGNER_HOME", str(DATA_DESIGNER_DIR))

# Import Data Designer
from data_designer.interface.data_designer import DataDesigner
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig, ModelProvider, ChatCompletionInferenceParams
from data_designer.config.column_configs import SamplerColumnConfig, LLMStructuredColumnConfig
from data_designer.config.seed_source import DataFrameSeedSource
from data_designer.config.sampler_params import CategorySamplerParams
from data_designer.engine.secret_resolver import PlaintextResolver

# =============================================================================
# SHARED SCENARIO INFRASTRUCTURE (copied from main generator for independence)
# =============================================================================

import requests
from dataclasses import dataclass

def check_nim_connection(base_url: str) -> bool:
    """Check if local NIM is accessible."""
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def generate_dtg(base_time: datetime) -> str:
    """Generate military Date-Time Group."""
    return base_time.strftime("%d%H%MZ %b %y").upper()

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
    reporting_period_hours: int
    classification: str
    num_flash_reports: int
    num_isr_reports: int
    num_sensor_reports: int
    include_missile_warning: bool
    include_cyber_report: bool
    include_partner_report: bool
    include_narrative: bool
    include_humint: bool
    include_sigint: bool
    include_sub_contact: bool
    weather_pattern: dict
    primary_adversary_nation: str

class ScenarioBackbone(BaseModel):
    """Scenario facts for consistent cross-report correlation."""
    primary_event_dtg: str = Field(description="Primary event DTG")
    secondary_event_dtg: str = Field(description="Secondary event DTG")
    isr_observation_dtg: str = Field(description="ISR collection DTG")
    adversary_unit: str = Field(description="Adversary unit designation")
    adversary_vessel_count: int = Field(ge=1, le=50)
    adversary_aircraft_count: int = Field(ge=0, le=100)
    adversary_formation_location: str
    adversary_grid_ref: str
    adversary_activity: str
    missile_type: Literal["SRBM", "MRBM", "IRBM", "cruise", "none"] = "none"
    missile_launch_location: str = ""
    missile_trajectory: str = ""
    missile_count: int = Field(ge=0, le=20, default=0)
    friendly_arg_name: str
    friendly_air_wing: str
    friendly_readiness_pct: int = Field(ge=70, le=100)
    friendly_munitions_status: str
    friendly_personnel_count: int = Field(ge=500, le=50000)
    friendly_personnel_status: str = ""
    cyber_target: str = ""
    cyber_method: str = ""
    ew_effect: str = ""
    partner_nation: Literal[
        "Japan", "South Korea", "Australia", "Philippines", "Taiwan",
        "India", "Singapore", "New Zealand", "United Kingdom", "Canada",
        "Thailand", "Vietnam", "Indonesia", "Malaysia"
    ] = "Japan"
    partner_request: str = ""
    weather_visibility: Literal["GOOD", "FAIR", "POOR"] = "FAIR"
    sea_state: int = Field(ge=1, le=6, default=3)
    weather_forecast: str = ""
    primary_assessment: str
    threat_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    confidence_level: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"

# Scenario profiles with report density characteristics
SCENARIO_PROFILES = {
    "taiwan_invasion": {"description": "Full-scale amphibious invasion of Taiwan", "missile_prob": 0.95, "cyber_prob": 0.90, "partner_prob": 0.85},
    "taiwan_blockade": {"description": "Naval and air blockade of Taiwan", "missile_prob": 0.75, "cyber_prob": 0.85, "partner_prob": 0.90},
    "korean_peninsula_tension": {"description": "Elevated tension on Korean peninsula", "missile_prob": 0.70, "cyber_prob": 0.65, "partner_prob": 0.80},
    "dprk_missile_launch": {"description": "North Korean ballistic missile launch", "missile_prob": 0.98, "cyber_prob": 0.55, "partner_prob": 0.85},
    "south_china_sea_standoff": {"description": "Naval standoff near disputed islands", "missile_prob": 0.30, "cyber_prob": 0.50, "partner_prob": 0.70},
    "submarine_tracking": {"description": "Tracking adversary submarine", "missile_prob": 0.15, "cyber_prob": 0.35, "partner_prob": 0.60},
    "island_militarization": {"description": "Militarization of disputed islands", "missile_prob": 0.25, "cyber_prob": 0.45, "partner_prob": 0.75},
    "gray_zone_harassment": {"description": "Maritime militia harassment ops", "missile_prob": 0.08, "cyber_prob": 0.40, "partner_prob": 0.65},
    "fishing_fleet_incursion": {"description": "Fishing fleet EEZ incursion", "missile_prob": 0.05, "cyber_prob": 0.30, "partner_prob": 0.80},
    "vessel_incursion": {"description": "Single vessel territorial incursion", "missile_prob": 0.05, "cyber_prob": 0.20, "partner_prob": 0.40},
    "routine_patrol": {"description": "Routine patrol with minor activity", "missile_prob": 0.02, "cyber_prob": 0.15, "partner_prob": 0.30},
    "exercise_malabar": {"description": "Multilateral naval exercise with QUAD", "missile_prob": 0.08, "cyber_prob": 0.30, "partner_prob": 0.98},
    "exercise_poseidon": {"description": "Joint allied exercise in Philippine Sea", "missile_prob": 0.10, "cyber_prob": 0.25, "partner_prob": 0.95},
    "carrier_strike_ops": {"description": "CSG operations in contested waters", "missile_prob": 0.35, "cyber_prob": 0.60, "partner_prob": 0.75},
    "humanitarian_disaster": {"description": "Typhoon disaster relief ops", "missile_prob": 0.02, "cyber_prob": 0.20, "partner_prob": 0.90},
    "search_and_rescue": {"description": "SAR for downed aircraft", "missile_prob": 0.03, "cyber_prob": 0.15, "partner_prob": 0.70},
}

THEATER_LOCATIONS = [
    "Taiwan Strait", "Taiwan Strait - Northern Approaches", "Taiwan Strait - Southern Approaches",
    "South China Sea - Spratly Islands", "South China Sea - Paracel Islands", "South China Sea - Scarborough Shoal",
    "East China Sea - Senkaku Islands", "East China Sea - ADIZ boundary",
    "Philippine Sea - Western approaches", "Luzon Strait", "Bashi Channel",
    "Korean Peninsula - DMZ", "Korean Peninsula - West Sea", "Korean Peninsula - East Sea",
    "Miyako Strait", "Tsushima Strait", "Malacca Strait - Northern approaches",
    "Sea of Japan - Central", "Okinawa - East China Sea",
    "Guam - Western Pacific", "Andaman Sea", "Bay of Bengal - Eastern approaches",
]

REPORTING_UNITS = [
    "PACFLT", "7th Fleet", "3rd Fleet", "MARFORPAC", "III MEF", "INDOPACOM J2", "INDOPACOM J3",
    "CSG-5 (REAGAN)", "CSG-1 (VINSON)", "CSG-3 (STENNIS)", "CSG-9 (NIMITZ)",
    "CTF 70", "CTF 71", "CTF 72", "CTF 73", "CTF 74", "CTF 75", "CTF 76",
    "ARG/31st MEU", "ARG/11th MEU", "ARG/15th MEU",
    "COMSUBPAC", "CSS-15", "SUBRON-7",
    "PACAF", "5th Air Force", "7th Air Force", "CVW-5", "CVW-2", "CVW-9",
    "USARPAC", "I Corps", "SOCPAC", "CJTF-Japan", "CJTF-Korea",
]

ESCALATION_LEVELS = ["LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"]

ADVERSARY_NAVAL_UNITS = [
    "PLA Navy East Sea Fleet", "PLA Navy South Sea Fleet", "PLA Navy North Sea Fleet",
    "PLAN Carrier Strike Group (Liaoning)", "PLAN Carrier Strike Group (Shandong)",
    "PLAN Submarine Flotilla 32", "PLAN Amphibious Task Force", "PLAN Maritime Militia",
    "China Coast Guard Squadron", "DPRK Navy West Fleet", "DPRK Navy East Fleet",
    "Russian Pacific Fleet", "Russian Pacific Fleet Submarine Squadron",
]

ADVERSARY_AIR_UNITS = [
    "PLAAF Eastern Theater Command", "PLAAF Southern Theater Command", "PLA Navy Aviation",
    "PLAAF 1st Fighter Division", "PLAAF H-6K Bomber Regiment",
    "DPRK Air Force", "Russian Far East Aviation", "Russian Tu-95 Long Range Aviation",
]

WEATHER_PATTERNS = [
    {"name": "Clear", "visibility": "GOOD", "sea_state_range": (1, 3), "forecast": "Stable for 24 hours"},
    {"name": "Monsoon", "visibility": "POOR", "sea_state_range": (4, 6), "forecast": "Heavy rain continuing"},
    {"name": "Fog/Haze", "visibility": "POOR", "sea_state_range": (1, 2), "forecast": "Fog persisting"},
    {"name": "Scattered Storms", "visibility": "FAIR", "sea_state_range": (2, 4), "forecast": "Isolated thunderstorms"},
    {"name": "Ideal", "visibility": "GOOD", "sea_state_range": (1, 2), "forecast": "Optimal conditions"},
]


def generate_scenario_backbone(designer: DataDesigner, model_config: ModelConfig,
                                params: ScenarioParams) -> ScenarioBackbone:
    """Generate consistent scenario facts using LLMStructuredColumnConfig."""
    
    builder = DataDesignerConfigBuilder(model_configs=[model_config])
    
    # Select adversary units based on nation
    if params.primary_adversary_nation == "China":
        naval_units = [u for u in ADVERSARY_NAVAL_UNITS if "PLA" in u or "China" in u]
        air_units = [u for u in ADVERSARY_AIR_UNITS if "PLA" in u]
    elif params.primary_adversary_nation == "DPRK":
        naval_units = [u for u in ADVERSARY_NAVAL_UNITS if "DPRK" in u]
        air_units = [u for u in ADVERSARY_AIR_UNITS if "DPRK" in u]
    else:
        naval_units = [u for u in ADVERSARY_NAVAL_UNITS if "Russian" in u]
        air_units = [u for u in ADVERSARY_AIR_UNITS if "Russian" in u]
    
    naval_units = naval_units or ADVERSARY_NAVAL_UNITS[:5]
    air_units = air_units or ADVERSARY_AIR_UNITS[:5]
    
    seed_df = pd.DataFrame([{
        "scenario_type": params.scenario_type,
        "scenario_description": SCENARIO_PROFILES[params.scenario_type]['description'],
        "theater_location": params.theater_location,
        "escalation_level": params.escalation_level,
        "reporting_unit": params.reporting_unit,
        "dtg": params.dtg,
        "adversary_nation": params.primary_adversary_nation,
        "suggested_naval_units": ", ".join(naval_units[:5]),
        "suggested_air_units": ", ".join(air_units[:5]),
        "weather_visibility": params.weather_pattern["visibility"],
        "sea_state_range": f"{params.weather_pattern['sea_state_range'][0]}-{params.weather_pattern['sea_state_range'][1]}",
        "weather_forecast": params.weather_pattern["forecast"],
    }])
    builder.with_seed_dataset(DataFrameSeedSource(df=seed_df))
    
    builder.add_column(LLMStructuredColumnConfig(
        name="backbone",
        model_alias=model_config.alias,
        output_format=ScenarioBackbone,
        prompt="""Generate consistent facts for a military intelligence scenario.

Scenario Context:
- Type: {{ scenario_type }} - {{ scenario_description }}
- Location: {{ theater_location }}
- Escalation Level: {{ escalation_level }}
- Adversary: {{ adversary_nation }}
- Suggested naval units: {{ suggested_naval_units }}
- Suggested air units: {{ suggested_air_units }}
- Weather: {{ weather_visibility }} visibility, sea state {{ sea_state_range }}

Generate realistic military values. Use DDHHMMz format for timestamps.""",
    ))
    
    try:
        result = designer.preview(builder, num_records=1)
        row = result.dataset.iloc[0]
        backbone_data = row['backbone']
        
        if isinstance(backbone_data, ScenarioBackbone):
            return backbone_data
        elif isinstance(backbone_data, dict):
            return ScenarioBackbone(**backbone_data)
        elif isinstance(backbone_data, str):
            return ScenarioBackbone(**json.loads(backbone_data))
        else:
            raise ValueError(f"Unexpected backbone type: {type(backbone_data)}")
            
    except Exception as e:
        print(f"    Warning: Structured generation failed ({e}), using defaults")
        return ScenarioBackbone(
            primary_event_dtg=params.dtg.split()[0],
            secondary_event_dtg=params.dtg.split()[0],
            isr_observation_dtg=params.dtg.split()[0],
            adversary_unit=naval_units[0] if naval_units else "PLA Navy East Sea Fleet",
            adversary_vessel_count=8,
            adversary_aircraft_count=12,
            adversary_formation_location=f"15nm east of {params.theater_location}",
            adversary_grid_ref="25.0°N 122.0°E",
            adversary_activity="conducting coordinated maritime operations",
            friendly_arg_name="ARG-71",
            friendly_air_wing="CVW-5",
            friendly_readiness_pct=95,
            friendly_munitions_status="98% stockpiles; 2% shortfall",
            friendly_personnel_count=15000,
            partner_request="ISR sharing and coordination",
            primary_assessment="Adversary activity elevated; monitoring recommended",
        )

OUTPUT_DIR = PROJECT_ROOT / "data" / "supplementary"

# =============================================================================
# INSTRUCTION PROMPTS FOR EACH DOCUMENT TYPE
# =============================================================================

INTSUM_INSTRUCTION = """You are a military intelligence analyst. Based on the following intelligence inputs, generate a consolidated Intelligence Summary (INTSUM).

FORMAT REQUIREMENTS:
- Use NUMBERED section headers exactly as shown
- Separate sections with "---" dividers
- Use bullet points with "-" for list items

REQUIRED SECTIONS:
1. EXECUTIVE SUMMARY - 2-3 sentences on key developments
2. CURRENT SITUATION - Detailed breakdown of current activity
3. ENEMY ORDER OF BATTLE - Known/estimated enemy forces
4. ENEMY INTENTIONS - Assessed adversary goals and likely actions
5. COLLECTION PRIORITIES - Intelligence gaps requiring collection
6. OUTLOOK - 24-72 hour forecast

Generate ONLY the INTSUM using the exact numbered format above.

=== INTELLIGENCE INPUTS ===
"""

SPOTREP_INSTRUCTION = """You are a tactical military observer. Based on the following observation, generate a SPOT Report using the SALUTE format.

FORMAT REQUIREMENTS:
- Use the exact SALUTE format headings
- Be concise - each field should be 1-2 lines maximum
- Include DTG and grid reference

SALUTE FORMAT:
S - SIZE: Number and type of personnel/equipment
A - ACTIVITY: What they are doing
L - LOCATION: Grid coordinates and description
U - UNIT: Unit identification (if known)
T - TIME: Date-time group of observation
E - EQUIPMENT: Weapons, vehicles, special equipment

Generate ONLY the SPOTREP using the exact SALUTE format above.

=== OBSERVATION DATA ===
"""

CONTACT_INSTRUCTION = """You are a tactical unit leader. Generate an immediate Contact Report based on the following enemy contact situation.

FORMAT REQUIREMENTS:
- Use standard Contact Report format
- Be urgent and concise
- Include grid reference and time

CONTACT REPORT FORMAT:
CONTACT REPORT
DTG: [Date-time group]
FROM: [Reporting unit]
CONTACT TYPE: [Visual/Acoustic/Electronic/Engagement]
ENEMY FORCE: [Size and type]
LOCATION: [Grid reference]
ACTIVITY: [What enemy is doing]
FRIENDLY STATUS: [Casualties/damage if any]
REQUEST: [Fire support/reinforcement/etc.]

Generate ONLY the Contact Report using the format above.

=== CONTACT SITUATION ===
"""

OPORD_INSTRUCTION = """You are a military operations officer. Generate an Operations Order (OPORD) based on the following mission requirements.

FORMAT REQUIREMENTS:
- Use the 5-paragraph OPORD format
- Include specific task organization
- Be detailed but actionable

5-PARAGRAPH OPORD FORMAT:
1. SITUATION
   a. Enemy Forces
   b. Friendly Forces
   c. Attachments/Detachments

2. MISSION
   [Who, What, When, Where, Why - single paragraph]

3. EXECUTION
   a. Commander's Intent
   b. Concept of Operations
   c. Tasks to Subordinate Units
   d. Coordinating Instructions

4. SUSTAINMENT
   a. Logistics
   b. Personnel Services

5. COMMAND AND SIGNAL
   a. Command
   b. Signal

Generate ONLY the OPORD using the exact 5-paragraph format above.

=== MISSION REQUIREMENTS ===
"""

FRAGO_INSTRUCTION = """You are a military operations officer. Generate a Fragmentary Order (FRAGO) to modify an existing operation.

FORMAT REQUIREMENTS:
- Reference parent OPORD
- Only include changed elements
- Be concise and actionable

FRAGO FORMAT:
FRAGMENTARY ORDER [Number]
REFERENCE: [Parent OPORD]
DTG: [Current date-time]
TIME ZONE: ZULU

1. SITUATION: [Changes only]
2. MISSION: [Changes only if modified]
3. EXECUTION: [Specific changes to tasks]
4. SUSTAINMENT: [Changes only]
5. COMMAND AND SIGNAL: [Changes only]

ACKNOWLEDGE.

Generate ONLY the FRAGO using the format above.

=== SITUATION CHANGE ===
"""

WATCHLOG_INSTRUCTION = """You are a watch officer maintaining the unit watch log. Generate watch log entries for the following events.

FORMAT REQUIREMENTS:
- Use standard military log format
- Each entry has DTG, event, action taken
- Chronological order

WATCH LOG FORMAT:
WATCH LOG - [Unit]
DATE: [Date]
WATCH: [Watch period, e.g., "0000-0800"]

[DTG] - [Event description] - [Action taken/reported to]
[DTG] - [Event description] - [Action taken/reported to]
...

WATCH OFFICER: [Name/Rank]
RELIEVED BY: [Name/Rank] at [DTG]

Generate ONLY the Watch Log using the format above.

=== WATCH EVENTS ===
"""

# Document type configurations
DOC_TYPES = {
    "intsum": {
        "name": "Intelligence Summary",
        "instruction": INTSUM_INSTRUCTION,
        "weight": 0.30,  # 30% of supplementary data
    },
    "spotrep": {
        "name": "Spot Report",
        "instruction": SPOTREP_INSTRUCTION,
        "weight": 0.20,
    },
    "contact": {
        "name": "Contact Report",
        "instruction": CONTACT_INSTRUCTION,
        "weight": 0.15,
    },
    "opord": {
        "name": "Operations Order",
        "instruction": OPORD_INSTRUCTION,
        "weight": 0.15,
    },
    "frago": {
        "name": "Fragmentary Order",
        "instruction": FRAGO_INSTRUCTION,
        "weight": 0.10,
    },
    "watchlog": {
        "name": "Watch Officer Log",
        "instruction": WATCHLOG_INSTRUCTION,
        "weight": 0.10,
    },
}


# =============================================================================
# DATA DESIGNER SETUP (reuses existing infrastructure)
# =============================================================================

def create_designer():
    """Create DataDesigner instance using local NIM."""
    if not check_nim_connection(NIM_BASE_URL):
        raise ConnectionError(f"Local NIM not accessible at {NIM_BASE_URL}")
    
    model_provider = ModelProvider(
        name="local-nim",
        endpoint=NIM_BASE_URL,
        provider_type="openai",
        api_key="not-needed",
    )
    
    model_config = ModelConfig(
        alias="intel-gen",
        model=NIM_MODEL,
        provider="local-nim",
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.8,
            max_tokens=4096,
        ),
    )
    
    designer = DataDesigner(
        model_providers=[model_provider],
        secret_resolver=PlaintextResolver(),
    )
    
    return designer, model_config


def llm_call(prompt: str, max_tokens: int = 4096, retries: int = 3) -> str:
    """Direct LLM call via OpenAI client."""
    client = OpenAI(base_url=NIM_BASE_URL, api_key="not-needed")
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=NIM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content
            raise ValueError("Empty response")
        except Exception as e:
            if attempt < retries - 1:
                import time
                time.sleep(1)
            else:
                return f"[Error: {e}]"
    return "[Error: No response]"


# =============================================================================
# SCENARIO PARAMETER GENERATION (simplified from main script)
# =============================================================================

def generate_simple_params(designer: DataDesigner, model_config: ModelConfig) -> ScenarioParams:
    """Generate scenario parameters using Data Designer samplers."""
    
    builder = DataDesignerConfigBuilder(model_configs=[model_config])
    
    # Sample scenario type
    builder.add_column(SamplerColumnConfig(
        name="scenario_type",
        sampler_type="category",
        params=CategorySamplerParams(values=list(SCENARIO_PROFILES.keys())),
    ))
    
    builder.add_column(SamplerColumnConfig(
        name="theater_location",
        sampler_type="category",
        params=CategorySamplerParams(values=THEATER_LOCATIONS),
    ))
    
    builder.add_column(SamplerColumnConfig(
        name="escalation_level",
        sampler_type="category",
        params=CategorySamplerParams(
            values=ESCALATION_LEVELS,
            weights=[0.15, 0.30, 0.25, 0.20, 0.10],
        ),
    ))
    
    builder.add_column(SamplerColumnConfig(
        name="reporting_unit",
        sampler_type="category",
        params=CategorySamplerParams(values=REPORTING_UNITS),
    ))
    
    builder.add_column(SamplerColumnConfig(
        name="adversary_nation",
        sampler_type="category",
        params=CategorySamplerParams(
            values=["China", "DPRK", "Russia"],
            weights=[0.60, 0.30, 0.10],
        ),
    ))
    
    result = designer.preview(builder, num_records=1)
    row = result.dataset.iloc[0]
    
    scenario_type = row['scenario_type']
    profile = SCENARIO_PROFILES[scenario_type]
    weather_pattern = random.choice(WEATHER_PATTERNS)
    
    base_time = datetime.now() - timedelta(days=random.randint(0, 90))
    dtg = generate_dtg(base_time)
    
    reporting_hours = random.choice([6, 8, 12, 24])
    end_time = base_time
    start_time = end_time - timedelta(hours=reporting_hours)
    
    return ScenarioParams(
        scenario_type=scenario_type,
        theater_location=row['theater_location'],
        escalation_level=row['escalation_level'],
        reporting_unit=row['reporting_unit'],
        dtg=dtg,
        reporting_period_start=start_time.strftime("%d%H%MZ").upper(),
        reporting_period_end=end_time.strftime("%d%H%MZ").upper(),
        reporting_period_hours=reporting_hours,
        classification="SECRET//NOFORN",
        num_flash_reports=2,
        num_isr_reports=2,
        num_sensor_reports=2,
        include_missile_warning=random.random() < profile["missile_prob"],
        include_cyber_report=random.random() < profile["cyber_prob"],
        include_partner_report=random.random() < profile["partner_prob"],
        include_narrative=False,
        include_humint=False,
        include_sigint=False,
        include_sub_contact=False,
        weather_pattern=weather_pattern,
        primary_adversary_nation=row['adversary_nation'],
    )


# =============================================================================
# INPUT GENERATORS (simplified versions for supplementary docs)
# =============================================================================

def generate_intel_inputs(params: ScenarioParams, backbone: ScenarioBackbone) -> str:
    """Generate simplified intelligence inputs for document generation."""
    
    inputs = f"""FLASH REPORT
DTG: {backbone.primary_event_dtg}
EVENT: {backbone.adversary_unit} observed {backbone.adversary_activity} at {backbone.adversary_formation_location}
VESSELS: {backbone.adversary_vessel_count}
AIRCRAFT: {backbone.adversary_aircraft_count}
THREAT LEVEL: {backbone.threat_level}
CONFIDENCE: {backbone.confidence_level}

ISR SUMMARY
DTG: {backbone.isr_observation_dtg}
Location: {backbone.adversary_formation_location} ({backbone.adversary_grid_ref})
Activity: {backbone.adversary_activity}
Weather: Visibility {backbone.weather_visibility}, Sea State {backbone.sea_state}

FRIENDLY STATUS
Unit: {backbone.friendly_arg_name}
Air Wing: {backbone.friendly_air_wing}
Readiness: {backbone.friendly_readiness_pct}%
Munitions: {backbone.friendly_munitions_status}
Personnel: {backbone.friendly_personnel_count} troops

ASSESSMENT
{backbone.primary_assessment}
Partner ({backbone.partner_nation}) requesting: {backbone.partner_request}
"""
    return inputs


# =============================================================================
# DOCUMENT GENERATORS
# =============================================================================

def generate_intsum(params: ScenarioParams, backbone: ScenarioBackbone) -> tuple[str, str]:
    """Generate Intelligence Summary (INTSUM)."""
    
    inputs = generate_intel_inputs(params, backbone)
    
    prompt = f"""You are a senior intelligence analyst. Generate an Intelligence Summary (INTSUM).

SCENARIO CONTEXT:
- Scenario: {SCENARIO_PROFILES[params.scenario_type]['description']}
- Theater: {params.theater_location}
- Escalation: {params.escalation_level}
- DTG: {params.dtg}
- Reporting Unit: {params.reporting_unit}

KEY FACTS:
- Adversary: {backbone.adversary_unit}
- Location: {backbone.adversary_formation_location}
- Activity: {backbone.adversary_activity}
- Vessels: {backbone.adversary_vessel_count}, Aircraft: {backbone.adversary_aircraft_count}
- Threat Level: {backbone.threat_level}
- Friendly Forces: {backbone.friendly_arg_name}, {backbone.friendly_air_wing} at {backbone.friendly_readiness_pct}% readiness
- Partner: {backbone.partner_nation}

INTELLIGENCE INPUTS:
{inputs}

Generate an INTSUM with these sections:
1. EXECUTIVE SUMMARY (2-3 sentences)
2. CURRENT SITUATION (detailed breakdown)
3. ENEMY ORDER OF BATTLE (known/estimated forces)
4. ENEMY INTENTIONS (assessed goals and likely actions)
5. COLLECTION PRIORITIES (intelligence gaps)
6. OUTLOOK (24-72 hour forecast)

Use "---" dividers between sections. Generate ONLY the INTSUM."""

    output = llm_call(prompt, max_tokens=4096)
    full_input = INTSUM_INSTRUCTION + inputs
    return full_input, output


def generate_spotrep(params: ScenarioParams, backbone: ScenarioBackbone) -> tuple[str, str]:
    """Generate Spot Report (SPOTREP) using SALUTE format."""
    
    observation = f"""OBSERVATION DATA:
- Observer: {params.reporting_unit}
- DTG: {backbone.isr_observation_dtg}
- Location: {backbone.adversary_formation_location}
- Grid: {backbone.adversary_grid_ref}
- Observed Unit: {backbone.adversary_unit}
- Activity: {backbone.adversary_activity}
- Vessel Count: {backbone.adversary_vessel_count}
- Aircraft Count: {backbone.adversary_aircraft_count}
- Weather: {backbone.weather_visibility} visibility
"""
    
    prompt = f"""You are a tactical observer. Generate a SPOT Report using SALUTE format.

{observation}

SALUTE FORMAT:
S - SIZE: Number and type of personnel/equipment observed
A - ACTIVITY: What they are doing
L - LOCATION: {backbone.adversary_grid_ref} ({backbone.adversary_formation_location})
U - UNIT: {backbone.adversary_unit} (if identified)
T - TIME: {backbone.isr_observation_dtg}
E - EQUIPMENT: Weapons, vehicles, special equipment observed

Generate ONLY the SPOTREP. Be concise - each field 1-2 lines maximum."""

    output = llm_call(prompt, max_tokens=1024)
    full_input = SPOTREP_INSTRUCTION + observation
    return full_input, output


def generate_contact_report(params: ScenarioParams, backbone: ScenarioBackbone) -> tuple[str, str]:
    """Generate Contact Report."""
    
    # Determine contact type based on scenario
    if backbone.adversary_aircraft_count > backbone.adversary_vessel_count:
        contact_type = "VISUAL - AIR"
    elif "submarine" in params.scenario_type.lower():
        contact_type = "ACOUSTIC - SUBSURFACE"
    else:
        contact_type = "VISUAL - SURFACE"
    
    situation = f"""CONTACT SITUATION:
- Reporting Unit: {params.reporting_unit}
- DTG: {backbone.primary_event_dtg}
- Location: {backbone.adversary_grid_ref}
- Contact Type: {contact_type}
- Enemy Force: {backbone.adversary_unit}
- Vessels: {backbone.adversary_vessel_count}
- Aircraft: {backbone.adversary_aircraft_count}
- Enemy Activity: {backbone.adversary_activity}
- Threat Level: {backbone.threat_level}
- Friendly Status: {backbone.friendly_arg_name} at {backbone.friendly_readiness_pct}% readiness
"""
    
    prompt = f"""You are a tactical unit leader. Generate an immediate Contact Report.

{situation}

Generate a Contact Report with:
- DTG
- FROM: {params.reporting_unit}
- CONTACT TYPE: {contact_type}
- ENEMY FORCE: Size and type
- LOCATION: Grid reference
- ACTIVITY: Current enemy actions
- FRIENDLY STATUS: Any casualties/damage (assume none unless engagement)
- REQUEST: Appropriate support request based on threat level {backbone.threat_level}

Generate ONLY the Contact Report. Be urgent and concise."""

    output = llm_call(prompt, max_tokens=1024)
    full_input = CONTACT_INSTRUCTION + situation
    return full_input, output


def generate_opord(params: ScenarioParams, backbone: ScenarioBackbone) -> tuple[str, str]:
    """Generate Operations Order (OPORD)."""
    
    # Determine mission type based on scenario
    mission_types = {
        "taiwan_invasion": "conduct defensive operations",
        "taiwan_blockade": "enforce freedom of navigation",
        "korean_peninsula_tension": "deter aggression and support ROK forces",
        "south_china_sea_standoff": "maintain presence and deter escalation",
        "submarine_tracking": "locate and track hostile submarine",
        "gray_zone_harassment": "protect partner nation vessels",
        "humanitarian_disaster": "conduct humanitarian assistance operations",
        "exercise_malabar": "conduct combined maritime exercise",
    }
    mission_verb = mission_types.get(params.scenario_type, "conduct operations")
    
    requirements = f"""MISSION REQUIREMENTS:
- Commander: {params.reporting_unit}
- AO: {params.theater_location}
- Timeframe: {params.dtg} to +72 hours
- Mission: {mission_verb}
- Escalation Level: {params.escalation_level}

ENEMY SITUATION:
- Force: {backbone.adversary_unit}
- Location: {backbone.adversary_formation_location}
- Strength: {backbone.adversary_vessel_count} vessels, {backbone.adversary_aircraft_count} aircraft
- Activity: {backbone.adversary_activity}
- Assessed Intent: {backbone.primary_assessment}

FRIENDLY SITUATION:
- Main Effort: {backbone.friendly_arg_name}
- Supporting: {backbone.friendly_air_wing}
- Readiness: {backbone.friendly_readiness_pct}%
- Munitions: {backbone.friendly_munitions_status}

PARTNER COORDINATION:
- Partner: {backbone.partner_nation}
- Request: {backbone.partner_request}

WEATHER:
- Visibility: {backbone.weather_visibility}
- Sea State: {backbone.sea_state}
- Forecast: {backbone.weather_forecast}
"""
    
    prompt = f"""You are a military operations officer. Generate an Operations Order (OPORD).

{requirements}

Generate a 5-paragraph OPORD:
1. SITUATION (Enemy, Friendly, Attachments)
2. MISSION (Who, What, When, Where, Why)
3. EXECUTION (Intent, Concept, Tasks, Coordinating Instructions)
4. SUSTAINMENT (Logistics, Personnel)
5. COMMAND AND SIGNAL

Use appropriate military formatting. Generate ONLY the OPORD."""

    output = llm_call(prompt, max_tokens=4096)
    full_input = OPORD_INSTRUCTION + requirements
    return full_input, output


def generate_frago(params: ScenarioParams, backbone: ScenarioBackbone) -> tuple[str, str]:
    """Generate Fragmentary Order (FRAGO)."""
    
    # Simulate a change to existing operations
    changes = [
        f"Adversary has repositioned to {backbone.adversary_grid_ref}",
        f"Partner {backbone.partner_nation} requests increased coordination",
        f"Weather degrading to {backbone.weather_visibility} visibility",
        f"Threat level elevated to {backbone.threat_level}",
        f"Additional {backbone.adversary_aircraft_count} aircraft detected",
    ]
    change_reason = random.choice(changes)
    
    situation = f"""SITUATION CHANGE:
- Parent OPORD: OPORD 01-26 (Operation PACIFIC SHIELD)
- Current DTG: {params.dtg}
- AO: {params.theater_location}

CHANGE DRIVER:
{change_reason}

CURRENT DISPOSITION:
- {backbone.friendly_arg_name}: Operating in AO
- {backbone.friendly_air_wing}: {backbone.friendly_readiness_pct}% readiness
- Adversary: {backbone.adversary_unit} conducting {backbone.adversary_activity}

REQUIRED MODIFICATIONS:
- Adjust scheme of maneuver
- Update coordination measures
- Revise timeline as needed
"""
    
    prompt = f"""You are a military operations officer. Generate a Fragmentary Order (FRAGO).

{situation}

Generate a FRAGO that:
- References parent OPORD
- Only includes CHANGED elements
- Is concise and actionable
- Ends with ACKNOWLEDGE

Use standard FRAGO format. Generate ONLY the FRAGO."""

    output = llm_call(prompt, max_tokens=2048)
    full_input = FRAGO_INSTRUCTION + situation
    return full_input, output


def generate_watchlog(params: ScenarioParams, backbone: ScenarioBackbone) -> tuple[str, str]:
    """Generate Watch Officer Log."""
    
    # Generate realistic watch events
    events = f"""WATCH EVENTS:
Unit: {params.reporting_unit}
Date: {params.dtg[:2]} {datetime.now().strftime('%b %Y').upper()}
Watch: 0000-0800 ZULU

Events to log:
1. {backbone.primary_event_dtg} - Flash traffic received regarding {backbone.adversary_unit}
2. {backbone.isr_observation_dtg} - ISR report confirms {backbone.adversary_vessel_count} vessels at {backbone.adversary_formation_location}
3. {backbone.secondary_event_dtg} - {backbone.friendly_arg_name} reports {backbone.friendly_readiness_pct}% readiness
4. [Time +30min] - METOC update: {backbone.weather_visibility} visibility, Sea State {backbone.sea_state}
5. [Time +45min] - {backbone.partner_nation} LNO requests coordination call
6. [Time +60min] - Threat level assessment updated to {backbone.threat_level}

Watch Officer: LCDR J. SMITH
"""
    
    prompt = f"""You are a watch officer. Generate watch log entries.

{events}

Generate a Watch Log with:
- Header (Unit, Date, Watch period)
- Chronological entries with DTG, event, action taken
- Each entry should be 1-2 lines
- End with Watch Officer info and relief

Generate ONLY the Watch Log."""

    output = llm_call(prompt, max_tokens=2048)
    full_input = WATCHLOG_INSTRUCTION + events
    return full_input, output


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_document(designer: DataDesigner, model_config: ModelConfig,
                     doc_type: str, doc_num: int) -> dict:
    """Generate a single supplementary document."""
    
    print(f"\n  [{doc_num}] Generating {DOC_TYPES[doc_type]['name']}...")
    
    # Generate scenario context
    params = generate_simple_params(designer, model_config)
    print(f"      Scenario: {params.scenario_type} @ {params.theater_location}")
    
    # Generate backbone for consistent facts
    backbone = generate_scenario_backbone(designer, model_config, params)
    print(f"      Adversary: {backbone.adversary_unit}")
    
    # Generate document based on type
    generators = {
        "intsum": generate_intsum,
        "spotrep": generate_spotrep,
        "contact": generate_contact_report,
        "opord": generate_opord,
        "frago": generate_frago,
        "watchlog": generate_watchlog,
    }
    
    full_input, output = generators[doc_type](params, backbone)
    
    return {
        "input": full_input,
        "output": output,
        "doc_type": doc_type,
        "scenario_type": params.scenario_type,
        "theater_location": params.theater_location,
        "escalation_level": params.escalation_level,
    }


def select_doc_type_weighted() -> str:
    """Select document type based on configured weights."""
    types = list(DOC_TYPES.keys())
    weights = [DOC_TYPES[t]["weight"] for t in types]
    return random.choices(types, weights=weights, k=1)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate supplementary military document training data"
    )
    parser.add_argument(
        "--count", type=int, default=10,
        help="Number of documents to generate (default: 10)"
    )
    parser.add_argument(
        "--type", type=str, default="all",
        choices=["all", "intsum", "spotrep", "contact", "opord", "frago", "watchlog"],
        help="Document type to generate (default: all - weighted mix)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL file path (default: data/supplementary/[type]_training.jsonl)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Supplementary Military Document Generator")
    print("="*60)
    print(f"  Documents to generate: {args.count}")
    print(f"  Document type: {args.type}")
    print(f"  Using NIM: {NIM_BASE_URL}")
    print(f"  Model: {NIM_MODEL}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine output file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"{args.type}_training.jsonl"
    
    print(f"  Output: {output_path}")
    
    # Create designer
    designer, model_config = create_designer()
    
    # Generate documents
    all_records = []
    type_counts = {t: 0 for t in DOC_TYPES.keys()}
    
    for i in range(1, args.count + 1):
        try:
            # Select document type
            if args.type == "all":
                doc_type = select_doc_type_weighted()
            else:
                doc_type = args.type
            
            record = generate_document(designer, model_config, doc_type, i)
            all_records.append(record)
            type_counts[doc_type] += 1
            
            print(f"      Done ({len(record['output'])} chars)")
            
        except Exception as e:
            print(f"      ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Save to JSONL
    if all_records:
        with open(output_path, 'w') as f:
            for record in all_records:
                # Training format: input/output only
                training_record = {
                    "input": record["input"],
                    "output": record["output"],
                }
                f.write(json.dumps(training_record) + '\n')
        
        print(f"\n  Saved {len(all_records)} documents to {output_path}")
        
        # Print type distribution
        print("\n  Document type distribution:")
        for doc_type, count in type_counts.items():
            if count > 0:
                print(f"    - {DOC_TYPES[doc_type]['name']}: {count}")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
