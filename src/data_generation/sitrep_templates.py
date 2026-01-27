"""
Sit rep templates and generation prompts for synthetic data.
Uses Jinja2 templates to create structured prompts for LLM generation.
"""

from jinja2 import Template
from typing import List, Dict, Any


# Base prompt template for sit rep generation
SITREP_GENERATION_PROMPT = Template("""
You are a logistics analyst generating a comprehensive situation report (sit rep) for deployment decision-making.

## Context
- **Scenario**: {{ scenario_name }}
- **Day**: {{ day_number }} of {{ total_days }}
- **Season**: {{ season }}
- **Mission**: Analyze {{ num_locations }} potential deployment sites and recommend the optimal location

## Logistics Data

{% for location in locations %}
### {{ location.location_name }} ({{ location.location_id }})

**Geographic Profile:**
- Coordinates: {{ location.latitude }}°N, {{ location.longitude }}°W
- Terrain: {{ location.terrain_type }} at {{ location.elevation_m }}m elevation
- Distance from Distribution Center: {{ location.distance_from_dc_km }} km

**Current Conditions:**
- Weather: {{ location.weather_condition }}, {{ location.temperature_c }}°C
- Wind Speed: {{ location.wind_speed_kmh }} km/h
- Precipitation: {{ location.precipitation_mm }} mm
- Road Quality: {{ location.road_quality }}

**Infrastructure:**
- Airport Access: {{ "Yes" if location.has_airport else "No" }}
- Seaport Access: {{ "Yes" if location.has_seaport else "No" }}
- Rail Access: {{ "Yes" if location.has_rail_access else "No" }}

**Market Analysis:**
- Population Served: {{ "{:,}".format(location.population_served) }}
- Current Demand: {{ "{:,}".format(location.current_demand_units) }} units/day
- Projected Growth: {{ location.demand_growth_projection_pct }}%
- Market Saturation: {{ location.market_saturation_pct }}%
- Competitors: {{ location.num_competitors }}

**Financial Metrics:**
- Base Deployment Cost: ${{ "{:,.0f}".format(location.base_deployment_cost_usd) }}
- Transport Cost per Unit: ${{ "{:.2f}".format(location.transportation_cost_per_unit_usd) }}
- Labor Cost: ${{ "{:.2f}".format(location.labor_cost_per_hour_usd) }}/hour
- Facility Rental: ${{ "{:,.0f}".format(location.facility_rental_monthly_usd) }}/month

**Risk Assessment:**
- Political Stability: {{ location.political_stability_score }}/10
- Natural Disaster Risk: {{ location.natural_disaster_risk_score }}/10
- Supply Chain Risk: {{ location.supply_chain_risk_score }}/10

{% endfor %}

{% if previous_day_summary %}
## Previous Day Context (Day {{ day_number - 1 }})
{{ previous_day_summary }}
{% endif %}

## Task

Generate a **comprehensive, well-formatted markdown sit rep** with the following structure:

# {{ scenario_name }} - Day {{ day_number }} Logistics Deployment Analysis

## TLDR
**Recommended Deployment Site:** [Location Name]
**Key Insight:** [One sentence critical finding]
**Rationale:**
- [Primary reason with data citation] (See: Section Name)
- [Secondary reason with data citation] (See: Section Name)
- [Risk consideration with data citation] (See: Section Name)

## Executive Summary
[2-3 paragraphs providing overview of analysis, key findings, and urgent considerations]

## Location Analysis
[Detailed analysis of each location with subsections]

### [Location 1 Name]
[Comprehensive analysis with all relevant metrics, strengths, weaknesses]

### [Location 2 Name]
[Comprehensive analysis with all relevant metrics, strengths, weaknesses]

[Continue for all locations...]

## Comparative Analysis
[Tables comparing key metrics across locations, identify leaders and laggards in each category]

| Metric | Location 1 | Location 2 | Location 3 | Best Choice |
|--------|-----------|-----------|-----------|-------------|
| Cost | $XXX | $XXX | $XXX | Location X |
| Demand | XXX units | XXX units | XXX units | Location X |
| Risk | X/10 | X/10 | X/10 | Location X |

## Temporal Trends
[Analysis of how conditions have changed since previous days, emerging patterns]

## Recommendations

### Primary Recommendation: [Location Name]
[Detailed justification with specific data citations, expected outcomes, implementation timeline]

**Strengths:**
- [Strength 1 with data]
- [Strength 2 with data]

**Risks and Mitigations:**
- **Risk:** [Description]
  - **Mitigation:** [Strategy]

### Alternative Options
[Brief discussion of backup options and conditions under which they might be preferred]

## Action Items
- [ ] Immediate: [Specific task with timeline]
- [ ] Short-term (24-48h): [Specific task]
- [ ] Planning: [Strategic task]

---
*Analysis generated for Day {{ day_number }} of {{ scenario_name }} deployment scenario*

## Formatting Requirements

1. Use proper markdown headers (# for title, ## for main sections, ### for subsections)
2. Bold important terms and location names
3. Use tables for comparative data
4. Use checkboxes for action items
5. Include data citations in parentheses: (Location Analysis: Alpha Base)
6. Keep TLDR concise and actionable
7. Include specific numbers and percentages in recommendations
8. Use bullet points and numbered lists appropriately
""")


# Simplified prompt for few-shot learning examples
SITREP_EXAMPLE_PROMPT = Template("""
Generate a logistics deployment sit rep for Day {{ day_number }}.

Available locations: {{ location_names | join(', ') }}

Focus on:
1. Clear recommendation with rationale
2. Proper markdown formatting with all required sections
3. Data-driven comparisons
4. Actionable next steps

Use the standard sit rep structure with TLDR, Executive Summary, Location Analysis, Comparative Analysis, Recommendations, and Action Items.
""")


# Template for temporal variation prompts
TEMPORAL_VARIATION_PROMPT = Template("""
Day {{ day_number }} Update:

Key changes from Day {{ previous_day }}:
{% for change in changes %}
- {{ change }}
{% endfor %}

Analyze how these changes affect the deployment decision and update your recommendation accordingly.
""")


# Template for formatting locations data for input
def format_locations_for_prompt(
    locations: List[Dict[str, Any]],
    scenario_name: str,
    day_number: int,
    total_days: int = 7,
    previous_day_summary: str = None
) -> str:
    """
    Format location data into a prompt for sit rep generation.

    Args:
        locations: List of location dictionaries
        scenario_name: Name of the deployment scenario
        day_number: Current day number
        total_days: Total days in scenario
        previous_day_summary: Optional summary from previous day

    Returns:
        Formatted prompt string
    """
    # Determine season based on scenario or day
    season = "Summer"  # Default, can be made more sophisticated

    return SITREP_GENERATION_PROMPT.render(
        scenario_name=scenario_name,
        day_number=day_number,
        total_days=total_days,
        season=season,
        num_locations=len(locations),
        locations=locations,
        previous_day_summary=previous_day_summary
    )


# Template for evaluation prompt (testing model output)
EVALUATION_PROMPT = Template("""
Generate a logistics deployment situation report for Day {{ day_number }}.

## Logistics Data

{% for location in locations %}
### Location: {{ location.location_name }} (ID: {{ location.location_id }})
- Coordinates: {{ location.latitude }}, {{ location.longitude }}
- Weather: {{ location.weather_condition }}, {{ location.temperature_c }}°C
- Distance from DC: {{ location.distance_from_dc_km }} km
- Demand: {{ location.current_demand_units }} units/day
- Competitors: {{ location.num_competitors }}
- Deployment Cost: ${{ "{:,.0f}".format(location.base_deployment_cost_usd) }}
- Political Stability: {{ location.political_stability_score }}/10
- Infrastructure: Airport={{ location.has_airport }}, Port={{ location.has_seaport }}, Rail={{ location.has_rail_access }}

{% endfor %}

Generate a comprehensive sit rep with all required sections: TLDR, Executive Summary, Location Analysis, Comparative Analysis, Temporal Trends, Recommendations, and Action Items.
""")


def format_evaluation_prompt(locations: List[Dict[str, Any]], day_number: int) -> str:
    """
    Format a simplified prompt for model evaluation.

    Args:
        locations: List of location dictionaries
        day_number: Current day number

    Returns:
        Formatted evaluation prompt
    """
    return EVALUATION_PROMPT.render(
        day_number=day_number,
        locations=locations
    )


# Example output structure for reference
EXAMPLE_SITREP = """
# Coastal Expansion - Day 3 Logistics Deployment Analysis

## TLDR
**Recommended Deployment Site:** Alpha Base
**Key Insight:** Alpha Base offers optimal cost-to-demand ratio despite moderate competition.
**Rationale:**
- Lowest total deployment cost at $87,500 with high demand of 6,200 units/day (See: Location Analysis)
- Superior infrastructure with airport, seaport, and rail access (See: Comparative Analysis)
- Weather improving from Day 2 conditions, reducing operational risk (See: Temporal Trends)

## Executive Summary

Analysis of five potential deployment sites for Day 3 reveals Alpha Base as the optimal choice for immediate deployment. Despite facing 4 competitors in the market, Alpha Base demonstrates the strongest overall value proposition with its combination of low deployment costs, high demand, and superior infrastructure.

Key findings indicate that while Bravo Station offers lower competition (2 competitors), its significantly higher transportation costs ($7.80/unit vs $5.45/unit) and poor road quality create operational challenges that outweigh the competitive advantage. Charlie Point emerges as a strong secondary option with excellent political stability (8.5/10) but lacks the critical infrastructure access that Alpha Base provides.

Weather conditions across all sites have stabilized since Day 2, with clearing patterns observed in coastal regions. This improvement in operational conditions supports accelerated deployment timelines.

## Location Analysis

### Alpha Base (LOC-001)
Alpha Base represents the most balanced deployment opportunity with strong fundamentals across all evaluation criteria. Located at 33.2°N, 97.4°W at 145m elevation, the site offers coastal plain terrain with excellent accessibility.

**Strengths:**
- Comprehensive infrastructure: Airport, seaport, and rail access all available
- High demand volume: 6,200 units/day with 12% projected growth
- Competitive deployment cost: $87,500 base cost
- Good weather conditions: Clear, 24°C, minimal precipitation (2mm)

**Concerns:**
- Moderate competition: 4 competitors present
- Market saturation at 67%, limiting expansion potential
- Supply chain risk: 6.2/10 requires contingency planning

### Bravo Station (LOC-002)
Bravo Station offers advantages in market position but faces infrastructure challenges that increase operational complexity and costs.

**Strengths:**
- Low competition: Only 2 competitors
- Strong political stability: 8.1/10
- Large population served: 385,000
- Rail access available

**Concerns:**
- Poor road quality significantly impacts logistics
- High transportation costs: $7.80/unit
- No airport or seaport access
- Higher natural disaster risk: 7.3/10

[Continue for remaining locations...]

## Comparative Analysis

| Metric | Alpha Base | Bravo Station | Charlie Point | Delta Hub | Echo Terminal | Winner |
|--------|-----------|--------------|---------------|-----------|---------------|--------|
| Base Cost | $87,500 | $142,000 | $95,300 | $118,500 | $103,200 | Alpha Base |
| Transport Cost | $5.45 | $7.80 | $6.10 | $6.85 | $5.90 | Alpha Base |
| Demand (units/day) | 6,200 | 4,800 | 5,500 | 3,900 | 5,100 | Alpha Base |
| Competitors | 4 | 2 | 3 | 5 | 3 | Bravo Station |
| Political Stability | 7.8/10 | 8.1/10 | 8.5/10 | 6.9/10 | 7.4/10 | Charlie Point |
| Infrastructure Score | 3/3 | 1/3 | 2/3 | 2/3 | 3/3 | Alpha/Echo |

## Temporal Trends

Comparing Day 3 to previous days reveals positive developments in coastal regions:

**Weather Improvements:**
- Alpha Base: Clearing from Day 2 rain (15mm → 2mm precipitation)
- Echo Terminal: Temperature normalization (18°C → 23°C)

**Market Dynamics:**
- Demand growth accelerating: Average +2.1% day-over-day
- Competition remains stable at previously observed sites

**Risk Factors:**
- Supply chain stability improving in all coastal locations
- Political conditions remain favorable across evaluation zone

## Recommendations

### Primary Recommendation: Alpha Base

Deploy to Alpha Base immediately based on superior overall value proposition. The site offers the optimal combination of low costs, high demand, and comprehensive infrastructure that will support both immediate operations and future scaling.

**Deployment Plan:**
1. Initiate site preparation within 24 hours
2. Leverage all three infrastructure modes (air, sea, rail) for supply redundancy
3. Implement competitive positioning strategy to address 4-competitor market
4. Establish supply chain monitoring given 6.2/10 risk score

**Expected Outcomes:**
- 90-day breakeven projection based on 6,200 units/day demand
- Market share target: 25-30% (1,550-1,860 units/day)
- Revenue projection: $465K-$558K monthly at standard pricing

**Risks and Mitigations:**
- **Risk:** Market saturation at 67% limits expansion
  - **Mitigation:** Focus on service differentiation and quality; prepare Echo Terminal as expansion site
- **Risk:** Supply chain disruption (6.2/10 risk score)
  - **Mitigation:** Multi-modal supply strategy using air/sea/rail; maintain 15-day safety stock

### Alternative Options

**Charlie Point** serves as primary backup if Alpha Base deployment encounters delays. While lacking seaport access, its superior political stability (8.5/10) and good demand (5,500 units/day) provide solid fundamentals.

**Echo Terminal** should be considered for Phase 2 expansion given matching infrastructure capabilities and complementary geographic positioning.

## Action Items

- [ ] Immediate: Contact Alpha Base facility management for site survey and rental agreement
- [ ] Immediate: Initiate supply chain planning with all three infrastructure modes
- [ ] Short-term (24-48h): Deploy advance team for site preparation and competitive analysis
- [ ] Short-term (48-72h): Establish local supplier relationships for contingency supply chain
- [ ] Planning: Develop Charlie Point contingency deployment plan
- [ ] Planning: Model Phase 2 expansion scenarios for Echo Terminal integration

---
*Analysis generated for Day 3 of Coastal Expansion deployment scenario*
"""
