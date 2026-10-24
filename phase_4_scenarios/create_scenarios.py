#!/usr/bin/env python3
"""
Create and validate Curonian Lagoon management scenarios for Ecosim modeling.

This module defines 5 policy-relevant scenarios and provides validation functions
to ensure parameters are realistic and consistent.

Usage:
    from create_scenarios import load_scenarios, validate_all_scenarios
    scenarios = load_scenarios()
    validate_all_scenarios(scenarios)
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import sys


@dataclass
class ScenarioFishingParams:
    """Fishing effort multipliers for a scenario."""
    Bream: float
    Perch: float
    Roach: float
    Vimba: float
    Zander: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def check_valid_range(self) -> bool:
        """Verify all multipliers in [0.0, 2.0]."""
        for species, mult in asdict(self).items():
            if not (0.0 <= mult <= 2.0):
                return False
        return True

    def mean_multiplier(self) -> float:
        """Calculate mean effort across all species."""
        values = asdict(self).values()
        return sum(values) / len(values)


@dataclass
class ScenarioDefinition:
    """Complete scenario definition."""
    id: int
    code: str
    name: str
    description: str
    policy_focus: str
    period: str
    years: int
    fishing_effort: ScenarioFishingParams
    spatial_closure: Optional[Dict] = None
    seasonal_restrictions: Optional[Dict] = None
    environmental_forcing: str = "historical_mean"
    expected_annual_catch_tonnes: str = ""
    expected_mean_biomass_tonnes: str = ""
    expected_trophic_level: str = ""
    expected_economic_value_eur: str = ""
    output_file: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['fishing_effort'] = self.fishing_effort.to_dict()
        return data

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate scenario parameters. Returns (valid, errors)."""
        errors = []

        # Check fishing parameters
        if not self.fishing_effort.check_valid_range():
            errors.append(f"Scenario {self.id}: Fishing multipliers outside [0.0, 2.0]")

        # Check period
        if self.years != 11:
            errors.append(f"Scenario {self.id}: Period should be 11 years, got {self.years}")

        # Check that scenarios span the policy space
        # (uniform reduction/increase is fine; only BAU should be all 1.0)

        return len(errors) == 0, errors


def create_baseline_scenario() -> ScenarioDefinition:
    """Scenario 1: Business-as-usual (1.0× effort)."""
    return ScenarioDefinition(
        id=1,
        code="BAU",
        name="Baseline_BAU_2025_2035",
        description="Business-as-usual: maintain current fishing effort",
        policy_focus="Status quo",
        period="2025-2035",
        years=11,
        fishing_effort=ScenarioFishingParams(
            Bream=1.0,
            Perch=1.0,
            Roach=1.0,
            Vimba=1.0,
            Zander=1.0
        ),
        environmental_forcing="historical_mean",
        expected_annual_catch_tonnes="12-15",
        expected_mean_biomass_tonnes="2100-2400",
        expected_trophic_level="2.6-2.8",
        expected_economic_value_eur="65000-95000",
        output_file="results_BAU_2025_2035.csv",
        rationale="Reference scenario: maintain 2015-2018 average fishing effort"
    )


def create_reduction_50_scenario() -> ScenarioDefinition:
    """Scenario 2: Fishing reduction 50% (0.5× effort)."""
    return ScenarioDefinition(
        id=2,
        code="RED50",
        name="Fishing_Reduction_50pct",
        description="Reduce fishing pressure by 50% to rebuild stocks",
        policy_focus="Conservation",
        period="2025-2035",
        years=11,
        fishing_effort=ScenarioFishingParams(
            Bream=0.50,
            Perch=0.50,
            Roach=0.50,
            Vimba=0.50,
            Zander=0.50
        ),
        environmental_forcing="historical_mean",
        expected_annual_catch_tonnes="6-8",
        expected_mean_biomass_tonnes="2900-3200",
        expected_trophic_level="2.9-3.1",
        expected_economic_value_eur="35000-55000",
        output_file="results_RED50_2025_2035.csv",
        rationale="Ecosystem-based management: allow stock recovery and improve stability"
    )


def create_increase_30_scenario() -> ScenarioDefinition:
    """Scenario 3: Fishing increase 30% (1.3× effort)."""
    return ScenarioDefinition(
        id=3,
        code="INC30",
        name="Fishing_Increase_30pct",
        description="Increase fishing pressure by 30% to maximize yields",
        policy_focus="Productivity",
        period="2025-2035",
        years=11,
        fishing_effort=ScenarioFishingParams(
            Bream=1.30,
            Perch=1.30,
            Roach=1.30,
            Vimba=1.30,
            Zander=1.30
        ),
        environmental_forcing="historical_mean",
        expected_annual_catch_tonnes="16-20",
        expected_mean_biomass_tonnes="1600-1900",
        expected_trophic_level="2.4-2.6",
        expected_economic_value_eur="95000-140000",
        output_file="results_INC30_2025_2035.csv",
        rationale="Productivity focus: maximize annual catch revenue and economic return"
    )


def create_mpa_25_scenario() -> ScenarioDefinition:
    """Scenario 4: Spatial closure 25% (0.75× effort in fished area)."""
    return ScenarioDefinition(
        id=4,
        code="MPA25",
        name="Spatial_Closure_25pct",
        description="25% of lagoon closed to fishing (marine protected area)",
        policy_focus="Protection",
        period="2025-2035",
        years=11,
        fishing_effort=ScenarioFishingParams(
            Bream=0.75,
            Perch=0.75,
            Roach=0.75,
            Vimba=0.75,
            Zander=0.75
        ),
        spatial_closure={
            "area_percent": 0.25,
            "location": "north_lagoon_sanctuary",
            "area_km2": 212,
            "fishing_area_km2": 637
        },
        environmental_forcing="historical_mean",
        expected_annual_catch_tonnes="10-13",
        expected_mean_biomass_tonnes="2300-2600",
        expected_trophic_level="2.7-2.9",
        expected_economic_value_eur="60000-85000",
        output_file="results_MPA25_2025_2035.csv",
        rationale="Marine spatial planning: ecosystem protection plus spillover effects"
    )


def create_selective_harvest_scenario() -> ScenarioDefinition:
    """Scenario 5: Selective species fishing (varied multipliers)."""
    return ScenarioDefinition(
        id=5,
        code="SEL",
        name="Selective_Species_Fishing",
        description="Target high-value species with restrictions on low-value forage",
        policy_focus="Economics",
        period="2025-2035",
        years=11,
        fishing_effort=ScenarioFishingParams(
            Bream=1.20,        # Increase: high-value
            Perch=0.70,        # Reduce: low-value, small
            Roach=0.60,        # Reduce: forage species
            Vimba=0.80,        # Reduce: medium value
            Zander=1.30        # Increase: highest value, apex predator
        ),
        seasonal_restrictions={
            "Zander": {
                "months": "March-April",
                "multiplier": 1.0,
                "reason": "Spawning protection"
            }
        },
        environmental_forcing="historical_mean",
        expected_annual_catch_tonnes="10-13",
        expected_mean_biomass_tonnes="2200-2500",
        expected_trophic_level="2.8-3.0",
        expected_economic_value_eur="90000-140000",
        output_file="results_SEL_2025_2035.csv",
        rationale="Economic optimization: maximize value per tonne through selective harvesting"
    )


def create_all_scenarios() -> List[ScenarioDefinition]:
    """Create all 5 scenarios."""
    return [
        create_baseline_scenario(),
        create_reduction_50_scenario(),
        create_increase_30_scenario(),
        create_mpa_25_scenario(),
        create_selective_harvest_scenario()
    ]


def validate_all_scenarios(scenarios: List[ScenarioDefinition]) -> Tuple[bool, Dict]:
    """Validate all scenarios. Returns (all_valid, report_dict)."""
    report = {
        'total_scenarios': len(scenarios),
        'individual_validation': [],
        'cross_scenario_checks': [],
        'overall_valid': True
    }

    # Validate individual scenarios
    for scenario in scenarios:
        valid, errors = scenario.validate()
        report['individual_validation'].append({
            'scenario_id': scenario.id,
            'scenario_code': scenario.code,
            'valid': valid,
            'errors': errors
        })
        if not valid:
            report['overall_valid'] = False

    # Cross-scenario checks
    multipliers_by_scenario = {
        s.code: s.fishing_effort.mean_multiplier() for s in scenarios
    }

    # Check 1: Baseline must be 1.0×
    if multipliers_by_scenario.get('BAU', 1.0) != 1.0:
        report['cross_scenario_checks'].append({
            'check': 'Baseline multiplier',
            'passed': False,
            'message': 'Baseline scenario must have 1.0× mean multiplier'
        })
        report['overall_valid'] = False
    else:
        report['cross_scenario_checks'].append({
            'check': 'Baseline multiplier',
            'passed': True,
            'message': 'Baseline scenario correctly set to 1.0×'
        })

    # Check 2: Must have conservation and productivity scenarios
    mult_min = min(multipliers_by_scenario.values())
    mult_max = max(multipliers_by_scenario.values())
    if mult_min < 1.0 and mult_max > 1.0:
        report['cross_scenario_checks'].append({
            'check': 'Policy space coverage',
            'passed': True,
            'message': f'Scenarios span conservation ({mult_min:.2f}×) to productivity ({mult_max:.2f}×)'
        })
    else:
        report['cross_scenario_checks'].append({
            'check': 'Policy space coverage',
            'passed': False,
            'message': f'Scenarios must include both <1.0× and >1.0× multipliers'
        })
        report['overall_valid'] = False

    # Check 3: Minimum 4 scenarios
    if len(scenarios) >= 4:
        report['cross_scenario_checks'].append({
            'check': 'Number of scenarios',
            'passed': True,
            'message': f'Defined {len(scenarios)} scenarios (minimum 4)'
        })
    else:
        report['cross_scenario_checks'].append({
            'check': 'Number of scenarios',
            'passed': False,
            'message': f'Need at least 4 scenarios, have {len(scenarios)}'
        })
        report['overall_valid'] = False

    return report['overall_valid'], report


def print_validation_report(valid: bool, report: Dict) -> None:
    """Pretty-print validation report."""
    print("\n" + "=" * 70)
    print("SCENARIO VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTotal scenarios: {report['total_scenarios']}")

    print("\n--- Individual Scenario Validation ---")
    for item in report['individual_validation']:
        status = "[PASS] PASS" if item['valid'] else "[FAIL] FAIL"
        print(f"{status}: Scenario {item['scenario_id']} ({item['scenario_code']})")
        if item['errors']:
            for error in item['errors']:
                print(f"      - {error}")

    print("\n--- Cross-Scenario Checks ---")
    for check in report['cross_scenario_checks']:
        status = "[PASS] PASS" if check['passed'] else "[FAIL] FAIL"
        print(f"{status}: {check['check']}")
        print(f"      - {check['message']}")

    status = "[PASS] ALL SCENARIOS VALID" if report['overall_valid'] else "[FAIL] VALIDATION FAILED"
    print(f"\n{status}")
    print("=" * 70 + "\n")


def save_scenarios_json(scenarios: List[ScenarioDefinition], output_path: Path) -> None:
    """Save scenarios to JSON file."""
    data = {
        'metadata': {
            'study': 'Ecosystem-Based Fisheries Management for Curonian Lagoon',
            'modeling_period': '2025-2035',
            'years': 11,
            'time_steps_per_year': 12,
            'total_time_steps': 132
        },
        'scenarios': [s.to_dict() for s in scenarios]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[PASS] Saved scenarios to {output_path}")


def load_scenarios(json_path: Optional[Path] = None) -> List[ScenarioDefinition]:
    """Load scenarios from JSON file or create defaults."""
    if json_path and json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        scenarios = []
        for sc_data in data.get('scenarios', []):
            fishing_params = ScenarioFishingParams(**sc_data['fishing_effort'])
            sc_data['fishing_effort'] = fishing_params
            scenarios.append(ScenarioDefinition(**sc_data))
        return scenarios
    else:
        return create_all_scenarios()


if __name__ == '__main__':
    # Create scenarios
    scenarios = create_all_scenarios()

    # Validate
    valid, report = validate_all_scenarios(scenarios)
    print_validation_report(valid, report)

    # Save to JSON
    output_path = Path(__file__).parent / 'curonian_scenarios_2025_2035.json'
    save_scenarios_json(scenarios, output_path)

    # Print summary
    print("\nSCENARIO SUMMARY:")
    print("-" * 70)
    for scenario in scenarios:
        mean_mult = scenario.fishing_effort.mean_multiplier()
        print(f"\n{scenario.id}. {scenario.code}: {scenario.name}")
        print(f"   Mean multiplier: {mean_mult:.2f}×")
        print(f"   Expected catch: {scenario.expected_annual_catch_tonnes} tonnes/year")
        print(f"   Expected biomass: {scenario.expected_mean_biomass_tonnes} tonnes")
        print(f"   Policy focus: {scenario.policy_focus}")

    # Exit with appropriate code
    sys.exit(0 if valid else 1)



