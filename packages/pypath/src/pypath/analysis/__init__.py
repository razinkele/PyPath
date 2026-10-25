"""Analysis utilities for PyPath.

This module provides diagnostic and analysis tools for Ecopath models.
"""

from .prebalance import (
    calculate_biomass_range,
    calculate_biomass_slope,
    calculate_predator_prey_ratios,
    calculate_vital_rate_ratios,
    generate_prebalance_report,
    plot_biomass_vs_trophic_level,
    plot_vital_rate_vs_trophic_level,
    print_prebalance_summary,
)
from .scenario_optimizer import (
    ObjectiveSpec,
    ScenarioCandidate,
    ScenarioOptimizer,
    ScenarioScore,
    make_spatial_management_evaluator,
)

__all__ = [
    "calculate_biomass_slope",
    "calculate_biomass_range",
    "calculate_predator_prey_ratios",
    "calculate_vital_rate_ratios",
    "plot_biomass_vs_trophic_level",
    "plot_vital_rate_vs_trophic_level",
    "generate_prebalance_report",
    "print_prebalance_summary",
    "ObjectiveSpec",
    "ScenarioCandidate",
    "ScenarioOptimizer",
    "ScenarioScore",
    "make_spatial_management_evaluator",
]
