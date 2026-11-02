# Analysis API Reference

Tools for model diagnostics, pre-balance checks, and post-simulation analysis.

## Functional Development Workflows

| Workflow | Primary API Entry Points | Typical Outcome |
|----------|--------------------------|-----------------|
| **Pre-run plausibility gate** | `generate_prebalance_report()`, `print_prebalance_summary()` | Early detection of trophic/parameter inconsistencies before Ecopath balancing |
| **Management scenario ranking** | `ObjectiveSpec`, `ScenarioCandidate`, `ScenarioOptimizer` | Weighted multi-objective ranking with feasibility penalties and risk tie-breaking |
| **Coupled scenario evaluator wiring** | `make_spatial_management_evaluator()` | Reusable evaluator that runs coupled simulations and returns objective metrics |

### Scenario optimization pattern

```python
from pypath.analysis import (
    ObjectiveSpec,
    ScenarioOptimizer,
    make_spatial_management_evaluator,
)

evaluator = make_spatial_management_evaluator(base_scenario, ecospace, years=range(1, 31))
optimizer = ScenarioOptimizer(
    objectives=[
        ObjectiveSpec("catch_total", weight=0.5, goal="maximize"),
        ObjectiveSpec("mean_biomass", weight=0.3, goal="maximize"),
        ObjectiveSpec("collapse_risk", weight=0.2, goal="minimize", max_value=0.25),
    ],
    evaluator=evaluator,
)

ranked = optimizer.optimize_grid(
    {
        "effort_scale": [0.8, 1.0, 1.2],
        "vulnerability_scale": [0.9, 1.0, 1.1],
    }
)
```

## Pre-Balance Diagnostics

Pre-balance diagnostics help identify parameter issues before running
the Ecopath mass-balance solver. Checks include biomass ratios,
ecotrophic efficiency bounds, and diet composition consistency.

::: pypath.analysis.prebalance
    options:
      show_root_heading: true
      members_order: source

## Network Indices

Network analysis functions for food web structure: connectance, omnivory
index, mean trophic level, and system throughput.

See also [Core API: Analysis & Network Indices](core.md#analysis-network-indices)
for `mixed_trophic_impact()` and `summarize_ecosim_output()`.


## Scenario Optimization

Risk-aware multi-objective scenario ranking helpers for coupled
Ecosim-Ecospace management experiments.

::: pypath.analysis.scenario_optimizer
    options:
      show_root_heading: true
      members_order: source
