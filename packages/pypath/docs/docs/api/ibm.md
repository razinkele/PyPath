# Individual-Based Model (IBM) API Reference

The IBM module tracks populations of super-individuals with explicit
body size, age, energy reserves, and spatial location. Each
super-individual represents a cohort of real organisms. The IBM
integrates into Ecosim's derivative loop: IBM groups override the
standard ODE derivative for their functional group while the rest of the
food web runs normally.

See the [IBM Parameterization Guide](../guides/ibm-parameterization.md)
for practical tuning advice.

## Base Classes

Core data structures: `SuperIndividual`, `IBMGroup` (abstract base),
`IBMStepResult`, and `SpatialContext`.

::: pypath.ibm.base
    options:
      show_root_heading: true
      members_order: source

## Bioenergetics

Wisconsin bioenergetics model: Q10 temperature scaling, allometric
metabolism, assimilation, and growth. Drives individual weight and
energy reserve changes each timestep.

Key functions:

- `growth_step()` — single-individual energy budget and weight update
- `growth_step_batch()` — vectorized batch processing for all individuals
  (uses NumPy arrays, significantly faster for large populations)
- `q10_temperature_factor()` — Q10 temperature scaling
- `metabolism()` — weight-dependent metabolic rate with temperature correction
- `allometric_length()` — weight-to-length conversion
- `BioenergParams` — dataclass with all bioenergetics parameters

::: pypath.ibm.bioenergetics
    options:
      show_root_heading: true
      members_order: source

## Predation

Size-structured predation mortality using a log-normal selectivity
curve. Distributes Ecosim group-level mortality across super-individuals
based on body length.

::: pypath.ibm.predation
    options:
      show_root_heading: true
      members_order: source

## Reproduction

Stochastic spawning with weight-dependent fecundity and Cushing
match/mismatch larval survival. Mature females produce eggs; surviving
larvae become new recruit super-individuals.

::: pypath.ibm.reproduction
    options:
      show_root_heading: true
      members_order: source

## Behavior (Spatial Movement & Adaptive Foraging)

Score-based stochastic patch selection over a sparse adjacency graph,
plus profitability-based adaptive prey allocation.

::: pypath.ibm.behavior
    options:
      show_root_heading: true
      members_order: source

## EwE Integration

Bridge between the IBM and Ecosim/Ecospace derivative systems.
`apply_ibm_to_derivative()` overrides the Ecosim derivative for IBM
groups and subtracts IBM consumption from prey derivatives.

::: pypath.ibm.integration
    options:
      show_root_heading: true
      members_order: source

## Smelt Implementation

Concrete `IBMGroup` implementation for Baltic smelt (Osmerus eperlanus),
combining all sub-modules into a 5-phase `compute_step()`. Use as a
template for building custom IBM species.

::: pypath.ibm.smelt
    options:
      show_root_heading: true
      members_order: source
