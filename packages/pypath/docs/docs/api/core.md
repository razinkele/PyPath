# Core API Reference

The core module implements the Ecopath mass-balance solver, Ecosim dynamic
simulation, and supporting subsystems (stanzas, forcing, optimization).

## Ecopath (Mass-Balance)

::: pypath.core.ecopath
    options:
      show_root_heading: true
      members_order: source

## Parameters

::: pypath.core.params
    options:
      show_root_heading: true
      members_order: source

## Ecosim (Dynamic Simulation)

The Ecosim engine runs time-dynamic food web simulations using foraging
arena theory. Key entry points:

- `rsim_scenario()` — create a scenario from a balanced Ecopath model
- `rsim_run()` — run the simulation with RK4 or Adams-Bashforth integration

### Integration methods

| Method | Description |
|--------|-------------|
| `"RK4"` | 4th-order Runge-Kutta. Stable, self-starting. Good for short runs. |
| `"AB"` | Adams-Bashforth 2-step (matches Rpath). Uses 1 month RK4 warmup then AB2: `B_{n+1} = B_n + dt/2 * (3*f_n - f_{n-1})`. Includes dynamic fast equilibrium for NoIntegrate groups and Rpath-style biomass bounds. Recommended for calibration and comparison with EwE/Rpath. |

### Consumption formula

The consumption kernel uses the full Rpath foraging arena functional response:

```
Q = QQbase * PDY * PYY^(HandleSwitch*COUPLED) *
    DD / (DD - 1 + ((1-Hself)*PYY + Hself*HandleSuite)^(HandleSwitch*COUPLED)) *
    VV / (VV - 1 + (1-Sself)*PDY + Sself*PredSuite)
```

Where `PredSuite` and `HandleSuite` are pooled competitor biomass sums
across all predator-prey links sharing the same prey or predator.

### Foraging time

Predators dynamically adjust foraging time using the Rpath formula:

```
Ftime_new = 0.1 + 0.9 * Ftime_old * ((1 - FtimeAdj) + FtimeAdj * QBopt / (FoodGain / B))
```

Capped at 2.0. Groups with `FtimeAdj = 0` do not adjust foraging time.

### NoIntegrate groups (fast equilibrium)

Groups flagged as NoIntegrate (typically detritus, meiobenthos, and other
fast-turnover groups) use instantaneous equilibrium instead of ODE
integration when running with the AB method:

```
biomeq = TotGain / (TotLoss / B)
B_new = 0.5 * biomeq + 0.5 * B_old
```

This matches Rpath's SORWT=0.5 smoothing.

::: pypath.core.ecosim
    options:
      show_root_heading: true
      members_order: source

## ODE Derivatives

The consumption kernel and derivative vector used by the Ecosim ODE solver.

Key functions:

- `deriv_vector()` — compute dB/dt for all groups at a given state
- `_compute_consumption_python()` — foraging arena consumption matrix (Rpath-compatible)
- `compute_biomeq()` — fast equilibrium for NoIntegrate groups
- `integrate_rk4()` — single RK4 step
- `integrate_ab()` — single AB2 step with Rpath-style biomass bounds

::: pypath.core.ecosim_deriv
    options:
      show_root_heading: true
      members_order: source

## Advanced Ecosim (State Forcing & Diet Rewiring)

State-variable forcing and dynamic diet rewiring extensions for Ecosim.

::: pypath.core.ecosim_advanced
    options:
      show_root_heading: true
      members_order: source

## Stanzas (Multi-Stanza Groups)

::: pypath.core.stanzas
    options:
      show_root_heading: true

## Adjustments

::: pypath.core.adjustments
    options:
      show_root_heading: true

## Forcing

::: pypath.core.forcing
    options:
      show_root_heading: true

## Optimization

::: pypath.core.optimization
    options:
      show_root_heading: true

## Plotting

::: pypath.core.plotting
    options:
      show_root_heading: true

## Analysis & Network Indices

Mixed Trophic Impacts, network indices (connectance, omnivory), and
Ecosim output summary statistics.

::: pypath.core.analysis
    options:
      show_root_heading: true
      members_order: source

## Autofix (Stability Diagnostics)

Automatic parameter diagnostics and repair routines to prevent Ecosim
simulation crashes and improve model stability.

Key functions:

- `diagnose_crash_causes()` — identify critical issues (EE > 1, incomplete
  diets) and warnings (low biomass, high vulnerability, extreme QQ)
- `autofix_parameters()` — automatically cap vulnerabilities, enforce minimum
  biomass, reduce extreme density dependence
- `validate_and_fix_scenario()` — combined validation and repair in one step
- `AutofixResult` — dataclass with applied fixes and diagnostic info

See the [Autofix Guide](../guides/autofix.md) for practical usage.

::: pypath.core.autofix
    options:
      show_root_heading: true
      members_order: source

## Sparse Link Array

Compressed prey-predator link lists for efficient consumption kernel
computation in sparse food webs.

::: pypath.core.link_array
    options:
      show_root_heading: true

## Constants

Physical and biological constants used throughout PyPath.

::: pypath.core.constants
    options:
      show_root_heading: true

## Time Series & Calibration

Time series data structures for observed data and calibration against
EwE reference series.

::: pypath.core.timeseries
    options:
      show_root_heading: true
      members_order: source

## Calibration

Sum-of-squares fitting against reference time series.

::: pypath.core.calibration
    options:
      show_root_heading: true
      members_order: source

## Mediation Functions

Trophic mediation functions that modify predator-prey vulnerability
based on the biomass of a mediating group.

::: pypath.core.mediation
    options:
      show_root_heading: true
      members_order: source

## Pedigree System (Phase 1B.1)

Data quality tracking using EwE pedigree indices (0-4 scale mapping to
HIGH/MEDIUM/LOW confidence). Integrates with Monte Carlo sampling for
uncertainty quantification.

::: pypath.core.pedigree
    options:
      show_root_heading: true
      members_order: source

**Key Classes:**
- `PedigreeQuality` — Enum mapping EwE pedigree indices to quality levels
- `EwePedigreeConfig` — Configuration for pedigree parameters (index, min, max)
- `PedigreeContainer` — Multi-group pedigree management with validation

## Monte Carlo Sampling

::: pypath.core.montecarlo
    options:
      show_root_heading: true
      members_order: source

## Taxonomic Linkage System (Phase 1B.2)

Hierarchical taxonomic linking for species identification across external
databases (WoRMS, ITIS, FishBase). Supports multi-level taxonomy assignment,
validation, and integration with biodiversity data systems.

::: pypath.core.taxonomy
    options:
      show_root_heading: true
      members_order: source

**Key Classes:**
- `TaxonLink` — Full taxonomic hierarchy (kingdom through species) with external IDs
- `TaxonomyLinkage` — Multi-group taxonomy container with WoRMS integration stubs
- `resolve_worms_link()` — Look up WoRMS taxon by name (stub function)
- `batch_resolve_worms_taxonomy()` — Batch resolution for multiple groups

**Integration with biodata.py:**
- `integrate_taxonomy_with_biodata()` — Merge taxonomy with FishBase/WoRMS data
- `resolve_worms_taxonomy()` — Fetch and store WoRMS records
- `sync_taxonomy_with_rpath()` — Sync TaxonomyLinkage with RpathParams

## Size Distribution System (Phase 1B.3)

Population size structure and age-size conversions for age-structured
modeling and Individual-Based Model (IBM) frameworks.

::: pypath.core.size
    options:
      show_root_heading: true
      members_order: source

**Key Classes:**
- `SizeClass` — Size interval definition with midpoint, width, and contains logic
- `SizeDistribution` — Population size distribution (PSD) container with statistics
- `SizeStructure` — Multi-group size distributions with aggregation

**Von Bertalanffy Functions:**
- `vb_length(age, Loo, K, t0)` — Age-to-length conversion
- `vb_age(length, Loo, K, t0)` — Length-to-age conversion
- `age_at_size_class()` — Estimate mean age from size interval
- `estimate_psd_from_vb()` — Generate theoretical PSD from VBGF parameters

## Fleet & Discard Dynamics (Phase 1B.4)

Dynamic fishing fleet management with gear types, catchability, effort
allocation, and discard routing to multiple fates. Highest complexity
Phase 1B module with integration to Ecosim fishing mortality.

::: pypath.core.fleet
    options:
      show_root_heading: true
      members_order: source

**Key Classes:**
- `GearType` — Enum for 6 gear types (SEINE, NET, TRAP, TRAWL, HOOK, OTHER)
- `FleetGroup` — Fleet definition with catchability matrix, fuel efficiency, base effort
- `DiscardPath` — Discard routing with 4 fates (lost, predation, detritus, benthos)
- `FleetDynamics` — Multi-fleet container with allocation and validation

**Allocation Functions:**
- `allocate_catch()` — Distribute catch among prey based on catchability
- `allocate_discards()` — Route discards to specified fates
- `estimate_catch_value()` — Economic value estimation (stub)
- `estimate_fuel_cost()` — Fuel cost calculation (stub)

## Nutrient Cycling System (Phase 1B.5)

Ecosystem-level nutrient cycling (N, P, C) with excretion, uptake kinetics,
and stoichiometric relationships. Implements Monod saturation for producer
nutrient limitation.

::: pypath.core.nutrients
    options:
      show_root_heading: true
      members_order: source

**Key Classes:**
- `NutrientElement` — Enum for N, P, C with molecular weights
- `NutrientContent` — Elemental composition (%) with validation
- `NutrientExcretion` — Excretion rates with fecal/dissolved partitioning
- `NutrientUptake` — Producer kinetics (half-saturation, max rate)
- `NutrientCycle` — Ecosystem-level cycling with mass balance

**Key Functions:**
- `uptake_rate_monod()` — Monod saturation function for producer uptake
- `validate_nutrient_cycle()` — Consistency checks across groups
- `estimate_excretion_from_pb()` — Heuristic excretion estimation from P/B

## Enhanced Landings System (Phase 1B.6)

Catch and landings tracking with temporal (seasonal) distribution, landing
types, and aggregation. Interfaces with dynamic Ecosim monthly dispatch.

::: pypath.core.landings
    options:
      show_root_heading: true
      members_order: source

**Key Classes:**
- `LandingType` — Enum for 5 catch categories (COMMERCIAL, SUBSISTENCE, RECREATIONAL, UNREPORTED, DISCARDS)
- `LandingRecord` — Month-level catch tracking with optional value and fleet ID
- `LandingSeasonality` — Monthly catch distribution patterns (12-element array)
- `LandingsData` — Multi-record container with aggregation and validation

**Key Functions:**
- `estimate_seasonality_from_records()` — Infer monthly patterns from data
- `aggregate_landings()` — Flexible aggregation (by group, year, type, season)
- `dispatch_to_monthly_steps()` — Convert annual landings to monthly for Ecosim

## Sensitivity Analysis

Morris OAT screening and Sobol variance-based sensitivity analysis.

::: pypath.core.sensitivity
    options:
      show_root_heading: true
      members_order: source

## Ecotracer (Contaminant Tracking)

Track contaminant concentrations through the food web coupled to
Ecosim biomass dynamics.

::: pypath.core.ecotracer
    options:
      show_root_heading: true
      members_order: source

## Fleet Dynamics

Dynamic fleet effort response to profitability with quota management.

::: pypath.core.fleet_dynamics
    options:
      show_root_heading: true
      members_order: source

## Ecological Indicators

Ecosystem-level indicators including Ulanowicz flow analysis,
Finn Cycling Index, transfer efficiency, and system maturity indices.

::: pypath.core.indicators
    options:
      show_root_heading: true
      members_order: source

