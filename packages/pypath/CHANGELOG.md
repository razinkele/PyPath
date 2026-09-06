# Changelog — pypath-ewe

All notable changes to the **pypath-ewe** core package.

Format: [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

<!--next-version-placeholder-->

## v0.5.0 (2026-09-06)

### Changed (breaking)

- **core/analysis.py**: `keystoneness_index()` now returns `n_groups` values instead of `n_groups + 1`, and `export_ecopath_to_dataframe()` emits group **names** rather than integer indices. Callers indexing the old padded array need updating.

### Fixed

- **core/analysis.py**, **core/indicators.py**, **core/plotting.py**: Converted from a 1-based Rpath layout that no real `Rpath` uses. Real arrays are 0-based of length `NUM_GROUPS`, `DC` is `(NUM_GROUPS + 1, NUM_LIVING)` with a trailing Import row, and `Landings`/`Discards` are `(NUM_GROUPS, NUM_GEARS)`. On the shipped example model `mixed_trophic_impacts()`, `flow_analysis()` and `plot_foodweb()` all raised `IndexError`; the Analysis page swallowed those and rendered zeros or an error onto the canvas.
- **core/indicators.py**: `Landings[i, 1:]` dropped gear 0, so a one-fleet model reported zero catch — mean trophic level of the catch came back `NaN` and catch/biomass `0.0`.
- **core/plotting.py**: `plot_trophic_spectrum()` failed silently, dropping the first living group (the primary producer) and counting a detritus group as living — a 22% error in total biomass on the example model.
- **core/plotting.py**: Food web node labels now use real group names instead of positional `G{i}` labels.
- **ibm/smelt.py**: Removed a dead `o2` that the per-individual, zone-specific `ind_o2` had superseded.
- **io/ewemdb.py**: Replaced a KeyError-probe assignment with an explicit column-membership check.

### Added

- **spatial/fishing.py**: `effort_multipliers()` turns a `SpatialFishing` allocation into per-patch, per-gear multipliers normalised to mean 1.0, so total fleet effort is conserved and a uniform allocation reproduces a run with no spatial fishing exactly.
- **spatial/integration.py**: `rsim_run_spatial(..., spatial_fishing=)` and `deriv_vector_spatial(..., effort_multiplier=)`, both optional and additive. MPA closures compose with an allocation by multiplication.

### Internal

- Cleared 181 ruff errors that had kept CI red since at least 2026-03-12.


## v0.4.2 (2026-04-06)

### Fixed

- **ibm/integration.py**: Fix 1-based → 0-based prey index and derivative offset (critical runtime bug)
- **core/ecosim_advanced.py**: Add missing `stanza_biomass=None` default arg to `RsimOutput`
- **core/ecosim.py**: Add `dead_idx` definition; move `stanza_biomass` to end with default; add `TYPE_CHECKING` imports for `EcotracerResult`, `FleetDynamicsResult`
- **core/forcing.py**: Fix `update_diet` return type annotation to `np.ndarray | None`
- **spatial/dispersal.py**: Guard division by zero with `safe_distances`
- **core/montecarlo.py**, **pedigree.py**, **sensitivity.py**: Add `TYPE_CHECKING` imports for `RpathParams`, `PedigreeConfig`
- **core/timeseries.py**, **spatial/ecospace_params.py**: Add `TYPE_CHECKING` import for `pandas`
- **io/ewemdb.py**: Add `TYPE_CHECKING` imports; tighten SQL identifier regex (no Unicode); replace `pd.read_sql` with cursor-based approach to avoid `RuntimeWarning`
- **io/biodata.py**: Add `TYPE_CHECKING` import for `TaxonomyData`
- **spatial/integration.py**: Add `TYPE_CHECKING` import for `MPAConfig`; bounds check for `habitat_capacity` shape
- **io/_access_writer.py**: Fix `Optional` param and type annotations for local variables
- **io/ewe_writer.py**: `Union` type annotation for writer variable
- **core/ecosim_advanced.py**: `range | list[int]` annotation for indices
- **spatial/habitat.py**: Use `np.clip` in geometric mean to avoid bias
- **spatial/connectivity.py**: Named constant `_KM_PER_DEGREE_LAT = 111.0`
- **ibm/predation.py**: Validate `optimal_prey_length` and `selectivity_sd > 0`
- **ibm/behavior.py**: Additive inertia bonus after normalization
- **spatial/external_flux.py**: Use context manager for `xr.open_dataset`
- **core/analysis.py**: `np.divide` with `where=` to suppress div-by-zero warnings
- **core/ecosim_deriv.py**: Cache resolved `INSTRUMENT_GROUPS` to prevent 574 `DeprecationWarnings` per simulation run (815 → 33 warnings per test run)
- **tests/test_import_diet.py**: Remove spurious `return` from test function



### Fixed
- Thornton-Lessem: NaN/overflow protection with input validation (CK1/CK4 in (0,1), CTO>CQ, CTL>CTM), exp() overflow clamping, isfinite check
- Division-by-zero guards on all `o2_lethal` divisions in egg mortality and smelt.py
- Hypoxic yolk-sac mortality now correctly sets `n_represented=0.0` before skipping (fixes mortality accounting)
- Sigmoid overflow clip in `growth_step_batch_ontogenetic` (matching smelt.py's -30/30 clip)
- Weak test assertions in oxygen tests replaced with unconditional checks
- Dead code removed in `growth_step` (identical if/else branches)
- `energy_reserve` docstring corrected from "dimensionless 0-1" to "grams-equivalent"

### Added
- `LifeStage` IntEnum (EGG=0, YOLK_SAC=1, LARVA=2, JUVENILE=3, ADULT=4) for self-documenting comparisons
- `baltic_defaults_zonal()` factory method (populates ZoneParams)
- `EggParams.eggs_per_cohort` field (replaces magic number 1e6)
- O2 behavioral avoidance integrated into Phase 5 zonal movement scoring
- `__post_init__` validation on EggParams, YolkSacParams, LarvalParams
- Zone forcing pre-computation cache in compute_step (performance)
- Warnings logged when O2 or temperature defaults are used (missing env_forcing keys)
- PRCC zero-variance residual warning
- Tests: juvenile transition, full lifecycle multi-step, senescent removal

## v0.4.0 (2026-03-22)

### Added

**Smelt IBM Early Life Stages — Complete Lifecycle Model**

Based on Drewes et al. (2025) *Ecological Modelling* 510:111313 and Keller et al. (2020) *J Fish Biol* 97(2):368-381.

- **Egg stage** — Temperature-dependent degree-day development (DD_hatch=149 °C·day, T₀=1.8°C), 3-source mortality (thermal/oxygen/background), per-zone spawning deposition
- **Yolk-sac stage** — Q10-scaled basal metabolism yolk depletion, first feeding transition with point-of-no-return starvation (4 days), Cushing match/mismatch now emergent from mechanistic processes
- **Larval bioenergetics** — Thornton-Lessem (Fish Bioenergetics 3.0) temperature dome for Cmax, ontogenetic sigmoid interpolation (Rs+Ra metabolism split, size-dependent assimilation efficiency), consumption blending from concentration-dependent Type II → adaptive foraging
- **Oxygen physiology** — Full lifecycle Pcrit-based metabolic scope reduction (Pcrit 4.0→2.0 mg/L egg→adult), lethal thresholds for early stages, stress-accelerated yolk depletion, behavioral O2 avoidance
- **Curonian Lagoon zonal model** — 3-zone spatial (river spawning/lagoon nursery/coastal feeding), zone-specific environmental forcing, passive drift for yolk-sac/larvae, ontogenetic habitat constraints, spawning migration
- **Calibration framework** — `calibrate_els()` wrapper for IBM-coupled parameter fitting, `lhs_sensitivity()` Latin Hypercube Sampling, `partial_rank_correlation()` PRCC analysis

**New parameter dataclasses:** `EggParams`, `YolkSacParams`, `LarvalParams`, `OxygenParams`, `ZoneParams`

**New factory method:** `SmeltParams.baltic_defaults_els()` — enables early life stages with literature defaults. `baltic_defaults()` unchanged (backward compatible).

**New `SuperIndividual` fields:** `life_stage` (0=egg, 1=yolk_sac, 2=larva, 3=juvenile, 4=adult), `degree_days`, `starvation_days`, `yolk_energy_kj`

**Population management:** Configurable super-individual cap (default 2000) with biomass-preserving cohort consolidation.

**67 new tests** (229 IBM tests total, 1502 total across all packages).

### Changed
- `compute_step()` restructured with life-stage routing (Phase 1a eggs, 1b yolk-sac, 1c ontogenetic bioenergetics)
- `growth_step_batch()` and `growth_step_batch_ontogenetic()` now accept array temperatures for per-zone thermal forcing

## v0.3.3 (2026-03-12)

### Added

**Time Series & Calibration (Phase 1)**
- `EweTimeSeries`, `EweTimeSeriesCollection` — time series data structures for biomass, catch, effort, fishing mortality
- `apply_timeseries_drivers()` — apply time series as Ecosim forcing
- `load_timeseries()` — load time series from CSV
- `fit_to_timeseries()` — SS fitting against reference time series (EwE's standard calibration approach)
- `CalibrationResult` — calibration output with fitted parameters and diagnostics
- `read_timeseries()` — read time series from EwE databases (4 tables: EcosimTimeSeries, EcosimTimeSeriesDataset, EcosimTimeSeriesGroup, EcosimTimeSeriesFleet)

**Mediation Functions (Phase 2)**
- `MediationShape`, `MediationLink`, `MediationCollection` — trophic mediation function data structures
- `make_positive_shape()`, `make_negative_shape()`, `make_ushape()` — shape constructors
- Mediation multipliers applied in Ecosim consumption kernel (`deriv_vector`)
- `read_mediation()` — read mediation shapes/weights from EwE databases
- Write support for mediation tables

**Monte Carlo / Pedigree (Phase 3)**
- `PedigreeConfig`, `ScalarDistribution`, `DietDistribution` — pedigree-based parameter distributions
- `build_distributions()`, `sample_parameters()`, `apply_sample()` — sampling engine
- `MCConfig`, `MCResult`, `run_montecarlo()` — Monte Carlo simulation with parallel support (joblib)
- `MorrisResult`, `SobolResult`, `SensitivityConfig`, `run_sensitivity()` — Morris OAT and Sobol variance-based sensitivity analysis (SALib)
- `read_pedigree()` — read pedigree tables from EwE databases

**Ecotracer (Phase 4)**
- `EcotracerParams`, `EcotracerResult` — contaminant tracking parameters and results
- `create_ecotracer_params()`, `ecotracer_deriv()`, `ecotracer_step()` — tracer mass balance equations coupled to Ecosim dynamics
- `read_ecotracer()` — read Ecotracer tables from EwE databases

**Fleet Dynamics & MSE (Phase 5)**
- `FleetEconParams`, `FleetDynamicsResult` — fleet economic parameters and dynamics results
- `create_fleet_econ_params()`, `fleet_dynamics_step()`, `apply_quota_caps()` — effort response to profit, TAC allocation
- `read_fleet_dynamics()` — read fleet scenario tables from EwE databases

**Advanced Ecospace I/O (Phase 6)**
- Extended `EcospaceReadResult` with 8 new fields (driver_layers, migration_maps, monthly_maps, weight_layers, etc.)
- Full 16-table Ecospace write support (was 2)
- MPA zone support: `MPAConfig`, `MPAZone`, `read_mpa_config()`, `write_mpa()`
- Capacity driver runtime integration (scalar weight application)

**Ecological Indicators (Phase 7)**
- `FlowAnalysis`, `flow_analysis()` — Ulanowicz ascendency framework (TST, ascendency, capacity, overhead)
- `finn_cycling_index()` — Leontief inverse method
- `transfer_efficiency()` — per-trophic-level transfer efficiency
- `EcosystemIndicators`, `ecosystem_indicators()` — MTL, Marine Trophic Index, Shannon diversity, Kempton's Q
- `ecosystem_indicators_timeseries()` — dynamic indicators from Ecosim output
- `SystemMaturityIndices`, `system_maturity()` — Odum's ecosystem development indicators (P/R, B/TST, mean path length)
- Morris and Sobol sensitivity analysis

**Value Chain Economics I/O (Phase 8)**
- `ValueChainData` — dataclass for 21 c-prefix EwE value chain tables
- `read_value_chain()` — read value chain economics tables from EwE databases
- Write support for all 21 value chain tables (CSV and Access backends)
- `write_ewemdb()` now accepts `value_chain=` parameter

**Taxonomy Integration (Phase 9)**
- `TaxonomyData`, `TaxonomyRecord` — taxonomy data structures
- `read_taxonomy()` — read EcopathTaxon, EcopathGroupTaxon, EcopathStanzaTaxon tables
- Write support for taxonomy tables

**EwE Database Table Coverage**
- Now covers 72 of 84 EwE database tables (86% coverage, up from 25%)
- Schema definitions for all tables in `_ewe_schema.py`

## v0.3.2 (2026-03-11)

### Fixed
- **EwE 6.6+ Schema Compatibility**: Rewrote export schema to match native
  EwE 6.6+ desktop database format. Exported `.eweaccdb` files now load
  correctly in EwE 6.6+ without errors.
  - Renamed ~40 columns to match EwE 6.6+ names (`PB` -> `ProdBiom`,
    `QB` -> `ConsBiom`, `EE` -> `EcoEfficiency`, `BioAcc` -> `BiomAcc`,
    `DetInput` -> `DtImports`, etc.)
  - Renamed ~15 tables (`EcosimGroupInfo` -> `EcosimScenarioGroup`,
    `EcosimForcing` -> `EcosimShape`, `Ecospace*` -> `EcospaceScenario*`)
  - Replaced template database with full 88-table EwE 6.6+ schema
  - Added type coercion for Access integer enum fields (`UnitCurrency`,
    `UnitTime`)
  - Added `Sequence`-based ordering in reader for stable round-trips
  - Fixed `Required field` errors caused by NULL values in Access columns
    with Required constraints

## v0.3.1 (2026-03-10)

### Added
- **EwE Export**: New `write_ewemdb()` function exports PyPath models back to
  native EwE `.eweaccdb` format (Access via pyodbc) or `.ewecsv.zip` (CSV
  bundle fallback for cross-platform use). Supports Ecopath groups, diet,
  fleets, stanzas, Ecosim scenarios, and Ecospace spatial data.
- `ecosim_scenario_from_ewemdb()` — load complete Ecosim scenarios from native EwE
  databases with vulnerability overrides, foraging time, forced biomass, fishing effort,
  and environmental forcing
- `autofix` module — automatic crash diagnosis (`diagnose_crash_causes`) and parameter
  repair (`autofix_parameters`) for Ecosim stability
- `growth_step_batch()` — vectorized IBM bioenergetics for batch processing of all
  super-individuals (replaces per-individual scalar loop)
- KDTree-based nearest-neighbor salinity sampling in `marine_data.SalinityLoader`
  (replaces O(n*m) brute-force)
- `connectivity` module — `haversine_distance`, `validate_adjacency_symmetry`,
  `get_connectivity_graph_stats` for spatial patch analysis
- LT2022 EwE database integration tests (27 tests covering Ecopath loading, Ecosim
  scenario building, and simulation runs for scenarios 1 and 16)
- `test_autofix.py` — 8 unit tests for crash diagnostics and autofix
- `test_connectivity.py` — 11 unit tests for spatial connectivity functions
- `test_ibm_bioenergetics.py` — batch growth step tests

### Fixed
- Boolean array casting in `growth_step_batch` (`is_mature & ...` now uses
  `np.asarray(is_mature, dtype=bool)` to avoid numpy `bitwise_and` TypeError)
- Vectorized diet normalization and detritus consumption loops in Ecopath solver

### Changed
- Vectorized NetCDF salinity loading with broadcasting (no per-patch loop)
- IBM `SmeltIBM.compute_step()` Phase 1 uses batch growth step for all individuals

## v0.3.0 (2026-03-09)

### Added
- Numba JIT compilation for Ecosim ODE solver (43% speedup)
- Sparse link-array format for food web iteration
- Parallel spatial patch computation via ThreadPoolExecutor
- NumPy-accelerated Gauss solver in Ecopath
- Vectorized IBM predation mortality
- EMODnet marine data integration (Ecospace Wizard)
- defusedxml for XXE-safe XML parsing in EcoBase

### Changed
- Narrowed exception handling in ewemdb.py to specific types
- Standardized docstrings to NumPy format in stanzas.py, adjustments.py
- Converted print() calls to logging in optimization.py, ecosim_advanced.py, prebalance.py
- Removed 18 unused constants from constants.py

### Fixed
- TYPE_CHECKING imports for marine_data.py (F821)
- Ruff lint config moved to [tool.ruff.lint]

## v0.2.2 (2025-12-19)

- Initial release (see commit history)
