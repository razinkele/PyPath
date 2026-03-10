# Changelog — pypath-ewe

All notable changes to the **pypath-ewe** core package.

Format: [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

<!--next-version-placeholder-->

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
