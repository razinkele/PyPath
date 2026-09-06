# Changelog — pypath-shiny

All notable changes to the **pypath-shiny** web frontend.

Format: [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

<!--next-version-placeholder-->

## v0.4.2 (2026-09-06)

### Fixed

- **Ecospace**: The Spatial Fishing controls (allocation method, gravity alpha, port distance decay, habitat targeting) now drive the simulation instead of only redrawing a preview. Added a Target Groups multi-select for `SpatialFishing.target_groups`; leaving it empty follows total biomass, as before.
- **Ecospace**: Fixed a decorator that had been captured by a helper inserted above the run handler, which left the Run Spatial Simulation button doing nothing.
- **Ecospace**: Removed a pruning block that discarded 9 of 28 hexagonal patches; habitat CSV parsing now sniffs for a header instead of eating the first data row.
- **Ecopath**: Table edits are applied through `set_patch_fn`; the previous `input.<id>_cell_edit` effects do not exist in Shiny 1.7, so edits to the model, diet and fisheries tables were silently discarded. Rows resolve by group name rather than position.
- **All pages**: Eight `@render.download` handlers returned a `str`, which Shiny treats as a file path; `download_params` raised `WinError 123`. All now yield.
- **Ecosim**: The Biomass Forcing card now writes into `forcing.ForcedPrey`; previously it had no effect on the scenario.
- **Analysis**: Food web, trophic spectrum and network indices work again, following the indexing fixes in pypath-ewe 0.5.0.
- **logger.py**: `configure_logging()` clears existing handlers and creates the log directory on every run.

### Changed

- Requires `pypath-ewe >= 0.5.0`.


## v0.4.1 (2026-04-06)

### Fixed
- **pages/analysis.py**: Return `ui.div()` instead of `None` from `render.ui` (prevents blank panel crash)
- **pages/ecosim.py**: Guard `scen is not None` before accessing `scen.params`

### Tests
- Comprehensive test coverage: 143 new tests (280 total, up from 137)
- New test files covering: config dataclasses + IBM/SmeltParams alignment, all 5 validation
  functions, `format_dataframe_for_display` (sentinel masking, rounding, type labels, stanza/remarks),
  `create_cell_styles` (no_data priority, QB non-applicable, remarks, stanza styles),
  `get_model_info` (balanced/unbalanced models), `load_rpath_diagnostics` (corrupted JSON, missing
  CSVs), ecopath helpers (`_get_groups_from_model`, `_recreate_params_from_model` with diet
  reconstruction), `_get_version`, tutorial helpers (`_code_block`, `_step_card` with badge),
  `_resolve_repo_root` (env var, walk-up), UI render assertions for all page modules,
  demo page server signature checks



### Changed
- Updated to use pypath-ewe v0.3.3 (time series calibration, mediation,
  Monte Carlo, ecotracer, fleet dynamics, ecological indicators, value
  chain I/O, taxonomy integration — 86% EwE table coverage)

## v0.3.2 (2026-03-11)

### Changed
- Updated to use pypath-ewe v0.3.2 (EwE 6.6+ schema compatibility fix)

## v0.3.1 (2026-03-10)

### Added
- Ecospace Data Wizard improvements with KDTree-based environmental sampling

### Changed
- Updated to use pypath-ewe v0.3.1 core improvements

## v0.3.0 (2026-03-09)

### Added
- Ecospace Data Wizard page with EMODnet integration
- Sidebar navigation with Bootstrap Icons

### Changed
- Replaced inline hasattr() checks with is_balanced_model()/is_rpath_params() utilities
- Replaced brittle parents[5] with _resolve_repo_root() in prebalance.py
- Ruff lint config moved to [tool.ruff.lint]

## v0.2.2 (2025-12-19)

- Initial release (see commit history)
