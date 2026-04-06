# Changelog — pypath-shiny

All notable changes to the **pypath-shiny** web frontend.

Format: [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

<!--next-version-placeholder-->

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
