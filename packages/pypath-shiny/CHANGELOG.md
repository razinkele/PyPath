# Changelog — pypath-shiny

All notable changes to the **pypath-shiny** web frontend.

Format: [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

<!--next-version-placeholder-->

## v1.3.1 (2026-11-12)

### Bug Fixes

- **stanzas**: EwE's age table and Ford-Walford plus group (EwE rule)
  ([`e32412a`](https://github.com/razinkele/PyPath/commit/e32412a97a20363a74744efbe4dc2a06572d2be6))


## v1.3.0 (2026-11-09)

### Bug Fixes

- Warn about detritus that flows to itself; Shiny defaults it to 0
  ([`a824df8`](https://github.com/razinkele/PyPath/commit/a824df897ae754d839e92dd90d2b512ff9bd699d))

- **shiny**: Home example detritus no longer flows to itself
  ([`05ecdf4`](https://github.com/razinkele/PyPath/commit/05ecdf498d5633fba4b0a34c775af2b6eab8e221))

### Features

- **shiny**: Run an imported EwE database's Ecosim scenario; choose the fast-group rule
  ([`4e5fc9e`](https://github.com/razinkele/PyPath/commit/4e5fc9e06b032d919c3d16d46509b2d482490a5f))


## v1.2.0 (2026-11-04)

### Features

- **shiny**: Phase 3 - Plotly charts, deck.gl Ecospace maps, browser-check fixes
  ([#46](https://github.com/razinkele/PyPath/pull/46),
  [`032a56a`](https://github.com/razinkele/PyPath/commit/032a56abc99c52d7f5161d8f83f87114c5d24dce))


## v1.1.0 (2026-11-03)

### Bug Fixes

- **shiny**: Data Import panel was invisible (sidebar outside a layout)
  ([`3c0b71f`](https://github.com/razinkele/PyPath/commit/3c0b71fcdca3a4622854af5b0eee79b6ac366f1d))

- **shiny**: Ecosim "Create Scenario" failed; charts sat below the window
  ([`49b66ee`](https://github.com/razinkele/PyPath/commit/49b66ee0ab19f08a4200529ded41d4abb1235d78))

- **shiny**: Give every Plotly widget a height (Ecosim Summary charts were ~1600 px)
  ([`1877eac`](https://github.com/razinkele/PyPath/commit/1877eac2be8d4d6c3588a0f31b0686805a4344a6))

- **shiny**: Handle proxied health endpoint paths
  ([`0daf001`](https://github.com/razinkele/PyPath/commit/0daf001519a6e40c28c332bc921033b76a7f46a0))

- **shiny**: Keep fleet discard fates when balancing
  ([`4b9c728`](https://github.com/razinkele/PyPath/commit/4b9c7280a99be59e1a1a295337be5c0f74ae8a6e))

- **shiny**: Readable header text in format_params_table
  ([`bdd118d`](https://github.com/razinkele/PyPath/commit/bdd118ddc8fa91f02028c3cf2a8f94faf979d2d5))

- **shiny**: Repair the Phase 2 great_tables integration
  ([`1c13b33`](https://github.com/razinkele/PyPath/commit/1c13b338105d84f2641fcffb019f8a82be88d409))

- **shiny**: Restore pypath-ewe dependency floor >=1.0.0
  ([`f08f58d`](https://github.com/razinkele/PyPath/commit/f08f58da421e0c32d1535871874a36244188cf7d))

- **shiny**: Show a message instead of an empty table in @render.ui outputs
  ([`f77c777`](https://github.com/razinkele/PyPath/commit/f77c777d2bcd5b5659606807891cb48396d440e6))

- **shiny**: Source health endpoint from pypath-ewe
  ([`5c960f0`](https://github.com/razinkele/PyPath/commit/5c960f0dd7bf1037254eb51f98c0c8d83bc819a0))

### Features

- Convert render.data_frame to great_tables in 3 pages
  ([`2da25e7`](https://github.com/razinkele/PyPath/commit/2da25e7d510225e4a477275ad1bec524ea4b56d2))

- **shiny**: Add /api/health endpoint
  ([`c45b751`](https://github.com/razinkele/PyPath/commit/c45b75161c0230dcda4e176f81e31130883d1735))

- **shiny**: Add GT formatters service for professional tables
  ([`d24a729`](https://github.com/razinkele/PyPath/commit/d24a729494e4b2e082e589ac896b16b3eb3b4b0d))

- **shiny**: Add modern UI stack dependencies
  ([`96621c6`](https://github.com/razinkele/PyPath/commit/96621c67a2f79dcec2a53bba0ab1368bd09aeb20))

- **shiny**: Ecopath trophic-level and EE charts on Plotly
  ([`857ca4e`](https://github.com/razinkele/PyPath/commit/857ca4e85099eb5c7480aca6752dffd6718c2d3d))

- **shiny**: Integrate great_tables into data import page
  ([`6600b5d`](https://github.com/razinkele/PyPath/commit/6600b5dbb87356f5f7ad08d99d19f3ea813e68c2))

- **shiny**: Integrate great_tables into ecopath page
  ([`e1c94bb`](https://github.com/razinkele/PyPath/commit/e1c94bb20284575a8820fa8a955c9195dabc60f8))

- **shiny**: Integrate great_tables into prebalance diagnostics tables
  ([`a80d0e8`](https://github.com/razinkele/PyPath/commit/a80d0e8355690eaa66ea67f1098e9aeb16ca6211))

- **shiny**: Integrate great_tables into results page and add time_series formatter
  ([`1ec5918`](https://github.com/razinkele/PyPath/commit/1ec591816d98025385093bab1e1a87c11984ef54))

- **shiny**: Interactive effort preview with one line per fleet
  ([`32e88aa`](https://github.com/razinkele/PyPath/commit/32e88aae688a6d8d97a589083906494fef28b93b))

- **shiny**: Interactive Plotly biomass and catch plots on the Ecosim page
  ([`92e2264`](https://github.com/razinkele/PyPath/commit/92e2264f1d94e3b48aa52082e59a3fdcb696e54b))

- **shiny**: Interactive Plotly charts on the Analysis page
  ([`56b827e`](https://github.com/razinkele/PyPath/commit/56b827ecad69003f8ee140c6872a4f48d9a82f49))

- **shiny**: Interactive Plotly charts on the Ecosim Summary tab
  ([`b1776bc`](https://github.com/razinkele/PyPath/commit/b1776bcfff2a83e7739f2456ca26d6076da75ba4))

- **shiny**: Phase 3 (WIP) - Plotly figures, UI components, home redesign
  ([`c472b6e`](https://github.com/razinkele/PyPath/commit/c472b6eaef3ddd9d39f5d36c4bddbc561c53ecc6))

- **shiny**: Results trophic charts and food web on Plotly; palette/style fixes
  ([`6ebae01`](https://github.com/razinkele/PyPath/commit/6ebae018f773c1d520f33833dbf56dd7232155a8))


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
