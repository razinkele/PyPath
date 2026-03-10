# Changelog — pypath-shiny

All notable changes to the **pypath-shiny** web frontend.

Format: [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

<!--next-version-placeholder-->

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
