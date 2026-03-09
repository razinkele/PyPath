# Changelog — pypath-ewe

All notable changes to the **pypath-ewe** core package.

Format: [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

<!--next-version-placeholder-->

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
