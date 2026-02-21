# Package Split Design: pypath-ewe + pypath-shiny

**Date:** 2026-02-21
**Status:** Approved
**Goal:** Split the monolithic PyPath repository into two independently publishable packages — a core algorithms library and a Shiny web frontend — while keeping them in a single monorepo.

---

## Package Summary

| | `pypath-ewe` | `pypath-shiny` |
|---|---|---|
| **PyPI name** | `pypath-ewe` | `pypath-shiny` |
| **Python import** | `import pypath` | `import pypath_shiny` |
| **Location** | `packages/pypath/` | `packages/pypath-shiny/` |
| **Purpose** | Core Ecopath/Ecosim/Ecospace algorithms, I/O, spatial modeling | Shiny web frontend for Rpath workflows |
| **Core deps** | numpy, pandas, scipy, matplotlib | shiny, shinyswatch, httpx, uvicorn, **pypath-ewe** |
| **Version** | 0.3.0 (signals restructuring) | 0.3.0 |
| **Target** | PyPI + conda-forge | PyPI + conda-forge |

Both names verified available on PyPI (404 on `pypi.org/simple/`).

---

## Monorepo Directory Structure

```
PyPath/
├── packages/
│   ├── pypath/                        # pypath-ewe package
│   │   ├── pyproject.toml
│   │   ├── src/pypath/
│   │   │   ├── __init__.py
│   │   │   ├── core/                  # ecopath, ecosim, stanzas, params, etc.
│   │   │   ├── io/                    # ecobase, ewemdb, biodata, utils
│   │   │   ├── spatial/               # ecospace, dispersal, grids, habitat
│   │   │   └── analysis/              # prebalance
│   │   ├── tests/                     # ~600 core algorithm tests
│   │   │   ├── conftest.py
│   │   │   ├── test_ecopath.py
│   │   │   ├── test_ecosim.py
│   │   │   ├── ... (all core tests)
│   │   │   └── data/rpath_reference/  # reference test data
│   │   ├── docs/                      # MkDocs API documentation
│   │   │   ├── mkdocs.yml
│   │   │   └── docs/
│   │   │       ├── index.md
│   │   │       ├── getting-started.md
│   │   │       ├── api/
│   │   │       │   ├── core.md
│   │   │       │   ├── io.md
│   │   │       │   ├── spatial.md
│   │   │       │   └── analysis.md
│   │   │       └── examples/
│   │   │           ├── basic-model.md
│   │   │           ├── ecosim-run.md
│   │   │           └── spatial.md
│   │   └── example_model_data/        # example models for testing/tutorials
│   │
│   └── pypath-shiny/                  # pypath-shiny package
│       ├── pyproject.toml
│       ├── src/pypath_shiny/
│       │   ├── __init__.py
│       │   ├── app.py                 # Main Shiny app
│       │   ├── config.py              # UI configuration
│       │   ├── logger.py              # App logging
│       │   ├── pages/                 # All 13 page modules
│       │   │   ├── home.py
│       │   │   ├── data_import.py
│       │   │   ├── ecopath.py
│       │   │   ├── ecosim.py
│       │   │   ├── ecospace.py
│       │   │   ├── prebalance.py
│       │   │   ├── multistanza.py
│       │   │   ├── forcing_demo.py
│       │   │   ├── diet_rewiring_demo.py
│       │   │   ├── optimization_demo.py
│       │   │   ├── analysis.py
│       │   │   ├── results.py
│       │   │   ├── about.py
│       │   │   ├── utils.py
│       │   │   └── validation.py
│       │   └── static/                # CSS, SVG icons
│       ├── tests/                     # ~40 app tests
│       │   ├── conftest.py
│       │   ├── test_shiny_app.py
│       │   ├── test_shiny_pages.py
│       │   ├── test_shiny_reactive.py
│       │   └── ui/
│       └── STYLE_GUIDE.md
│
├── README.md                          # Repo-level overview pointing to both packages
├── CONTRIBUTING.md
├── CHANGELOG.md
├── DEPLOYMENT.md
└── docs/
    ├── plans/                         # Project-level planning docs (stays here)
    └── archive/                       # Historical docs (stays here)
```

---

## Core Package: pypath-ewe

### pyproject.toml

```toml
[project]
name = "pypath-ewe"
version = "0.3.0"
description = "Python implementation of Ecopath with Ecosim (EwE) for food web modeling"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
interactive = [
    "plotly>=5.0",
    "networkx>=3.0",
]
spatial = [
    "geopandas>=0.12",
    "folium>=0.14",
    "shapely>=2.0",
]
biodata = [
    "pyworms>=0.2.1",
    "pyobis>=0.3.0",
    "requests>=2.28",
]
numba = [
    "numba>=0.57",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
all = [
    "pypath-ewe[interactive,spatial,biodata,numba,docs,dev]",
]

[project.urls]
Homepage = "https://github.com/razinkele/PyPath"
Documentation = "https://razinkele.github.io/PyPath/"
Repository = "https://github.com/razinkele/PyPath"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

### What moves here

From current repo:
- `src/pypath/` → `packages/pypath/src/pypath/` (no internal changes)
- `example_model_data/` → `packages/pypath/example_model_data/`
- Core tests → `packages/pypath/tests/`
- `docs/ECOSPACE_*.md`, `docs/BIODATA_QUICKSTART.md` → incorporated into MkDocs docs

### API Documentation (MkDocs)

- Theme: `mkdocs-material`
- Auto-generated API reference from docstrings via `mkdocstrings[python]`
- Deploys to GitHub Pages via `mkdocs gh-deploy`
- Sections: Getting Started, API Reference (core, io, spatial, analysis), Examples

---

## Frontend Package: pypath-shiny

### pyproject.toml

```toml
[project]
name = "pypath-shiny"
version = "0.3.0"
description = "Shiny web frontend for PyPath EwE food web modeling"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    "pypath-ewe[interactive,spatial]>=0.3.0",
    "shiny>=1.0.0",
    "shinyswatch>=0.7.0",
    "httpx>=0.24",
    "uvicorn>=0.23",
]

[project.optional-dependencies]
biodata = [
    "pypath-ewe[biodata]>=0.3.0",
]
dev = [
    "pytest>=7.0",
]

[project.urls]
Homepage = "https://github.com/razinkele/PyPath"
Repository = "https://github.com/razinkele/PyPath"

[project.scripts]
pypath-shiny = "pypath_shiny.app:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
pypath_shiny = ["static/*"]
```

### What moves here

From current repo:
- `app/app.py` → `packages/pypath-shiny/src/pypath_shiny/app.py`
- `app/config.py` → `packages/pypath-shiny/src/pypath_shiny/config.py`
- `app/logger.py` → `packages/pypath-shiny/src/pypath_shiny/logger.py`
- `app/pages/` → `packages/pypath-shiny/src/pypath_shiny/pages/`
- `app/static/` → `packages/pypath-shiny/src/pypath_shiny/static/`
- `app/STYLE_GUIDE.md` → `packages/pypath-shiny/STYLE_GUIDE.md`
- App tests → `packages/pypath-shiny/tests/`

### Import Migration

All internal imports change from the try/except fallback pattern to direct imports:

```python
# Before (current):
try:
    from app.config import UI
    from app.pages.utils import is_balanced_model
except ModuleNotFoundError:
    from config import UI
    from pages.utils import is_balanced_model

# After:
from pypath_shiny.config import UI
from pypath_shiny.pages.utils import is_balanced_model
```

The try/except fallback pattern is removed entirely — the package is always installed.

### Entry Point

CLI launch: `pypath-shiny` command runs the Shiny app.

Requires a `main()` function in `app.py`:

```python
def main():
    """Launch the PyPath Shiny application."""
    import uvicorn
    uvicorn.run("pypath_shiny.app:app", host="0.0.0.0", port=8000, reload=True)
```

---

## Test Split

### Core tests → `packages/pypath/tests/`

All test files EXCEPT those matching `test_shiny_*`, `test_app_*`, and `ui/`:

- `test_ecopath.py`, `test_ecosim.py`, `test_stanzas.py`, `test_analysis.py`
- `test_rpath_*.py` (12+ files)
- `test_spatial_*.py`, `test_dispersal.py`, `test_habitat.py`, etc.
- `test_forcing.py`, `test_diet_rewiring.py`, `test_optimization_*.py`
- `test_ecobase.py`, `test_ewemdb.py`, `test_biodata*.py`
- `test_plotting.py`
- `test_detritus_*.py`, `test_instrumentation_*.py`
- `data/rpath_reference/` (reference test data)

### App tests → `packages/pypath-shiny/tests/`

- `test_shiny_app.py`
- `test_shiny_pages.py`
- `test_shiny_reactive.py`
- `test_shiny_rpath_integration.py`
- `test_shinyswatch_integration.py`
- `test_app_import.py`
- `ui/test_prebalance_modal.py`

### conftest.py

Split into two:
- Core `conftest.py`: shared fixtures for spatial scenarios, model params
- App `conftest.py`: fixtures for Shiny app testing, mock reactive state

---

## Version Strategy

- Both packages start at **0.3.0** to signal the restructuring
- `pypath-shiny` pins `pypath-ewe>=0.3.0` as minimum
- Previous `pypath-ecopath` 0.2.2 on PyPI remains available for existing users
- Add deprecation notice to `pypath-ecopath` pointing to `pypath-ewe`

---

## Conda-forge Strategy

Two independent feedstocks:
- `pypath-ewe` feedstock — builds from `packages/pypath/pyproject.toml`
- `pypath-shiny` feedstock — builds from `packages/pypath-shiny/pyproject.toml`, depends on `pypath-ewe`

Both use standard `python-build` recipe (no special build steps needed).

---

## Existing Documentation Migration

| Source | Destination |
|--------|-------------|
| `docs/ECOSPACE_*.md` (5 files) | Incorporated into `packages/pypath/docs/docs/` MkDocs site |
| `docs/BIODATA_QUICKSTART.md` | Incorporated into MkDocs site |
| `docs/RPATH_REFERENCE_TESTING.md` | `packages/pypath/docs/docs/` |
| `docs/TESTING_BIODATA.md` | `packages/pypath/docs/docs/` |
| `docs/archive/` | Stays at repo root |
| `docs/plans/` | Stays at repo root |
| `app/STYLE_GUIDE.md` | `packages/pypath-shiny/STYLE_GUIDE.md` |
| Root `README.md` | Updated to point to both packages |

---

## What Does NOT Change

- Core library source code (`src/pypath/`) — no internal API changes
- Import paths for core library users (`from pypath.core.ecosim import ...`)
- Git repository URL (`razinkele/PyPath`)
- License
