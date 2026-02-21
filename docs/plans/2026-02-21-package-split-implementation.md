# Package Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the monolithic PyPath repository into two independently publishable packages (`pypath-ewe` and `pypath-shiny`) organized as a monorepo under `packages/`.

**Architecture:** Create `packages/pypath/` (core algorithms, `import pypath`) and `packages/pypath-shiny/` (Shiny web frontend, `import pypath_shiny`). Each package gets its own `pyproject.toml`, `src/` layout, and `tests/`. The core library source code and import paths are unchanged. The Shiny app module is renamed from `app` to `pypath_shiny` with all internal imports migrated from the try/except fallback pattern to absolute `pypath_shiny.*` imports.

**Tech Stack:** Python 3.10+, setuptools, pytest, MkDocs + mkdocs-material + mkdocstrings

**Design doc:** `docs/plans/2026-02-21-package-split-design.md`

---

## Task 1: Create packages/pypath directory scaffold

**Files:**
- Create: `packages/pypath/pyproject.toml`
- Create: `packages/pypath/src/pypath/.gitkeep` (placeholder, removed in Task 2)

**Step 1: Create the directory structure**

```bash
mkdir -p packages/pypath/src packages/pypath/tests packages/pypath/example_model_data packages/pypath/docs/docs
```

**Step 2: Create pyproject.toml for pypath-ewe**

Create `packages/pypath/pyproject.toml` with this exact content:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pypath-ewe"
version = "0.3.0"
description = "Python implementation of Ecopath with Ecosim (EwE) for food web modeling"
readme = "../../README.md"
license = {text = "MIT"}
authors = [
    {name = "PyPath Development Team"}
]
keywords = ["ecology", "ecosystem", "food web", "ecopath", "ecosim", "modeling"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]
requires-python = ">=3.10"
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

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
markers = [
    "integration: marks tests that require internet connection and real API calls",
    "slow: marks tests as slow",
    "worms: marks tests that use WoRMS API",
    "obis: marks tests that use OBIS API",
    "fishbase: marks tests that use FishBase API",
]
timeout = 300

[tool.black]
line-length = 88
target-version = ["py310", "py311", "py312"]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
```

**Step 3: Verify directory exists**

Run: `ls packages/pypath/`
Expected: `pyproject.toml  src  tests  example_model_data  docs`

**Step 4: Commit**

```bash
git add packages/pypath/pyproject.toml
git commit -m "chore: scaffold packages/pypath with pyproject.toml for pypath-ewe"
```

---

## Task 2: Move core library source into packages/pypath

**Files:**
- Move: `src/pypath/` → `packages/pypath/src/pypath/`

**Step 1: Move the source tree**

```bash
# Move the entire src/pypath directory
mv src/pypath packages/pypath/src/pypath
```

**Step 2: Update version in __init__.py**

Edit `packages/pypath/src/pypath/__init__.py` line 7: change `__version__ = "0.2.2"` to `__version__ = "0.3.0"`.

**Step 3: Remove the now-empty src directory**

```bash
rmdir src
```

**Step 4: Verify the move**

Run: `ls packages/pypath/src/pypath/core/`
Expected: `__init__.py  adjustments.py  analysis.py  autofix.py  constants.py  ecopath.py  ecosim.py  ecosim_advanced.py  ecosim_deriv.py  forcing.py  optimization.py  params.py  plotting.py  stanzas.py`

Run: `python -c "import sys; sys.path.insert(0, 'packages/pypath/src'); from pypath.core.ecopath import rpath; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move src/pypath to packages/pypath/src/pypath"
```

---

## Task 3: Move core tests into packages/pypath

**Files:**
- Move: `tests/` core test files → `packages/pypath/tests/`
- Move: `tests/data/` → `packages/pypath/tests/data/`
- Move: `tests/scripts/` → `packages/pypath/tests/scripts/`
- Move: `tests/integration/` → `packages/pypath/tests/integration/`

**Step 1: Move test data and scripts first**

```bash
mv tests/data packages/pypath/tests/data
mv tests/scripts packages/pypath/tests/scripts
mv tests/integration packages/pypath/tests/integration
```

**Step 2: Move all core test files**

Move every test file EXCEPT the app/shiny tests. The app tests to KEEP in `tests/` are:
- `test_shiny_app.py`
- `test_shiny_pages.py`
- `test_shiny_reactive.py`
- `test_shiny_rpath_integration.py`
- `test_shinyswatch_integration.py`
- `test_app_import.py`
- `ui/` directory

Move all OTHER test files:

```bash
# Core ecopath/ecosim tests
mv tests/test_ecopath.py packages/pypath/tests/
mv tests/test_ecopath_input_conversion.py packages/pypath/tests/
mv tests/test_ecosim.py packages/pypath/tests/
mv tests/test_ecosim_model_type.py packages/pypath/tests/
mv tests/test_ecosim_qlink.py packages/pypath/tests/
mv tests/test_ecosim_stanzas.py packages/pypath/tests/
mv tests/test_stanzas.py packages/pypath/tests/
mv tests/test_adjustments.py packages/pypath/tests/
mv tests/test_analysis.py packages/pypath/tests/
mv tests/test_forcing.py packages/pypath/tests/
mv tests/test_debug_forcing.py packages/pypath/tests/
mv tests/test_advanced_features.py packages/pypath/tests/

# Detritus tests
mv tests/test_detritus_consumption.py packages/pypath/tests/
mv tests/test_detritus_fish_detfrac_addition.py packages/pypath/tests/
mv tests/test_detritus_fish_discard_reproducer.py packages/pypath/tests/
mv tests/test_detritus_nointegrate.py packages/pypath/tests/

# Rpath compatibility tests
mv tests/test_rpath_compatibility.py packages/pypath/tests/
mv tests/test_rpath_diagnostics_meta.py packages/pypath/tests/
mv tests/test_rpath_ecosim_core.py packages/pypath/tests/
mv tests/test_rpath_macrobenthos_ab_parity.py packages/pypath/tests/
mv tests/test_rpath_nointegrate.py packages/pypath/tests/
mv tests/test_rpath_qq_provided_strict.py packages/pypath/tests/
mv tests/test_rpath_seabirds_regression.py packages/pypath/tests/
mv tests/test_rpath_m0_persistence.py packages/pypath/tests/
mv tests/test_rpath_reference.py packages/pypath/tests/

# Spatial tests
mv tests/test_dispersal.py packages/pypath/tests/
mv tests/test_environmental.py packages/pypath/tests/
mv tests/test_grid_creation.py packages/pypath/tests/
mv tests/test_habitat.py packages/pypath/tests/
mv tests/test_hexagonal_grids.py packages/pypath/tests/
mv tests/test_irregular_grids.py packages/pypath/tests/
mv tests/test_spatial_fishing.py packages/pypath/tests/
mv tests/test_spatial_integration.py packages/pypath/tests/
mv tests/test_spatial_validation.py packages/pypath/tests/
mv tests/test_spatial_ecosim_integration.py packages/pypath/tests/
mv tests/test_spatial_performance.py packages/pypath/tests/

# Optimization tests
mv tests/test_optimization_unit.py packages/pypath/tests/
mv tests/test_optimization_integration.py packages/pypath/tests/
mv tests/test_optimization_scenarios.py packages/pypath/tests/

# Diet/rewiring
mv tests/test_diet_rewiring.py packages/pypath/tests/

# IO tests
mv tests/test_biodata.py packages/pypath/tests/
mv tests/test_biodata_integration.py packages/pypath/tests/
mv tests/test_biodata_workflow.py packages/pypath/tests/
mv tests/test_ecobase.py packages/pypath/tests/
mv tests/test_ewemdb.py packages/pypath/tests/
mv tests/test_file_format_support.py packages/pypath/tests/
mv tests/test_import_diet.py packages/pypath/tests/

# Pre-balance tests
mv tests/test_pb_validation_fix.py packages/pypath/tests/
mv tests/test_pb_simple.py packages/pypath/tests/
mv tests/test_prebalance_rpath_summary.py packages/pypath/tests/
mv tests/test_prebalance_rpath_verify.py packages/pypath/tests/

# Ecosim detailed tests
mv tests/test_rsim_detritus_link_coverage.py packages/pypath/tests/
mv tests/test_rsim_integration_fish_discard_effect.py packages/pypath/tests/
mv tests/test_seabirds_monthly_m0_flag.py packages/pypath/tests/
mv tests/test_seabirds_q_diff_threshold.py packages/pypath/tests/
mv tests/test_seabirds_termwise_comparison.py packages/pypath/tests/

# Instrumentation tests
mv tests/test_instrumentation_ab_direct.py packages/pypath/tests/
mv tests/test_instrumentation_deprecation.py packages/pypath/tests/
mv tests/test_instrumentation_indices.py packages/pypath/tests/

# Other core tests
mv tests/test_output_shape.py packages/pypath/tests/
mv tests/test_q_matrix_shapes.py packages/pypath/tests/
mv tests/test_lt_model.py packages/pypath/tests/
mv tests/test_plotting.py packages/pypath/tests/
mv tests/test_backward_compatibility.py packages/pypath/tests/
```

**Step 3: Create core conftest.py**

Create `packages/pypath/tests/conftest.py` with only the core fixtures (no sys.path hacks for app):

```python
"""Shared pytest configuration for pypath-ewe core tests.

The pypath package should be installed via `pip install -e packages/pypath`.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_1d_grid


@pytest.fixture
def spatial_scenario():
    """Create a balanced 5-group Ecosim scenario for spatial tests.

    Returns (scenario, rpath_params) tuple.
    Model: Phyto (producer) -> Zoo -> Fish (fished) + Det + Fleet.
    """
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "Det", "Fleet"],
        types=[1, 0, 0, 2, 3],
    )

    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8

    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9

    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 5.0
    params.model.loc[2, "EE"] = 0.5

    params.model.loc[3, "Biomass"] = 100.0

    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model.loc[4, "BioAcc"] = np.nan
    params.model.loc[4, "Unassim"] = np.nan

    params.model["Det"] = 1.0
    params.model.loc[4, "Det"] = np.nan

    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    params.model.loc[2, "Fleet"] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    scenario = rsim_scenario(model, params, years=range(1, 11))
    return scenario, params


@pytest.fixture
def single_patch_ecospace(spatial_scenario):
    """1-patch EcospaceParams -- spatial should equal non-spatial."""
    scenario, _ = spatial_scenario
    ng = scenario.params.NUM_GROUPS + 1  # +1 for index-0 "Outside"
    grid = create_1d_grid(n_patches=1)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, 1)),
        habitat_capacity=np.ones((ng, 1)),
        dispersal_rate=np.zeros(ng),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
    )


@pytest.fixture
def simple_ecospace(spatial_scenario):
    """3-patch EcospaceParams with mild dispersal for dynamics tests."""
    scenario, _ = spatial_scenario
    ng = scenario.params.NUM_GROUPS + 1
    grid = create_1d_grid(n_patches=3, spacing=1.0)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, 3)),
        habitat_capacity=np.ones((ng, 3)),
        dispersal_rate=np.full(ng, 2.0),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
    )
```

**Step 4: Copy tests/__init__.py**

```bash
cp tests/__init__.py packages/pypath/tests/__init__.py
```

**Step 5: Verify core tests run**

Run: `pip install -e packages/pypath && pytest packages/pypath/tests/test_ecopath.py -x -q`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: move core tests to packages/pypath/tests"
```

---

## Task 4: Move example_model_data into packages/pypath

**Files:**
- Move: `example_model_data/` → `packages/pypath/example_model_data/`

**Step 1: Move the directory**

```bash
mv example_model_data/* packages/pypath/example_model_data/
rmdir example_model_data
```

**Step 2: Check for references to example_model_data in source or tests**

Search all `.py` files for `example_model_data` and update any relative paths. Common pattern is `Path(__file__).parent.parent / "example_model_data"` — these need to be updated to point to the new location within the package.

Run: `grep -r "example_model_data" packages/pypath/src packages/pypath/tests --include="*.py" -l`

Update any references found.

**Step 3: Verify**

Run: `ls packages/pypath/example_model_data/`
Expected: `README.md  detritus_fate.csv  diet.csv  discard.csv  discard_fate.csv  landing.csv  model.csv  stanza_groups.csv  stanza_individual.csv`

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move example_model_data to packages/pypath"
```

---

## Task 5: Create packages/pypath-shiny directory scaffold

**Files:**
- Create: `packages/pypath-shiny/pyproject.toml`

**Step 1: Create the directory structure**

```bash
mkdir -p packages/pypath-shiny/src/pypath_shiny/pages
mkdir -p packages/pypath-shiny/src/pypath_shiny/static
mkdir -p packages/pypath-shiny/tests/ui
```

**Step 2: Create pyproject.toml for pypath-shiny**

Create `packages/pypath-shiny/pyproject.toml` with this exact content:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pypath-shiny"
version = "0.3.0"
description = "Shiny web frontend for PyPath EwE food web modeling"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "PyPath Development Team"}
]
keywords = ["ecology", "ecosystem", "ecopath", "ecosim", "shiny", "dashboard"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Framework :: Shiny",
]

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

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
pypath_shiny = ["static/*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.black]
line-length = 88
target-version = ["py310", "py311", "py312"]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "W"]
ignore = ["E501"]
```

**Step 3: Verify**

Run: `ls packages/pypath-shiny/`
Expected: `pyproject.toml  src  tests`

**Step 4: Commit**

```bash
git add packages/pypath-shiny/pyproject.toml
git commit -m "chore: scaffold packages/pypath-shiny with pyproject.toml"
```

---

## Task 6: Move app source into packages/pypath-shiny and rename module

**Files:**
- Move: `app/app.py` → `packages/pypath-shiny/src/pypath_shiny/app.py`
- Move: `app/config.py` → `packages/pypath-shiny/src/pypath_shiny/config.py`
- Move: `app/logger.py` → `packages/pypath-shiny/src/pypath_shiny/logger.py`
- Move: `app/pages/` → `packages/pypath-shiny/src/pypath_shiny/pages/`
- Move: `app/static/` → `packages/pypath-shiny/src/pypath_shiny/static/`
- Move: `app/icon.jpg`, `app/icon_no_text.jpg` → `packages/pypath-shiny/src/pypath_shiny/static/`
- Create: `packages/pypath-shiny/src/pypath_shiny/__init__.py`
- Move: `app/STYLE_GUIDE.md` → `packages/pypath-shiny/STYLE_GUIDE.md`

**Step 1: Move source files**

```bash
# Move main app files
mv app/app.py packages/pypath-shiny/src/pypath_shiny/app.py
mv app/config.py packages/pypath-shiny/src/pypath_shiny/config.py
mv app/logger.py packages/pypath-shiny/src/pypath_shiny/logger.py

# Move pages directory
mv app/pages/* packages/pypath-shiny/src/pypath_shiny/pages/

# Move static files
mv app/static/* packages/pypath-shiny/src/pypath_shiny/static/
mv app/icon.jpg packages/pypath-shiny/src/pypath_shiny/static/
mv app/icon_no_text.jpg packages/pypath-shiny/src/pypath_shiny/static/

# Move style guide
mv app/STYLE_GUIDE.md packages/pypath-shiny/STYLE_GUIDE.md
```

**Step 2: Create __init__.py for pypath_shiny**

Create `packages/pypath-shiny/src/pypath_shiny/__init__.py`:

```python
"""PyPath Shiny - Web frontend for PyPath EwE food web modeling."""

__version__ = "0.3.0"
```

**Step 3: Remove old app directory**

```bash
# Remove remaining files and empty dirs
rm -rf app/
```

**Step 4: Verify the structure**

Run: `ls packages/pypath-shiny/src/pypath_shiny/`
Expected: `__init__.py  app.py  config.py  logger.py  pages  static`

Run: `ls packages/pypath-shiny/src/pypath_shiny/pages/`
Expected: All 16 page .py files

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move app to packages/pypath-shiny/src/pypath_shiny"
```

---

## Task 7: Migrate all pypath_shiny internal imports

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/app.py`
- Modify: `packages/pypath-shiny/src/pypath_shiny/pages/*.py` (all page files)
- Modify: `packages/pypath-shiny/src/pypath_shiny/pages/__init__.py`

This is the most critical task. All try/except import fallback patterns must be replaced with direct `pypath_shiny.*` imports.

**Step 1: Fix app.py imports**

In `packages/pypath-shiny/src/pypath_shiny/app.py`:

Remove the entire try/except import block (lines 53-92) and the sys.path setup (lines 46-48). Replace with:

```python
from pypath_shiny.config import UI
from pypath_shiny.pages import (
    about,
    analysis,
    data_import,
    diet_rewiring_demo,
    ecopath,
    ecosim,
    ecospace,
    forcing_demo,
    home,
    multistanza,
    optimization_demo,
    prebalance,
    results,
)
```

Also add a `main()` function at the bottom of the file (before the `app = App(...)` line is fine, but after the `server` function):

```python
def main():
    """Launch the PyPath Shiny application."""
    import uvicorn
    uvicorn.run("pypath_shiny.app:app", host="0.0.0.0", port=8000, reload=True)
```

**Step 2: Fix pages/__init__.py imports**

In `packages/pypath-shiny/src/pypath_shiny/pages/__init__.py`, replace all `from app.pages.X` / `from pages.X` try/except patterns with direct imports. The lazy-import pattern for optional heavy dependencies can stay, but the fallback path references should change.

**Step 3: Fix every page module**

For EACH file in `packages/pypath-shiny/src/pypath_shiny/pages/`, apply this transformation:

```python
# BEFORE (every page has this pattern):
try:
    from app.config import DEFAULTS, PLOTS, THRESHOLDS
    from app.logger import get_logger
    from app.pages.utils import is_balanced_model
except ModuleNotFoundError:
    from config import DEFAULTS, PLOTS, THRESHOLDS
    from logger import get_logger
    from pages.utils import is_balanced_model

# AFTER (clean absolute imports):
from pypath_shiny.config import DEFAULTS, PLOTS, THRESHOLDS
from pypath_shiny.logger import get_logger
from pypath_shiny.pages.utils import is_balanced_model
```

Files to update (all in `packages/pypath-shiny/src/pypath_shiny/pages/`):
- `about.py`
- `analysis.py`
- `data_import.py`
- `diet_rewiring_demo.py`
- `ecopath.py`
- `ecosim.py`
- `ecospace.py`
- `forcing_demo.py`
- `home.py`
- `multistanza.py`
- `optimization_demo.py`
- `prebalance.py`
- `results.py`
- `utils.py`
- `validation.py`

Also fix any lazy imports inside functions (Pattern C from the analysis):

```python
# BEFORE (in ecosim.py helper):
try:
    from app.pages.utils import is_balanced_model
except ModuleNotFoundError:
    from pages.utils import is_balanced_model

# AFTER:
from pypath_shiny.pages.utils import is_balanced_model
```

**Step 4: Verify imports resolve**

Run: `pip install -e packages/pypath && pip install -e packages/pypath-shiny && python -c "from pypath_shiny.app import app; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: migrate all pypath_shiny imports to absolute package paths"
```

---

## Task 8: Move app tests into packages/pypath-shiny

**Files:**
- Move: `tests/test_shiny_app.py` → `packages/pypath-shiny/tests/`
- Move: `tests/test_shiny_pages.py` → `packages/pypath-shiny/tests/`
- Move: `tests/test_shiny_reactive.py` → `packages/pypath-shiny/tests/`
- Move: `tests/test_shiny_rpath_integration.py` → `packages/pypath-shiny/tests/`
- Move: `tests/test_shinyswatch_integration.py` → `packages/pypath-shiny/tests/`
- Move: `tests/test_app_import.py` → `packages/pypath-shiny/tests/`
- Move: `tests/ui/test_prebalance_modal.py` → `packages/pypath-shiny/tests/ui/`
- Create: `packages/pypath-shiny/tests/conftest.py`
- Move: `tests/README_SHINY_TESTS.md` → `packages/pypath-shiny/tests/`

**Step 1: Move app test files**

```bash
mv tests/test_shiny_app.py packages/pypath-shiny/tests/
mv tests/test_shiny_pages.py packages/pypath-shiny/tests/
mv tests/test_shiny_reactive.py packages/pypath-shiny/tests/
mv tests/test_shiny_rpath_integration.py packages/pypath-shiny/tests/
mv tests/test_shinyswatch_integration.py packages/pypath-shiny/tests/
mv tests/test_app_import.py packages/pypath-shiny/tests/
mv tests/ui/test_prebalance_modal.py packages/pypath-shiny/tests/ui/
mv tests/README_SHINY_TESTS.md packages/pypath-shiny/tests/
```

**Step 2: Update imports in moved test files**

In each test file, update imports from `app.*` to `pypath_shiny.*`:

```python
# BEFORE:
from app.app import app
from app.logger import logger

# AFTER:
from pypath_shiny.app import app
from pypath_shiny.logger import logger
```

For `test_app_import.py`, update the importlib references:

```python
# BEFORE:
mod = importlib.import_module("app.app")
mod = importlib.import_module("app.logger")

# AFTER:
mod = importlib.import_module("pypath_shiny.app")
mod = importlib.import_module("pypath_shiny.logger")
```

**Step 3: Create app conftest.py**

Create `packages/pypath-shiny/tests/conftest.py`:

```python
"""Shared pytest configuration for pypath-shiny tests.

The pypath-shiny package should be installed via `pip install -e packages/pypath-shiny`.
"""
```

**Step 4: Create __init__.py files**

```bash
touch packages/pypath-shiny/tests/__init__.py
touch packages/pypath-shiny/tests/ui/__init__.py
```

**Step 5: Remove old tests directory leftovers**

```bash
# Remove the old tests directory (should be empty or nearly empty)
rm -f tests/conftest.py tests/__init__.py
rm -rf tests/ui
rm -rf tests
```

**Step 6: Verify app tests run**

Run: `pytest packages/pypath-shiny/tests/test_app_import.py -x -q`
Expected: 2 PASS

**Step 7: Commit**

```bash
git add -A
git commit -m "refactor: move app tests to packages/pypath-shiny/tests"
```

---

## Task 9: Remove old root pyproject.toml and update root config

**Files:**
- Remove: `pyproject.toml` (root-level, replaced by per-package configs)
- Modify: `README.md`

**Step 1: Remove root pyproject.toml**

The root pyproject.toml for `pypath-ecopath` is no longer needed. Remove it:

```bash
rm pyproject.toml
```

**Step 2: Update README.md**

Update the root README.md to point to both packages. Add a section near the top:

```markdown
## Packages

This repository contains two independently installable packages:

| Package | PyPI | Description |
|---------|------|-------------|
| [pypath-ewe](packages/pypath/) | `pip install pypath-ewe` | Core Ecopath/Ecosim/Ecospace algorithms |
| [pypath-shiny](packages/pypath-shiny/) | `pip install pypath-shiny` | Shiny web frontend for PyPath |

### Development Install

```bash
# Install both packages in development mode
pip install -e packages/pypath[all]
pip install -e packages/pypath-shiny[dev]
```
```

Update version references from `0.2.2` to `0.3.0`.

**Step 3: Verify no dangling references**

Run: `grep -r "pypath-ecopath" packages/ --include="*.py" --include="*.toml" -l`
Expected: no matches (old package name should not appear)

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove root pyproject.toml, update README for monorepo"
```

---

## Task 10: Update CI workflows for monorepo

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/ci-shiny-smoke.yml`

**Step 1: Update main CI workflow**

Edit `.github/workflows/ci.yml`. Change the install and test steps:

```yaml
      - name: Install core package
        run: |
          python -m pip install --upgrade pip
          pip install -e 'packages/pypath[dev]'

      - name: Ruff Lint
        run: ruff check packages/

      - name: Black check
        run: black --check packages/

      - name: Type check (mypy)
        run: mypy packages/pypath/src --config-file packages/pypath/pyproject.toml || true

      - name: Run core tests
        run: |
          pytest packages/pypath/tests -q -m "not integration and not slow" --maxfail=1 --disable-warnings
```

**Step 2: Update Shiny smoke workflow**

Edit `.github/workflows/ci-shiny-smoke.yml`. Change install and run steps:

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e 'packages/pypath[interactive,spatial]'
          pip install -e 'packages/pypath-shiny'
          pip install pytest playwright
          python -m playwright install --with-deps chromium

      - name: Run shinyswatch integration tests
        run: |
          python -m pytest -q packages/pypath-shiny/tests/test_shinyswatch_integration.py

      - name: Start Shiny server
        run: |
          nohup pypath-shiny > server.log 2>&1 &
          echo $! > server.pid
          for i in $(seq 1 30); do
            if curl -sSf http://127.0.0.1:8000 >/dev/null; then
              echo "server up"
              break
            fi
            sleep 1
          done
```

**Step 3: Verify YAML is valid**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('OK')"`
Expected: `OK` (requires PyYAML installed, otherwise just check syntax manually)

**Step 4: Commit**

```bash
git add -A
git commit -m "ci: update workflows for monorepo package layout"
```

---

## Task 11: Set up MkDocs API documentation for pypath-ewe

**Files:**
- Create: `packages/pypath/docs/mkdocs.yml`
- Create: `packages/pypath/docs/docs/index.md`
- Create: `packages/pypath/docs/docs/getting-started.md`
- Create: `packages/pypath/docs/docs/api/core.md`
- Create: `packages/pypath/docs/docs/api/io.md`
- Create: `packages/pypath/docs/docs/api/spatial.md`
- Create: `packages/pypath/docs/docs/api/analysis.md`

**Step 1: Create mkdocs.yml**

Create `packages/pypath/docs/mkdocs.yml`:

```yaml
site_name: PyPath EwE Documentation
site_description: Python implementation of Ecopath with Ecosim (EwE) for food web modeling
site_url: https://razinkele.github.io/PyPath/
repo_url: https://github.com/razinkele/PyPath
repo_name: razinkele/PyPath

theme:
  name: material
  features:
    - navigation.sections
    - navigation.expand
    - search.suggest
    - content.code.copy
  palette:
    - scheme: default
      primary: teal
      accent: teal

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: ["../src"]
          options:
            show_source: true
            show_root_heading: true
            heading_level: 3

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - API Reference:
      - Core (Ecopath/Ecosim): api/core.md
      - I/O: api/io.md
      - Spatial (Ecospace): api/spatial.md
      - Analysis: api/analysis.md
```

**Step 2: Create index.md**

Create `packages/pypath/docs/docs/index.md`:

```markdown
# PyPath EwE

Python implementation of Ecopath with Ecosim (EwE) for food web modeling.

## Installation

```bash
pip install pypath-ewe
```

### Optional dependencies

```bash
pip install pypath-ewe[spatial]      # Ecospace spatial modeling
pip install pypath-ewe[interactive]  # Plotly interactive plots
pip install pypath-ewe[biodata]      # Species data from WoRMS/OBIS
pip install pypath-ewe[all]          # Everything
```

## Quick Example

```python
from pypath import create_rpath_params, rpath, rsim_scenario, rsim_run

# Create a simple 3-group model
params = create_rpath_params(
    groups=["Phytoplankton", "Zooplankton", "Detritus"],
    types=[1, 0, 2],
)
# ... set biomass, PB, QB, diet matrix ...

# Balance the model
model = rpath(params)

# Run dynamic simulation
scenario = rsim_scenario(model, params, years=range(1, 51))
output = rsim_run(scenario)
```
```

**Step 3: Create API reference pages**

Create `packages/pypath/docs/docs/api/core.md`:

```markdown
# Core API Reference

## Ecopath (Mass-Balance)

::: pypath.core.ecopath

## Ecosim (Dynamic Simulation)

::: pypath.core.ecosim

## Parameters

::: pypath.core.params

## Stanzas (Multi-Stanza Groups)

::: pypath.core.stanzas

## Adjustments

::: pypath.core.adjustments

## Forcing

::: pypath.core.forcing
```

Create `packages/pypath/docs/docs/api/io.md`:

```markdown
# I/O API Reference

## EcoBase

::: pypath.io.ecobase

## EwE Database (MDBX)

::: pypath.io.ewemdb

## Biological Data

::: pypath.io.biodata

## Utilities

::: pypath.io.utils
```

Create `packages/pypath/docs/docs/api/spatial.md`:

```markdown
# Spatial API Reference (Ecospace)

::: pypath.spatial
```

Create `packages/pypath/docs/docs/api/analysis.md`:

```markdown
# Analysis API Reference

## Pre-Balance Diagnostics

::: pypath.analysis.prebalance
```

Create `packages/pypath/docs/docs/getting-started.md`:

```markdown
# Getting Started

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

```bash
pip install pypath-ewe
```

## Your First Model

See the [quick example](index.md#quick-example) on the home page.

## Development Setup

```bash
git clone https://github.com/razinkele/PyPath.git
cd PyPath
pip install -e packages/pypath[all]
pytest packages/pypath/tests -q
```
```

**Step 4: Verify MkDocs builds**

Run: `pip install mkdocs mkdocs-material "mkdocstrings[python]" && cd packages/pypath/docs && mkdocs build --strict`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add -A
git commit -m "docs: add MkDocs API documentation for pypath-ewe"
```

---

## Task 12: Migrate existing documentation

**Files:**
- Move: `docs/ECOSPACE_*.md` (5 files) → incorporated into MkDocs or kept at repo root
- Move: `docs/BIODATA_QUICKSTART.md` → incorporated into MkDocs
- Keep: `docs/plans/` at repo root
- Keep: `docs/archive/` at repo root

**Step 1: Create example/guide pages from existing docs**

Create `packages/pypath/docs/docs/examples/` directory:

```bash
mkdir -p packages/pypath/docs/docs/examples
```

Copy relevant content from existing docs into MkDocs example pages. Create simplified versions:

Create `packages/pypath/docs/docs/examples/basic-model.md` — adapt content from existing README examples.

Create `packages/pypath/docs/docs/examples/spatial.md` — adapt content from `docs/ECOSPACE_USER_GUIDE.md`.

**Step 2: Update mkdocs.yml nav to include examples**

Add to the `nav` section in `packages/pypath/docs/mkdocs.yml`:

```yaml
  - Examples:
      - Basic Model: examples/basic-model.md
      - Spatial Modeling: examples/spatial.md
```

**Step 3: Keep plan and archive docs at repo root**

`docs/plans/` and `docs/archive/` stay at the repo root — no changes needed.

**Step 4: Commit**

```bash
git add -A
git commit -m "docs: migrate existing documentation into MkDocs structure"
```

---

## Task 13: Final verification and cleanup

**Files:**
- Modify: `.gitignore` (if needed for new build artifacts)

**Step 1: Reinstall both packages in dev mode**

```bash
pip install -e packages/pypath[all]
pip install -e packages/pypath-shiny[dev]
```

**Step 2: Run core tests**

Run: `pytest packages/pypath/tests -q -m "not integration and not slow" --maxfail=3`
Expected: ~600 tests PASS

**Step 3: Run app tests**

Run: `pytest packages/pypath-shiny/tests -q --maxfail=3`
Expected: App tests PASS

**Step 4: Verify imports work**

```bash
python -c "from pypath import rpath, rsim_run; print('pypath OK')"
python -c "from pypath_shiny.app import app; print('pypath_shiny OK')"
python -c "import pypath; print(pypath.__version__)"  # Should print 0.3.0
python -c "import pypath_shiny; print(pypath_shiny.__version__)"  # Should print 0.3.0
```

**Step 5: Verify no old imports remain**

```bash
# No try/except app.config fallbacks in pypath_shiny
grep -r "from app\." packages/pypath-shiny/src --include="*.py" -l
# Expected: no matches

# No old package name references
grep -r "pypath-ecopath" packages/ --include="*.toml" --include="*.py" -l
# Expected: no matches
```

**Step 6: Clean up any remaining empty directories**

```bash
# Check for orphaned files
ls src/ 2>/dev/null  # Should not exist
ls app/ 2>/dev/null  # Should not exist
ls tests/ 2>/dev/null  # Should not exist
```

**Step 7: Verify the final directory structure**

```
PyPath/
├── packages/
│   ├── pypath/                        # pypath-ewe
│   │   ├── pyproject.toml
│   │   ├── src/pypath/
│   │   ├── tests/
│   │   ├── docs/
│   │   └── example_model_data/
│   └── pypath-shiny/                  # pypath-shiny
│       ├── pyproject.toml
│       ├── src/pypath_shiny/
│       ├── tests/
│       └── STYLE_GUIDE.md
├── docs/
│   ├── plans/
│   └── archive/
├── scripts/
├── deploy/
├── .github/workflows/
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── DEPLOYMENT.md
```

**Step 8: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup after package split"
```

---

## Summary

| Task | Description | Key Action |
|------|-------------|------------|
| 1 | Scaffold packages/pypath | Create pyproject.toml for pypath-ewe |
| 2 | Move core source | `src/pypath/` → `packages/pypath/src/pypath/` |
| 3 | Move core tests | 68 test files + data + scripts → `packages/pypath/tests/` |
| 4 | Move example data | `example_model_data/` → `packages/pypath/example_model_data/` |
| 5 | Scaffold packages/pypath-shiny | Create pyproject.toml for pypath-shiny |
| 6 | Move app source | `app/` → `packages/pypath-shiny/src/pypath_shiny/` |
| 7 | Migrate imports | Replace all try/except fallbacks with `pypath_shiny.*` |
| 8 | Move app tests | 7 test files → `packages/pypath-shiny/tests/` |
| 9 | Update root config | Remove root pyproject.toml, update README |
| 10 | Update CI | Fix workflows for monorepo layout |
| 11 | MkDocs setup | Create API documentation site |
| 12 | Migrate docs | Move relevant docs into MkDocs |
| 13 | Final verification | Run all tests, verify imports, cleanup |
