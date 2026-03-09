# Documentation, Deployment & Versioning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automate versioning with python-semantic-release, add fully automated PyPI publishing, populate API docs with GitHub Pages deployment, and update deployment routines.

**Architecture:** Per-package semantic-release configs drive version bumping, CHANGELOG generation, and PyPI publishing. A separate docs workflow builds mkdocs and deploys to GitHub Pages. The Shiny app gets a nav link to the docs site.

**Tech Stack:** python-semantic-release, python-build, mkdocs-material, mkdocstrings, GitHub Actions (OIDC trusted publisher), peaceiris/actions-gh-pages

---

### Task 1: Per-Package CHANGELOGs

**Files:**
- Create: `packages/pypath/CHANGELOG.md`
- Create: `packages/pypath-shiny/CHANGELOG.md`
- Modify: `CHANGELOG.md` (root)

**Step 1: Create core package CHANGELOG**

Create `packages/pypath/CHANGELOG.md`:

```markdown
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
```

**Step 2: Create shiny package CHANGELOG**

Create `packages/pypath-shiny/CHANGELOG.md`:

```markdown
# Changelog — pypath-shiny

All notable changes to the **pypath-shiny** web frontend.

Format: [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

<!--next-version-placeholder-->

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
```

**Step 3: Update root CHANGELOG to be a pointer**

Replace `CHANGELOG.md` contents with:

```markdown
# Changelog

This monorepo has per-package changelogs:

- **pypath-ewe** (core): [`packages/pypath/CHANGELOG.md`](packages/pypath/CHANGELOG.md)
- **pypath-shiny** (frontend): [`packages/pypath-shiny/CHANGELOG.md`](packages/pypath-shiny/CHANGELOG.md)
```

**Step 4: Commit**

```bash
git add packages/pypath/CHANGELOG.md packages/pypath-shiny/CHANGELOG.md CHANGELOG.md
git commit -m "docs: add per-package CHANGELOGs, root becomes pointer"
```

---

### Task 2: Semantic Release Config for pypath-ewe

**Files:**
- Modify: `packages/pypath/pyproject.toml`

**Step 1: Add semantic-release config to pypath-ewe pyproject.toml**

Append after the existing `[tool.mypy]` section:

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
version_variables = ["src/pypath/__init__.py:__version__"]
branch = "main"
commit_message = "chore(release): pypath-ewe v{version}"
tag_format = "pypath-ewe-v{version}"
changelog_file = "CHANGELOG.md"
build_command = "python -m build"

[tool.semantic_release.commit_parser_options]
allowed_tags = ["feat", "fix", "perf", "refactor", "style", "docs", "test", "ci", "chore", "build"]
minor_tags = ["feat"]
patch_tags = ["fix", "perf"]

[tool.semantic_release.changelog]
template_dir = "templates"
changelog_file = "CHANGELOG.md"

[tool.semantic_release.changelog.environment]
block_start_string = "{%"
block_end_string = "%}"
variable_start_string = "{{"
variable_end_string = "}}"
```

**Step 2: Add `build` to dev dependencies**

In `[project.optional-dependencies]`, update the `dev` list to include `"build>=1.0"` and `"python-semantic-release>=9.0"`:

```toml
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
    "build>=1.0",
    "python-semantic-release>=9.0",
]
```

**Step 3: Commit**

```bash
git add packages/pypath/pyproject.toml
git commit -m "build: add python-semantic-release config for pypath-ewe"
```

---

### Task 3: Semantic Release Config for pypath-shiny

**Files:**
- Modify: `packages/pypath-shiny/pyproject.toml`

**Step 1: Add semantic-release config to pypath-shiny pyproject.toml**

Append after the existing `[tool.ruff.lint]` section:

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
version_variables = ["src/pypath_shiny/__init__.py:__version__"]
branch = "main"
commit_message = "chore(release): pypath-shiny v{version}"
tag_format = "pypath-shiny-v{version}"
changelog_file = "CHANGELOG.md"
build_command = "python -m build"

[tool.semantic_release.commit_parser_options]
allowed_tags = ["feat", "fix", "perf", "refactor", "style", "docs", "test", "ci", "chore", "build"]
minor_tags = ["feat"]
patch_tags = ["fix", "perf"]
```

**Step 2: Add `build` and `python-semantic-release` to dev dependencies**

```toml
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "build>=1.0",
    "python-semantic-release>=9.0",
]
```

**Step 3: Commit**

```bash
git add packages/pypath-shiny/pyproject.toml
git commit -m "build: add python-semantic-release config for pypath-shiny"
```

---

### Task 4: PyPI Release Workflow

**Files:**
- Create: `.github/workflows/release.yml`

**Step 1: Create the release workflow**

Create `.github/workflows/release.yml`:

```yaml
name: Release & Publish

on:
  push:
    branches: [main]

permissions:
  contents: write
  id-token: write  # OIDC for PyPI trusted publisher

concurrency:
  group: release
  cancel-in-progress: false

jobs:
  release-core:
    name: Release pypath-ewe
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: packages/pypath
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install tools
        run: pip install python-semantic-release build

      - name: Semantic Release (pypath-ewe)
        id: release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          semantic-release version --no-push --no-commit 2>&1 | tee /tmp/sr-output.txt
          if grep -q "No release will be made" /tmp/sr-output.txt; then
            echo "released=false" >> $GITHUB_OUTPUT
          else
            echo "released=true" >> $GITHUB_OUTPUT
            echo "version=$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')" >> $GITHUB_OUTPUT
          fi

      - name: Build package
        if: steps.release.outputs.released == 'true'
        run: python -m build

      - name: Publish to PyPI
        if: steps.release.outputs.released == 'true'
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: packages/pypath/dist/

      - name: Commit version bump & tag
        if: steps.release.outputs.released == 'true'
        run: |
          cd ../..
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add packages/pypath/pyproject.toml packages/pypath/src/pypath/__init__.py packages/pypath/CHANGELOG.md
          git commit -m "chore(release): pypath-ewe v${{ steps.release.outputs.version }}"
          git tag "pypath-ewe-v${{ steps.release.outputs.version }}"
          git push origin main --tags

    outputs:
      released: ${{ steps.release.outputs.released }}
      version: ${{ steps.release.outputs.version }}

  release-shiny:
    name: Release pypath-shiny
    needs: release-core
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: packages/pypath-shiny
    steps:
      - name: Checkout (latest, after core release commit)
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          ref: main
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install tools
        run: pip install python-semantic-release build

      - name: Semantic Release (pypath-shiny)
        id: release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          semantic-release version --no-push --no-commit 2>&1 | tee /tmp/sr-output.txt
          if grep -q "No release will be made" /tmp/sr-output.txt; then
            echo "released=false" >> $GITHUB_OUTPUT
          else
            echo "released=true" >> $GITHUB_OUTPUT
            echo "version=$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')" >> $GITHUB_OUTPUT
          fi

      - name: Build package
        if: steps.release.outputs.released == 'true'
        run: python -m build

      - name: Publish to PyPI
        if: steps.release.outputs.released == 'true'
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: packages/pypath-shiny/dist/

      - name: Commit version bump & tag
        if: steps.release.outputs.released == 'true'
        run: |
          cd ../..
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add packages/pypath-shiny/pyproject.toml packages/pypath-shiny/src/pypath_shiny/__init__.py packages/pypath-shiny/CHANGELOG.md
          git commit -m "chore(release): pypath-shiny v${{ steps.release.outputs.version }}"
          git tag "pypath-shiny-v${{ steps.release.outputs.version }}"
          git push origin main --tags

  create-github-release:
    name: Create GitHub Release
    needs: [release-core, release-shiny]
    if: needs.release-core.outputs.released == 'true'
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          ref: main

      - name: Create Release
        uses: softprops/action-gh-release@v2
        with:
          tag_name: "pypath-ewe-v${{ needs.release-core.outputs.version }}"
          name: "PyPath v${{ needs.release-core.outputs.version }}"
          generate_release_notes: true
```

**Step 2: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add automated PyPI release workflow with OIDC trusted publisher"
```

---

### Task 5: GitHub Pages Docs Workflow

**Files:**
- Create: `.github/workflows/docs.yml`

**Step 1: Create the docs workflow**

Create `.github/workflows/docs.yml`:

```yaml
name: Deploy Documentation

on:
  push:
    branches: [main]
    paths:
      - "packages/pypath/docs/**"
      - "packages/pypath/src/**"
      - ".github/workflows/docs.yml"
  workflow_dispatch:

permissions:
  contents: write

jobs:
  deploy-docs:
    name: Build & Deploy MkDocs
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install -e "packages/pypath[docs]"
          pip install mkdocs-material mkdocstrings[python]

      - name: Build docs
        working-directory: packages/pypath/docs
        run: mkdocs build --strict

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: packages/pypath/docs/site
          publish_branch: gh-pages
```

**Step 2: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "ci: add GitHub Pages docs deployment workflow"
```

---

### Task 6: Populate API Documentation Content

**Files:**
- Modify: `packages/pypath/docs/docs/index.md`
- Modify: `packages/pypath/docs/docs/getting-started.md`
- Modify: `packages/pypath/docs/docs/api/core.md`
- Modify: `packages/pypath/docs/docs/api/io.md`
- Modify: `packages/pypath/docs/docs/api/spatial.md`
- Modify: `packages/pypath/docs/docs/api/ibm.md`
- Modify: `packages/pypath/docs/docs/api/analysis.md`
- Modify: `packages/pypath/docs/docs/examples/basic-model.md`
- Modify: `packages/pypath/docs/docs/examples/spatial.md`
- Modify: `packages/pypath/docs/mkdocs.yml`

**Step 1: Update index.md with richer content**

Replace `packages/pypath/docs/docs/index.md` with:

```markdown
# PyPath EwE

**Python implementation of Ecopath with Ecosim (EwE) for food web modeling.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/pypath-ewe)](https://pypi.org/project/pypath-ewe/)

PyPath extends the R package [Rpath](https://github.com/NOAA-EDAB/Rpath) with advanced features while maintaining full core compatibility.

## Installation

```bash
pip install pypath-ewe
```

### Optional Extras

```bash
pip install pypath-ewe[spatial]      # Ecospace spatial modeling
pip install pypath-ewe[interactive]  # Plotly interactive plots
pip install pypath-ewe[biodata]      # Species data from WoRMS/OBIS/FishBase
pip install pypath-ewe[numba]        # JIT-compiled ODE solver (~40% faster)
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
# Set biomass, PB, QB, diet matrix...

# Balance the model
model = rpath(params)

# Run 50-year dynamic simulation
scenario = rsim_scenario(model, params, years=range(1, 51))
output = rsim_run(scenario)
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Ecopath** | Mass-balance food web modeling with multi-stanza support |
| **Ecosim** | Dynamic simulation using foraging arena theory |
| **Ecospace** | Spatially-explicit modeling with hexagonal grids |
| **IBM** | Individual-based model coupling (bioenergetics, predation) |
| **State-Variable Forcing** | Data assimilation and prescribed scenarios |
| **Diet Rewiring** | Adaptive foraging and prey switching |
| **Optimization** | Bayesian parameter calibration |
| **Data Import** | EwE databases, EcoBase, WoRMS/OBIS/FishBase |

## Packages

This library is part of the PyPath monorepo:

- **pypath-ewe** — Core algorithms (this package)
- **[pypath-shiny](https://github.com/razinkele/PyPath)** — Interactive web frontend

## Web Frontend

Install the Shiny dashboard for a graphical interface:

```bash
pip install pypath-shiny
pypath-shiny  # Launches at http://localhost:8000
```
```

**Step 2: Update getting-started.md with a real walkthrough**

Replace `packages/pypath/docs/docs/getting-started.md` with:

```markdown
# Getting Started

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

```bash
pip install pypath-ewe
```

For spatial modeling and interactive plots:

```bash
pip install pypath-ewe[spatial,interactive]
```

## Your First Model

### 1. Create Parameters

```python
from pypath import create_rpath_params

params = create_rpath_params(
    groups=["Phytoplankton", "Zooplankton", "Small Fish", "Detritus", "Fleet"],
    types=[1, 0, 0, 2, 3],  # 1=producer, 0=consumer, 2=detritus, 3=fleet
)

# Set biomass (t/km2)
params.model.loc[0, "Biomass"] = 10.0   # Phytoplankton
params.model.loc[1, "Biomass"] = 5.0    # Zooplankton
params.model.loc[2, "Biomass"] = 2.0    # Small Fish
params.model.loc[3, "Biomass"] = 100.0  # Detritus

# Production/biomass ratios
params.model.loc[0, "PB"] = 200.0
params.model.loc[1, "PB"] = 50.0
params.model.loc[2, "PB"] = 1.0

# Consumption/biomass ratios (consumers only)
params.model.loc[1, "QB"] = 150.0
params.model.loc[2, "QB"] = 5.0

# Ecotrophic efficiency
params.model.loc[0, "EE"] = 0.8
params.model.loc[1, "EE"] = 0.9
params.model.loc[2, "EE"] = 0.5

# Diet matrix (who eats whom)
params.diet["Zooplankton"] = [1.0, 0.0, 0.0, 0.0, 0.0]
params.diet["Small Fish"]  = [0.0, 1.0, 0.0, 0.0, 0.0]
```

### 2. Balance the Model (Ecopath)

```python
from pypath import rpath

model = rpath(params)
print(model)
```

### 3. Run Dynamic Simulation (Ecosim)

```python
from pypath import rsim_scenario, rsim_run

scenario = rsim_scenario(model, params, years=range(1, 51))
output = rsim_run(scenario)

# Biomass trajectories: shape (months, groups)
print(output.biomass.shape)
```

### 4. Pre-Balance Diagnostics

```python
from pypath.analysis.prebalance import prebalance_diagnostics

diagnostics = prebalance_diagnostics(params)
```

## Loading Existing Models

### From EcoBase

```python
from pypath import search_ecobase_models, get_ecobase_model, ecobase_to_rpath

results = search_ecobase_models("Baltic Sea")
model_data = get_ecobase_model(model_id=123)
params = ecobase_to_rpath(model_data)
```

### From EwE Database (.eweaccdb)

```python
from pypath import read_ewemdb

params = read_ewemdb("path/to/model.eweaccdb")
```

## Development Setup

```bash
git clone https://github.com/razinkele/PyPath.git
cd PyPath
pip install -e "packages/pypath[all]"
pip install -e "packages/pypath-shiny[dev]"
pytest packages/pypath/tests -q -m "not integration and not slow"
```

## Next Steps

- [Basic Model Example](examples/basic-model.md) — Detailed walkthrough
- [Spatial Modeling](examples/spatial.md) — Ecospace setup
- [API Reference](api/core.md) — Full API docs
```

**Step 3: Add ecosystem derivatives and optimization to core API doc**

Replace `packages/pypath/docs/docs/api/core.md` with:

```markdown
# Core API Reference

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

::: pypath.core.ecosim
    options:
      show_root_heading: true
      members_order: source

## ODE Derivatives

::: pypath.core.ecosim_deriv
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
```

**Step 4: Update io.md to include marine_data**

Replace `packages/pypath/docs/docs/api/io.md` with:

```markdown
# I/O API Reference

## EcoBase

::: pypath.io.ecobase
    options:
      show_root_heading: true

## EwE Database (.eweaccdb)

::: pypath.io.ewemdb
    options:
      show_root_heading: true

## Biological Data (WoRMS/OBIS/FishBase)

::: pypath.io.biodata
    options:
      show_root_heading: true

## Marine Environmental Data (EMODnet)

::: pypath.io.marine_data
    options:
      show_root_heading: true

## Utilities

::: pypath.io.utils
    options:
      show_root_heading: true
```

**Step 5: Update mkdocs.yml nav to include all sections**

Replace `packages/pypath/docs/mkdocs.yml` with:

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
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.tabs.link
  palette:
    - scheme: default
      primary: teal
      accent: teal
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: teal
      accent: teal
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

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
            docstring_style: numpy
            show_signature_annotations: true
            separate_signature: true

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - API Reference:
      - Core (Ecopath/Ecosim): api/core.md
      - I/O: api/io.md
      - Spatial (Ecospace): api/spatial.md
      - IBM: api/ibm.md
      - Analysis: api/analysis.md
  - Examples:
      - Basic Model: examples/basic-model.md
      - Spatial Modeling: examples/spatial.md
```

**Step 6: Commit**

```bash
git add packages/pypath/docs/
git commit -m "docs: populate API documentation with mkdocstrings directives and content"
```

---

### Task 7: Add Documentation Link to Shiny App

**Files:**
- Modify: `packages/pypath-shiny/src/pypath_shiny/app.py:186-188`

**Step 1: Add docs nav link after the About panel**

In `app.py`, find this block (around line 186-188):

```python
        ui.nav_panel(
            _icon_label("bi-info-circle", "About"), about.about_ui(), value="About"
        ),
```

Insert immediately after it:

```python
        ui.nav_control(
            ui.tags.a(
                ui.TagList(
                    ui.tags.i(
                        class_="bi bi-book",
                        style="margin-right: 8px;",
                    ),
                    "API Documentation",
                ),
                href="https://razinkele.github.io/PyPath/",
                target="_blank",
                class_="btn btn-link text-start w-100 p-2",
                style="text-decoration: none;",
            ),
        ),
```

**Step 2: Commit**

```bash
git add packages/pypath-shiny/src/pypath_shiny/app.py
git commit -m "feat(shiny): add API Documentation link to sidebar navigation"
```

---

### Task 8: Update Deploy Workflow for New Tag Format

**Files:**
- Modify: `.github/workflows/deploy.yml:15-16`

**Step 1: Update tag pattern in deploy.yml**

Find the `on.push` section:

```yaml
  push:
    branches: [ main ]
    tags: ['v*']
```

Replace with:

```yaml
  push:
    branches: [ main ]
    tags: ['v*', 'pypath-*-v*']
```

**Step 2: Update production detection regex**

Find (line ~64):

```bash
          if [ -n "${GITHUB_REF:-}" ] && echo "${GITHUB_REF:-}" | grep -qE '^refs/tags/v'; then
```

Replace with:

```bash
          if [ -n "${GITHUB_REF:-}" ] && echo "${GITHUB_REF:-}" | grep -qE '^refs/tags/(v|pypath-.*-v)'; then
```

**Step 3: Add version logging after install**

Find the line `echo "Packages installed successfully."` (around line 156). Insert before it:

```bash
          # Log deployed version
          ssh "$REMOTE" "'${TARGET}/venv/bin/python' -c 'import pypath; print(f\"Deployed pypath-ewe v{pypath.__version__}\")'"
          ssh "$REMOTE" "'${TARGET}/venv/bin/python' -c 'import pypath_shiny; print(f\"Deployed pypath-shiny v{pypath_shiny.__version__}\")'"
```

**Step 4: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: update deploy workflow for semantic-release tag format and version logging"
```

---

### Task 9: Update prepare_package.ps1 Version Reading

**Files:**
- Modify: `deploy/prepare_package.ps1:135-147`

**Step 1: Replace date-based version with pyproject.toml version**

Find the version file creation block (around line 135-147):

```powershell
# Create version file
$Version = (Get-Date -Format "yyyy.MM.dd")
$GitHash = ""
try {
    $GitHash = (git -C $ProjectRoot rev-parse --short HEAD 2>$null)
} catch {}
```

Replace with:

```powershell
# Read version from pyproject.toml
$PyprojectPath = Join-Path $ProjectRoot "packages\pypath\pyproject.toml"
$Version = "unknown"
if (Test-Path $PyprojectPath) {
    $match = Select-String -Path $PyprojectPath -Pattern '^version\s*=\s*"(.+)"' | Select-Object -First 1
    if ($match) { $Version = $match.Matches.Groups[1].Value }
}
$GitHash = ""
try {
    $GitHash = (git -C $ProjectRoot rev-parse --short HEAD 2>$null)
} catch {}
```

**Step 2: Commit**

```bash
git add deploy/prepare_package.ps1
git commit -m "build: read version from pyproject.toml in prepare_package.ps1"
```

---

### Task 10: Create Initial Version Tags

**Files:** None (git operations only)

**Step 1: Create tags for current v0.3.0 release**

```bash
git tag pypath-ewe-v0.3.0
git tag pypath-shiny-v0.3.0
```

**Step 2: Verify tags**

Run: `git tag -l "pypath-*"`
Expected:
```
pypath-ewe-v0.3.0
pypath-shiny-v0.3.0
```

**Step 3: Push tags (after user confirmation)**

```bash
git push origin pypath-ewe-v0.3.0 pypath-shiny-v0.3.0
```

---

### Task 11: Verify Everything Works

**Step 1: Run ruff on modified files**

```bash
conda run -n shiny ruff check packages/pypath/pyproject.toml packages/pypath-shiny/pyproject.toml packages/pypath-shiny/src/pypath_shiny/app.py
```
Expected: No errors

**Step 2: Run core tests**

```bash
conda run -n shiny python -m pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts --tb=short
```
Expected: 770+ passed

**Step 3: Run shiny tests**

```bash
conda run -n shiny python -m pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui --tb=short
```
Expected: 135+ passed

**Step 4: Test mkdocs build locally**

```bash
conda run -n shiny pip install mkdocs-material "mkdocstrings[python]"
cd packages/pypath/docs && conda run -n shiny mkdocs build --strict
```
Expected: Site built successfully in `site/` directory

**Step 5: Verify semantic-release config**

```bash
cd packages/pypath && conda run -n shiny semantic-release version --noop --no-push --no-commit
cd packages/pypath-shiny && conda run -n shiny semantic-release version --noop --no-push --no-commit
```
Expected: Shows "No release will be made" (no new commits since tag)
