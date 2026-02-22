# PyPath - Ecopath with Ecosim in Python

## Project Overview

PyPath is a Python implementation of the Ecopath with Ecosim (EwE) framework for aquatic food web modeling. It is organized as a monorepo with two independently publishable packages.

## Repository Structure

```
packages/
  pypath/                          # Core algorithms (PyPI: pypath-ewe)
    src/pypath/
      core/                        # Ecopath, Ecosim, Ecospace engines
        ecopath.py                 # rpath() - mass-balance solver
        ecosim.py                  # rsim_run(), rsim_scenario()
        ecosim_deriv.py            # deriv_vector() - ODE derivatives
        params.py                  # create_rpath_params(), RpathParams
        stanzas.py                 # Multi-stanza age groups
        forcing.py                 # Environmental forcing
        optimization.py            # Parameter optimization
        plotting.py                # Food web visualization
      io/                          # Data import/export
        ewemdb.py                  # EwE model database reader
        biodata.py                 # OBIS/WoRMS/FishBase integration
        ecobase.py                 # EcoBase import
      spatial/                     # Ecospace spatial modeling
        integration.py             # Spatial simulation runner
        dispersal.py               # Species dispersal
        connectivity.py            # Patch connectivity
        habitat.py                 # Habitat preferences
      analysis/                    # Pre-balance diagnostics
      ibm/                         # Individual-Based Model module
        base.py                    # SuperIndividual, IBMStepResult, IBMGroup ABC
        bioenergetics.py           # Wisconsin model (growth, metabolism, Q10)
        predation.py               # Size-structured predation mortality
        behavior.py                # Spatial movement + adaptive foraging
        reproduction.py            # Stochastic spawning + larval survival
        integration.py             # Derivative override + mass balance checker
        smelt.py                   # SmeltIBM concrete implementation for Baltic smelt
    tests/                         # 551 core tests + 144 IBM tests
    example_model_data/            # CSV example model files
    docs/                          # MkDocs API documentation
    pyproject.toml                 # pypath-ewe v0.3.0

  pypath-shiny/                    # Web frontend (PyPI: pypath-shiny)
    src/pypath_shiny/
      app.py                       # Shiny app entry point, main()
      config.py                    # App configuration, VALIDATION
      pages/                       # UI page modules
        ecopath.py, ecosim.py, ecospace.py, prebalance.py,
        validation.py, analysis.py, data_import.py, results.py,
        home.py, about.py, utils.py, ...
      static/                      # CSS, logos, icons
    tests/                         # 115 app tests
      ui/                          # Playwright UI tests (optional)
    pyproject.toml                 # pypath-shiny v0.3.0
```

## Package Details

| | pypath-ewe | pypath-shiny |
|---|---|---|
| **PyPI name** | `pypath-ewe` | `pypath-shiny` |
| **Import** | `import pypath` | `import pypath_shiny` |
| **Version** | 0.3.0 | 0.3.0 |
| **Dependencies** | numpy, pandas, scipy, matplotlib | pypath-ewe, shiny, shinyswatch |
| **Entry point** | Library | `pypath-shiny` CLI command |

Dependency direction: `pypath-shiny` depends on `pypath-ewe`, never the reverse.

## Development Setup

```bash
pip install -e "packages/pypath[dev]"
pip install -e "packages/pypath-shiny[dev]"
```

## Running Tests

```bash
# Core tests (fast, excludes slow/integration)
pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts

# Full core tests
pytest packages/pypath/tests/ -q --ignore=packages/pypath/tests/scripts

# Shiny app tests (exclude Playwright UI tests)
pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui

# Single test file
pytest packages/pypath/tests/test_ecosim.py -v
```

## Code Style

- Formatter: black (line-length 88)
- Linter: ruff (E, F, I, W rules; E501 ignored)
- Python: >=3.10
- Logging: use `logger = logging.getLogger(__name__)` with `logger.debug()`, never bare `except Exception: pass`

## Key Conventions

- Test data paths must use `Path(__file__).parent / "data" / ...` (never hardcoded relative to repo root)
- Shiny app imports use `from pypath_shiny.pages import ...` (never `from pages import ...`)
- Core library imports use `from pypath.core.ecopath import rpath` etc.
- Tests that import from `pypath_shiny.*` belong in `packages/pypath-shiny/tests/`, not core
- The `scripts/` directory at repo root contains standalone analysis scripts

## CI Workflows

- `.github/workflows/ci.yml` - Core package lint + tests
- `.github/workflows/ci-shiny-smoke.yml` - Shiny app smoke test
- `.github/workflows/ci-auto-fix.yml` - Auto-format fixes
