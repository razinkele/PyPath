# PyPath – Copilot Instructions

PyPath is a Python monorepo implementing the Ecopath with Ecosim (EwE) food web modeling framework. It contains two independently publishable packages under `packages/`.

## Package Layout

| Package dir | PyPI name | Import as | Role |
|---|---|---|---|
| `packages/pypath/` | `pypath-ewe` | `import pypath` | Core algorithms (library) |
| `packages/pypath-shiny/` | `pypath-shiny` | `import pypath_shiny` | Shiny web frontend |

**Dependency direction is one-way:** `pypath-shiny` depends on `pypath-ewe`, never the reverse. Never add `pypath_shiny` imports to `packages/pypath/`.

## Development Setup

```bash
pip install -e "packages/pypath[dev]"
pip install -e "packages/pypath-shiny[dev]"
```

Optional extras for core package: `interactive`, `spatial`, `spatial-data`, `biodata`, `numba`, `ewedb`.

## Commands

### Tests
```bash
# Core – fast (exclude slow/integration)
pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts

# Core – full suite
pytest packages/pypath/tests/ -q --ignore=packages/pypath/tests/scripts

# Single test file
pytest packages/pypath/tests/test_ecosim.py -v

# Single test by name
pytest packages/pypath/tests/test_ecopath.py -v -k "test_basic_creation"

# Shiny tests (no Playwright)
pytest packages/pypath-shiny/tests/ -q --ignore=packages/pypath-shiny/tests/ui
```

### Lint & Format
```bash
ruff check packages/          # lint
ruff format --check packages/ # format check
ruff format packages/         # auto-format
black packages/               # alternative formatter (line-length 88)
mypy packages/pypath/src --config-file packages/pypath/pyproject.toml
```

### Run Shiny App
```bash
pypath-shiny         # via entry point after install
python run_app.py    # via repo root script
```

## Architecture: Simulation Pipeline

The core modeling pipeline is `Ecopath → Ecosim → Ecospace`:

1. **`rpath(params)`** (`core/ecopath.py`) – solves the static mass-balance. Returns an `Rpath` object with balanced biomass/mortality parameters.
2. **`rsim_scenario(model, params)`** (`core/ecosim.py`) – builds a dynamic Ecosim scenario from the balanced model. Returns a scenario object used for time-step integration.
3. **`rsim_run(scenario)`** – runs the ODE integration, calling `deriv_vector()` (`core/ecosim_deriv.py`) each time step. Timestep is `DELTA_T = 1/12` year (monthly); `STEPS_PER_YEAR = 12`.
4. **`spatial/integration.py`** – wraps the Ecosim scenario for spatially-explicit Ecospace runs across grid patches.

Key supporting modules:
- `core/params.py` – `create_rpath_params()` / `RpathParams` (DataFrame-based parameter container)
- `core/stanzas.py` – multi-stanza (age-structured) group handling
- `core/forcing.py` – environmental forcing time series wired into Ecosim
- `ibm/` – Individual-Based Model that overrides Ecosim derivatives for tagged species

### IBM Module

The IBM module lets a species bypass Ecosim's population equations with explicit super-individual tracking:

- All concrete IBM implementations must subclass `IBMGroup` (`ibm/base.py`) and implement its abstract `step()` method.
- `ibm/integration.py` bridges IBM ↔ Ecosim: it reads from the Ecosim consumption matrix (`QQ`), calls the IBM step, and writes results back into the derivative vector via `apply_ibm_to_derivative()`.
- `ibm/smelt.py` (`SmeltIBM`) is the reference implementation for Baltic smelt.
- `ibm/development.py` holds early-life-stage dataclasses: `EggParams`, `YolkSacParams`, `LarvalParams`, `OxygenParams`, `ZoneParams`.
- Use `TYPE_CHECKING` guards when importing IBM types in `ecosim.py` / `ecosim_deriv.py` to avoid circular imports:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from pypath.ibm.base import IBMGroup, IBMStepResult
  ```

### `RpathParams` group types

In `params.model`, the `Type` column encodes group kind:

| Value | Meaning |
|---|---|
| 0 | Consumer (living) |
| 1 | Primary producer |
| 2 | Detritus |
| 3 | Fleet |

## Key Conventions

### Imports
```python
# Core library
from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params, RpathParams
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.spatial import EcospaceParams, create_1d_grid

# Shiny pages — cross-package imports are absolute; within pages/ use relative
from pypath_shiny.pages import ecopath, ecosim
from .utils import format_dataframe_for_display   # within pages/
```

All core modules use `from __future__ import annotations` at the top.

### Logging
```python
logger = logging.getLogger(__name__)
logger.debug("message")   # use debug/info/warning, never print()
```
Never use bare `except Exception: pass`.

Set `PYPATH_SILENCE_DEBUG=1` (env var) to suppress all `pypath` debug-level output at runtime.

### Docstrings
Use NumPy-style docstrings for all public functions and classes:
```python
def my_func(x: np.ndarray) -> float:
    """Short one-line summary.

    Parameters
    ----------
    x : np.ndarray
        Description of x.

    Returns
    -------
    float
        Description of return value.
    """
```

### Data structures
Most parameter containers and result objects are `@dataclass`. `RpathParams` holds pandas DataFrames (`model`, `diet`, `pedigree`, `remarks`) plus a `RpathStanzaParams` for multi-stanza groups.

### shinyswatch (theme)
`shinyswatch` is optional — the app has a graceful fallback when it is not installed. Only install it when developing/testing theme-related UI. Theme-related changes belong in `tests/test_shinyswatch_integration.py`.

### Test data paths
```python
DATA_DIR = Path(__file__).parent / "data"   # always relative to test file
```
Never hardcode paths relative to the repo root.

### Test markers
Decorate tests appropriately:
```python
@pytest.mark.integration   # requires network / real API calls
@pytest.mark.slow          # long-running
@pytest.mark.worms         # uses WoRMS API
@pytest.mark.obis          # uses OBIS API
@pytest.mark.fishbase      # uses FishBase API
```

### Test placement
- Tests for `pypath.*` → `packages/pypath/tests/`
- Tests that import `pypath_shiny.*` → `packages/pypath-shiny/tests/`
- `scripts/` at repo root are standalone analysis scripts, not part of either package

### Fixtures (conftest.py)
Shared fixtures in `packages/pypath/tests/conftest.py` build a balanced 5-group model (`spatial_scenario`, `single_patch_ecospace`, `simple_ecospace`). Reuse these in spatial and Ecosim tests rather than rebuilding models from scratch.

### Releases (semantic-release)
Commit messages must follow Conventional Commits: `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`, `build:`. `feat` bumps minor, `fix`/`perf` bump patch. Tags: `pypath-ewe-v{version}` and `pypath-shiny-v{version}`.

## CI Workflows

| File | Purpose |
|---|---|
| `ci.yml` | Lint + typecheck + unit tests (Ubuntu + Windows, Python 3.10–3.12) |
| `ci-shiny-smoke.yml` | Shiny app smoke test |
| `ci-auto-fix.yml` | Auto-format PRs |
| `release.yml` | Publishes to PyPI on tag push |
| `docs.yml` | Builds MkDocs and deploys to GitHub Pages |
