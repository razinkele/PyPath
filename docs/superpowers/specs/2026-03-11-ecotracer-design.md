# Ecotracer (Contaminant Tracking) Design Spec

**Goal:** Track contaminant concentrations through the food web alongside Ecosim biomass dynamics.

**Approach:** Lightweight coupling via keyword argument to `rsim_run()`, following the mediation pattern. Tracer code lives in a separate module (`core/ecotracer.py`). Single contaminant per run.

---

## 1. Data Structures

### EcotracerParams

Dataclass holding per-group tracer parameters. All arrays are 0-based, length = number of living + detritus groups (no fleets, no padding).

```python
@dataclass
class EcotracerParams:
    czero: np.ndarray      # (n_groups,) initial concentration
    cenv: np.ndarray       # (n_groups,) environmental input concentration
    cimmig: np.ndarray     # (n_groups,) immigration input concentration
    cdecay: np.ndarray     # (n_groups,) decay rate
    cassim: np.ndarray     # (n_groups,) assimilation proportion, 0-1
    cmetab: np.ndarray     # (n_groups,) metabolism loss rate
```

### EcotracerResult

Dataclass holding output time series.

```python
@dataclass
class EcotracerResult:
    out_Conc: np.ndarray       # (n_months+1, n_groups) monthly concentrations
    annual_Conc: np.ndarray    # (n_years, n_groups) annual averages
    group_names: list[str]     # group name labels
```

### Factory

```python
def create_ecotracer_params(n_groups: int) -> EcotracerParams:
    """Create with defaults: czero=0, cenv=0, cimmig=0, cdecay=0, cassim=1.0, cmetab=0."""
```

---

## 2. Tracer ODE

### Mass balance for living group i

```
dC_i/dt = dietary_intake_i + cenv_i + cimmig_i - (cdecay_i + cmetab_i) * C_i
```

Where:
```
dietary_intake_i = cassim_i * sum_j(Q_ij * C_j) / B_i
```

- `Q_ij` = consumption of prey j by predator i (from Ecosim consumption matrix)
- `C_j` = current concentration in prey j
- `B_i` = current biomass of predator i

### Detritus groups

```
dC_det/dt = (weighted avg of contributor concentrations) - cdecay_det * C_det
```

Detritus receives contaminant from dead matter proportional to detritus fate fractions and contributor concentrations.

### Functions

```python
def ecotracer_deriv(
    conc: np.ndarray,        # (n_groups,) current concentrations
    biomass: np.ndarray,     # (n_groups,) current biomass
    Q_matrix: np.ndarray,    # (n_groups, n_groups) consumption matrix Q[prey, pred]
    params: EcotracerParams,
    detritus_fate: np.ndarray | None = None,  # (n_groups,) detritus fate fractions
) -> np.ndarray:
    """Compute dC/dt for all groups. Returns (n_groups,) array."""

def ecotracer_step(
    conc: np.ndarray,
    biomass: np.ndarray,
    Q_matrix: np.ndarray,
    params: EcotracerParams,
    dt: float,
    detritus_fate: np.ndarray | None = None,
) -> np.ndarray:
    """Euler step: conc + dC/dt * dt, clamped to >= 0. Returns updated (n_groups,) array."""
```

---

## 3. Integration with rsim_run

`rsim_run(scenario, ecotracer=ecotracer_params)` — keyword-only argument, default None.

When `ecotracer` is provided:

1. **Before loop**: Initialize `conc = ecotracer.czero.copy()`, allocate output array `out_Conc` of shape `(n_months+1, n_groups)`, store initial concentrations at t=0.

2. **Each monthly step** (after biomass integration):
   - Extract current biomass from state (0-based, exclude padding col 0)
   - Extract consumption matrix Q from the existing `QQ` variable in the loop
   - Call `ecotracer_step(conc, biomass, Q, ecotracer, dt=1/12)`
   - Store `conc` in `out_Conc[month]`

3. **After loop**: Compute `annual_Conc` by averaging monthly values per year. Attach `EcotracerResult` to `RsimOutput` as `.ecotracer` attribute.

**Return type**: `rsim_run` still returns `RsimOutput`. The `EcotracerResult` is accessed via `output.ecotracer` (None if not used). This requires adding an optional `ecotracer: EcotracerResult | None = None` field to `RsimOutput`.

### Consumption matrix extraction

The Ecosim loop already computes `QQ[prey, pred]` (the consumption matrix) each timestep. This matrix is available inside the monthly loop and can be passed directly to `ecotracer_step()`. The Q matrix uses 1-based indexing internally; the ecotracer functions receive the 0-based slice `QQ[1:n+1, 1:n+1]`.

---

## 4. I/O Layer

### Schema tables (from real EwE 6 database)

```python
"EcotracerScenario": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("ScenarioName", "TEXT"),
    ("Description", "TEXT"),
    ("Author", "TEXT"),
    ("Contact", "TEXT"),
    ("LastSaved", "TEXT"),
    ("ConForcingShapeID", "INTEGER"),
    ("Czero", "DOUBLE"),
    ("Cinflow", "DOUBLE"),
    ("Coutflow", "DOUBLE"),
    ("Cdecay", "DOUBLE"),
    ("LastSavedVersion", "TEXT"),
])

"EcotracerScenarioGroup": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("EcopathGroupID", "INTEGER"),
    ("Czero", "DOUBLE"),
    ("Cimmig", "DOUBLE"),
    ("Cenv", "DOUBLE"),
    ("Cdecay", "DOUBLE"),
    ("CassimProp", "DOUBLE"),
    ("CmetabolismRate", "DOUBLE"),
])
```

### read_ecotracer

```python
def read_ecotracer(db_path: str, n_groups: int) -> EcotracerParams:
    """Read Ecotracer parameters from EwE database.

    Reads EcotracerScenario for global defaults and
    EcotracerScenarioGroup for per-group overrides.
    Maps EcopathGroupID (1-based) to 0-based arrays.
    Returns default params if tables are missing/empty.
    """
```

---

## 5. File Structure

### New files
| File | Purpose |
|------|---------|
| `core/ecotracer.py` | EcotracerParams, EcotracerResult, create_ecotracer_params, ecotracer_deriv, ecotracer_step |
| `tests/test_ecotracer.py` | Unit tests for dataclasses, deriv, step |
| `tests/test_ecotracer_io.py` | Schema + read_ecotracer mock tests |
| `tests/test_ecotracer_integration.py` | End-to-end with 3-group Ecosim model |

### Modified files
| File | Change |
|------|--------|
| `core/ecosim.py` | rsim_run() gains `ecotracer=None` kwarg; monthly loop calls ecotracer_step when present |
| `core/ecosim.py` | RsimOutput gains `ecotracer: EcotracerResult \| None = None` field |
| `core/__init__.py` | Export EcotracerParams, EcotracerResult, create_ecotracer_params |
| `io/_ewe_schema.py` | Add EcotracerScenario, EcotracerScenarioGroup tables |
| `io/ewemdb.py` | Add read_ecotracer() |
| `io/__init__.py` | Export read_ecotracer |

---

## 6. Testing Strategy

### Unit tests (`test_ecotracer.py`)
- EcotracerParams construction and defaults
- create_ecotracer_params() shapes and default values
- ecotracer_deriv() with known inputs:
  - Zero concentration → only cenv/cimmig inputs
  - Zero dietary intake → only decay/metabolism losses
  - Known Q matrix → verifiable dietary uptake
- ecotracer_step() Euler integration correctness
- Concentration clamped to >= 0

### I/O tests (`test_ecotracer_io.py`)
- Schema tables exist with correct columns
- read_ecotracer() with mocked database
- Missing tables return default params
- DB exception returns default params

### Integration tests (`test_ecotracer_integration.py`, @pytest.mark.slow)
- 3-group model (Producer/Consumer/Detritus) with Ecosim + Ecotracer
- Consumer eats contaminated Producer → concentration increases
- Zero cenv + positive decay → concentration decreases
- rsim_run(scenario, ecotracer=params) returns output with .ecotracer attribute
- Result shapes: (n_months+1, n_groups) and (n_years, n_groups)
