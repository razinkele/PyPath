# Ecotracer (Contaminant Tracking) Design Spec

**Goal:** Track contaminant concentrations through the food web alongside Ecosim biomass dynamics.

**Approach:** Lightweight coupling via keyword argument to `rsim_run()`, following the mediation pattern. Tracer code lives in a separate module (`core/ecotracer.py`). Single contaminant per run.

---

## 1. Data Structures

### EcotracerParams

Dataclass holding per-group tracer parameters. All arrays are 0-based, length `n_groups = NUM_LIVING + NUM_DEAD` (matches `params.NUM_GROUPS` in Ecosim internals). No fleets, no padding column.

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
dietary_intake_i = cassim_i * sum_j(Q_ij * C_j) / B_i   (if B_i > 1e-10)
                 = 0                                      (if B_i <= 1e-10)
```

- `Q_ij` = consumption of prey j by predator i (from Ecosim consumption matrix)
- `C_j` = current concentration in prey j
- `B_i` = current biomass of predator i
- When `B_i` is near zero (crashed group), dietary intake is zero to avoid division by zero

### Detritus groups

```
dC_det/dt = (weighted avg of contributor concentrations) - cdecay_det * C_det
```

Detritus receives contaminant from dead matter proportional to detritus fate fractions and contributor concentrations. `detritus_fate` has shape `(n_living, n_detritus)` — fraction of each living group's non-predation mortality going to each detritus pool. Sourced from the Ecopath model's detritus fate matrix. When `None`, detritus concentration only decays (no new contaminant input from dead matter).

### Functions

```python
def ecotracer_deriv(
    conc: np.ndarray,        # (n_groups,) current concentrations
    biomass: np.ndarray,     # (n_groups,) current biomass
    Q_matrix: np.ndarray,    # (n_groups, n_groups) consumption matrix Q[prey, pred], 0-based
    params: EcotracerParams,
    detritus_fate: np.ndarray | None = None,  # (n_living, n_detritus) fate fractions
    n_living: int = 0,       # number of living groups (rest are detritus)
) -> np.ndarray:
    """Compute dC/dt for all groups. Returns (n_groups,) array.
    Guards against B_i <= 1e-10 (sets dietary_intake to 0)."""

def ecotracer_step(
    conc: np.ndarray,
    biomass: np.ndarray,
    Q_matrix: np.ndarray,
    params: EcotracerParams,
    dt: float,
    detritus_fate: np.ndarray | None = None,
    n_living: int = 0,
) -> np.ndarray:
    """Analytic update for tracer concentration (unconditionally stable).

    For each group i:
      input_i = dietary_intake_i + cenv_i + cimmig_i
      loss_rate_i = cdecay_i + cmetab_i
      if loss_rate_i > 0:
          C_i(t+dt) = input_i/loss_rate_i + (C_i(t) - input_i/loss_rate_i) * exp(-loss_rate_i*dt)
      else:
          C_i(t+dt) = C_i(t) + input_i * dt

    This is exact for constant input/loss within the timestep and avoids
    Euler instability when (cdecay + cmetab) * dt > 1. Result clamped to >= 0.
    Returns updated (n_groups,) array.
    """
```

---

## 3. Integration with rsim_run

`rsim_run(scenario, ecotracer=ecotracer_params)` — keyword-only argument, default None.

When `ecotracer` is provided:

1. **Before loop**: Initialize `conc = ecotracer.czero.copy()`, allocate output array `out_Conc` of shape `(n_months+1, n_groups)`, store initial concentrations at t=0.

2. **Each monthly step** (after biomass integration):
   - Extract current biomass: `biomass = state[1:n_groups+1]` (strip 1-based padding col 0)
   - Reuse the consumption matrix `QQ_month` already computed for Qlink tracking via `_compute_Q_matrix(params_dict, state, forcing_dict)`
   - Slice to 0-based: `Q = QQ_month[1:n_groups+1, 1:n_groups+1]`
   - Call `ecotracer_step(conc, biomass, Q, ecotracer, dt=1/12, detritus_fate, n_living)`
   - Store `conc` in `out_Conc[month]`

3. **After loop**: Compute `annual_Conc` by averaging monthly values per year, using the same window as `annual_Biomass`: `annual_Conc[yr] = mean(out_Conc[yr*12+1 : (yr+1)*12+1])`. Attach `EcotracerResult` to `RsimOutput` as `.ecotracer` attribute.

**Return type**: `rsim_run` still returns `RsimOutput`. The `EcotracerResult` is accessed via `output.ecotracer` (None if not used). Adding `ecotracer: EcotracerResult | None = None` as the **last field** of `RsimOutput` (with default `None`) is backward-compatible since all existing code uses keyword access.

### Consumption matrix extraction

`deriv_vector()` computes `QQ` internally but does not return it. The monthly loop in `rsim_run()` already calls `_compute_Q_matrix(params_dict, state, forcing_dict)` after each biomass integration step for Qlink tracking — this produces `QQ_month` with shape `(NUM_GROUPS+1, NUM_GROUPS+1)` using 1-based indexing. The ecotracer receives the 0-based slice `QQ_month[1:n_groups+1, 1:n_groups+1]`.

### Scenario-level defaults

`EcotracerScenario` has global defaults (`Czero`, `Cinflow`, `Coutflow`, `Cdecay`). In `read_ecotracer()`, these serve as fallback values when `EcotracerScenarioGroup` doesn't have a per-group override:
- `Czero` → default for `czero` array
- `Cinflow` → default for `cimmig` array (immigration inflow)
- `Cdecay` → default for `cdecay` array
- `Coutflow` → not mapped to EcotracerParams (outflow is modeled via metabolism/decay)
- `ConForcingShapeID` → not supported in this implementation (time-varying concentration forcing is out of scope)

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
- ecotracer_step() analytic update correctness (matches exact solution for constant input)
- ecotracer_step() stable for high decay rates (cdecay*dt > 1)
- Concentration clamped to >= 0
- B_i = 0 → dietary intake is zero (no division by zero)

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
