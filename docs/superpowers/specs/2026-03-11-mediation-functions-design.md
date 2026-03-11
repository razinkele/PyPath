# Phase 2: Mediation Functions — Design Spec

**Date:** 2026-03-11
**Status:** Approved
**Scope:** Add EwE mediation function support to PyPath — shape-based and parametric mediation for group, fleet, and landings interactions.

## 1. Data Structures

**New module:** `packages/pypath/src/pypath/core/mediation.py`

### MediationShape Dataclass

```python
@dataclass
class MediationShape:
    shape_id: int
    name: str
    x_points: np.ndarray   # Relative biomass values, e.g. [0.0, 0.25, 0.5, ..., 2.0]
    y_points: np.ndarray   # Corresponding multiplier values

    def evaluate(self, relative_biomass: float) -> float:
        """Linear interpolation between points, clamped to endpoint values."""
```

EwE 6 stores shapes as 9 fixed Y values (`YY1..YY9`) at evenly-spaced X points from 0 to 2.0. The `x_points` array generalizes this to arbitrary spacing/count.

### MediationLink Dataclass

```python
@dataclass
class MediationLink:
    shape_id: int           # Which MediationShape to use
    mediator_idx: int       # 0-based group index of the mediating species
    # Target identification — exactly one target type is specified:
    prey_idx: int | None = None           # Group mediation: prey in the link
    pred_idx: int | None = None           # Group mediation: predator in the link
    fleet_idx: int | None = None          # Fleet mediation: which fleet
    landing_group_idx: int | None = None  # Landings mediation: which group
    landing_fleet_idx: int | None = None  # Landings mediation: which fleet
    weight: float = 1.0                   # Weighting factor (AppliedWeight)
```

**Group mediation:** `prey_idx` and `pred_idx` are both set; others are None.
**Fleet mediation:** `fleet_idx` is set; prey/pred/landing are None. `mediator_idx` is the group whose biomass drives the effect.
**Landings mediation:** `landing_group_idx` and `landing_fleet_idx` are both set; others are None.

### MediationCollection Dataclass

```python
@dataclass
class MediationCollection:
    shapes: list[MediationShape]
    links: list[MediationLink]
```

**Filtered view properties:**

- `group_links` — links where `prey_idx is not None and pred_idx is not None`
- `fleet_links` — links where `fleet_idx is not None`
- `landing_links` — links where `landing_group_idx is not None`

**Precomputation methods:**

- `compute_group_multipliers(BB, Bbase, ActiveLink) -> np.ndarray` — Returns 2D `(n_groups+1, n_groups+1)` multiplier matrix. For each group link, evaluates the shape at `BB[mediator+1] / Bbase[mediator+1]` (converting 0-based mediator_idx to 1-based column), and sets the `[prey+1, pred+1]` entry. If multiple mediation links affect the same pred-prey pair, their multipliers are multiplied together. Unaffected entries are 1.0. The 2D shape works for both the dense and sparse consumption kernels.
- `compute_fleet_multipliers(BB, Bbase, n_fleets) -> np.ndarray` — Returns per-fleet multipliers (length `n_fleets`). Each fleet link evaluates shape at mediator relative biomass. Default 1.0.
- `compute_landing_multipliers(BB, Bbase, n_fleets, n_groups) -> np.ndarray` — Returns `(n_fleets, n_groups)` multipliers for landed proportions. Default 1.0.

### Parametric Convenience Factories

```python
def make_positive_shape(shape_id=0, name="positive", low=0.5, high=2.0, shape=1.0, n_points=9) -> MediationShape
def make_negative_shape(shape_id=0, name="negative", low=0.5, high=2.0, shape=1.0, n_points=9) -> MediationShape
def make_ushape(shape_id=0, name="u-shaped", low=0.5, high=2.0, shape=1.0, n_points=9) -> MediationShape
```

These generate X-Y point arrays equivalent to the parametric types in the existing `mediation_function()`:

- **Positive:** `y = low + (high - low) * x^shape / (1 + x^shape)` sampled at n_points
- **Negative:** `y = high - (high - low) * x^shape / (1 + x^shape)` sampled at n_points
- **U-shaped:** `y = high - (high - low) * |x-1|^shape / (1 + |x-1|^shape)` sampled at n_points

X points are evenly spaced from 0 to 2.0 (matching EwE 6 default range). The `shape` parameter controls the steepness of the response curve (default 1.0 matches `mediation_function()` default).

## 2. I/O Layer

### Database Reading

Extend `io/ewemdb.py` with `read_mediation(db_path) -> MediationCollection` — standalone function, same pattern as `read_timeseries()`.

**Source tables:**

| Table | Key Columns | Purpose |
|---|---|---|
| `EcosimShapeMediation` | ShapeID, Title, nPoints, YY1..YY9 | Shape definitions (9 Y values at evenly-spaced X from 0 to 2.0) |
| `EcosimScenarioshapeMedWeightsGroup` | ScenarioID, ShapeID, GroupID, PredID, PreyID, AppliedWeight | Group mediation assignments |
| `EcosimScenarioshapeMedWeightsFleet` | ScenarioID, ShapeID, GroupID, FleetID, AppliedWeight | Fleet mediation assignments |
| `EcosimScenarioshapeMedWeightsLandings` | ScenarioID, ShapeID, GroupID, FleetID, AppliedWeight | Landings mediation assignments |

In weight tables, `GroupID` is the **mediator** group (the third party whose biomass drives the effect). `PredID`/`PreyID` identify the target interaction. All group/fleet IDs in the database are 1-based; the reader converts to 0-based.

Missing tables → empty collection (graceful degradation). These 4 tables must be added to `io/_ewe_schema.py` with full column definitions before implementation.

### Schema Definitions

Add to `io/_ewe_schema.py`:

```python
"EcosimShapeMediation": OrderedDict([
    ("ShapeID", "INTEGER"),
    ("Title", "TEXT"),
    ("nPoints", "INTEGER"),
    ("YY1", "DOUBLE"), ("YY2", "DOUBLE"), ("YY3", "DOUBLE"),
    ("YY4", "DOUBLE"), ("YY5", "DOUBLE"), ("YY6", "DOUBLE"),
    ("YY7", "DOUBLE"), ("YY8", "DOUBLE"), ("YY9", "DOUBLE"),
]),
"EcosimScenarioshapeMedWeightsGroup": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("ShapeID", "INTEGER"),
    ("GroupID", "INTEGER"),
    ("PredID", "INTEGER"),
    ("PreyID", "INTEGER"),
    ("AppliedWeight", "DOUBLE"),
]),
"EcosimScenarioshapeMedWeightsFleet": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("ShapeID", "INTEGER"),
    ("GroupID", "INTEGER"),
    ("FleetID", "INTEGER"),
    ("AppliedWeight", "DOUBLE"),
]),
"EcosimScenarioshapeMedWeightsLandings": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("ShapeID", "INTEGER"),
    ("GroupID", "INTEGER"),
    ("FleetID", "INTEGER"),
    ("AppliedWeight", "DOUBLE"),
]),
```

### Export

Extend `io/_csv_bundle_writer.py` with `write_mediation(self, collection)` method on `CsvBundleWriter` — writes all 4 tables. `io/_access_writer.py` delegates to CSV writer (same pattern as `write_timeseries()`). `io/ewe_writer.py` gains a `mediation` parameter on `write_ewemdb()`.

### Usage Pattern

```python
rpath = read_ewemdb("model.eweaccdb")
mediation = read_mediation("model.eweaccdb")
scenario = rsim_scenario(rpath, params)
result = rsim_run(scenario, mediation=mediation)
```

## 3. Runtime Integration

### Threading Mediation Through the Simulation

Mediation is threaded through the existing `params` dict — no signature changes to `deriv_vector()`, `integrate_rk4()`, or `integrate_ab()`. In `rsim_run()`:

```python
if mediation is not None:
    params_dict["_mediation"] = mediation
```

Inside `deriv_vector()`, extract it:

```python
_mediation = params.get("_mediation", None)
```

This replaces the existing unused `_Mediation = params.get("Mediation", {})` at line 1017, which should be removed.

### Consumption Kernel (Group Mediation)

In `deriv_vector()`, before calling the consumption kernel:

1. If `_mediation` is provided, call `_mediation.compute_group_multipliers(BB, Bbase, ActiveLink)` → `med_multipliers` 2D array `(n_groups+1, n_groups+1)`
2. If no mediation, `med_multipliers = None`

Both consumption kernels (`_compute_consumption_python` and `_compute_consumption_sparse_python`) gain an optional `med_multipliers` parameter (default None). When provided, multiply into Q_calc:

```python
Q_calc = qbase * PDY * PYY_term * dd_term * vv_term
if med_multipliers is not None:
    Q_calc *= med_multipliers[prey, pred]
```

The 2D matrix shape works identically for both kernels (indexed by `[prey, pred]` in the dense kernel, `[link_prey[idx], link_pred[idx]]` in the sparse kernel).

When `med_multipliers is None`, no multiplication occurs (zero overhead for non-mediation models).

### Fleet Mediation

In `deriv_vector()`, in the fishing link loop (lines 1234-1246), where `effort_mult` is computed from `ForcedEffort[gear_idx]`:

1. Extract fleet mediation multipliers: `fleet_med = _mediation.compute_fleet_multipliers(BB, Bbase, n_fleets)` (computed once before the loop)
2. Multiply into the existing `effort_mult`: `effort_mult *= fleet_med[gear_idx]`

This scales the effective effort for each fleet based on mediator biomass, before it's applied to group-level fishing mortality (`FishMort[grp] += FishQ[i] * effort_mult`).

### Landings Mediation

The current `deriv_vector()` does not distinguish landings from total catch — it computes `FishMort[grp]` as total fishing mortality without a landing/discard split. Landings mediation modifies the proportion of catch that is landed vs discarded, which affects Ecopath-level accounting but not the Ecosim derivative directly.

**Implementation approach:** Landings mediation multipliers are computed and stored on the `MediationCollection` but applied **post-simulation** when computing landed catch from total catch (in output processing or export), not inside `deriv_vector()`. The `compute_landing_multipliers()` method is available for callers who need to split catch into landings and discards.

**Out of scope for deriv_vector modification.** The multipliers are computed, stored, and available for downstream use, but the Ecosim derivative is unaffected since it operates on total fishing mortality.

### rsim_run() Signature Change

```python
def rsim_run(
    scenario: RsimScenario,
    method: str = "RK4",
    years: Optional[range] = None,
    *,
    mediation: MediationCollection | None = None,
) -> RsimOutput:
```

The `*` makes `mediation` keyword-only, preventing breakage of existing positional callers. `rsim_run()` injects `mediation` into `params_dict["_mediation"]` before the integration loop. No changes to `integrate_rk4()` or `integrate_ab()` signatures.

### Existing mediation_function()

The standalone `mediation_function()` in `ecosim_deriv.py` (lines 716-767) is deprecated. Add a deprecation docstring pointing to `MediationShape.evaluate()`. Keep it for backward compatibility but it is no longer called by the engine. Also remove the dead `_Mediation = params.get("Mediation", {})` line at line 1017.

## 4. Testing Strategy

### Unit Tests (`test_mediation.py`)

- `MediationShape` construction and `evaluate()`:
  - Known X-Y points → exact Y at X points
  - Linear interpolation between points
  - Clamping at x < min and x > max
  - Edge case: single-point shape
- `MediationLink` construction: group, fleet, landing variants
- `MediationCollection`:
  - Filtered views: `group_links`, `fleet_links`, `landing_links`
  - `compute_group_multipliers()` with known shapes, verify values (2D matrix)
  - `compute_fleet_multipliers()` with known shapes
  - `compute_landing_multipliers()` with known shapes
  - Multiple mediation links on same simulation link → multiplied together
  - Empty collection → all multipliers are 1.0
- Parametric factories: `make_positive_shape`, `make_negative_shape`, `make_ushape`
  - Verify Y values match expected parametric formula at X points
  - Test with different `shape` exponents

### Integration Tests (`test_mediation_integration.py`)

- 3-group model (producer → consumer → predator):
  - Consumer mediates producer-predator link (positive shape): more consumer → more predation on producer
  - Run with and without mediation → biomass trajectories differ
  - Negative shape: more mediator → less predation → prey benefits
- Fleet mediation: mediator biomass scales fleet effort → verify catch changes
- Regression: `rsim_run()` without mediation → identical output to current behavior
- Mark: `@pytest.mark.slow`

### I/O Tests (`test_mediation_io.py`)

- Round-trip: create collection → write CSV bundle → read back → assert shapes and links equal
- Schema table definitions have correct columns and types
- Missing tables → empty collection
- Database reading: `@pytest.mark.integration`

## 5. Files Changed/Created

| File | Action |
|---|---|
| `src/pypath/core/mediation.py` | **Create** — MediationShape, MediationLink, MediationCollection, factories |
| `src/pypath/core/ecosim_deriv.py` | **Modify** — add `med_multipliers` to consumption kernels, remove dead `_Mediation` code |
| `src/pypath/core/ecosim.py` | **Modify** — add keyword-only `mediation` param to `rsim_run()`, inject into `params_dict` |
| `src/pypath/io/ewemdb.py` | **Modify** — add `read_mediation()` |
| `src/pypath/io/_ewe_schema.py` | **Modify** — add 4 mediation table definitions |
| `src/pypath/io/_csv_bundle_writer.py` | **Modify** — add `write_mediation()` method on `CsvBundleWriter` |
| `src/pypath/io/_access_writer.py` | **Modify** — add `write_mediation()` delegating to CsvBundleWriter |
| `src/pypath/io/ewe_writer.py` | **Modify** — add `mediation` param to `write_ewemdb()` |
| `src/pypath/core/__init__.py` | **Modify** — export mediation classes and factories |
| `src/pypath/io/__init__.py` | **Modify** — export `read_mediation` |
| `tests/test_mediation.py` | **Create** — unit tests |
| `tests/test_mediation_integration.py` | **Create** — integration tests |
| `tests/test_mediation_io.py` | **Create** — I/O tests |

### Public API Import Paths

```python
from pypath.core.mediation import (
    MediationShape, MediationLink, MediationCollection,
    make_positive_shape, make_negative_shape, make_ushape,
)
from pypath.io.ewemdb import read_mediation
```

## 6. Design Decisions

1. **New `core/mediation.py` module** rather than extending `forcing.py` — mediation is biomass-dependent link modification, structurally different from time-varying external forcing.
2. **Shape-based (EwE 6 compatible) + parametric convenience** — shapes are the canonical representation; parametric factories are syntactic sugar that produce equivalent point arrays. Factory functions accept a `shape` exponent parameter (default 1.0) matching `mediation_function()`.
3. **Precomputed 2D multiplier matrix** — `compute_group_multipliers()` returns a `(n_groups+1, n_groups+1)` matrix that works for both dense and sparse consumption kernels. Keeps kernel modifications minimal (one extra multiply per link).
4. **All three weight types (group, fleet, landings)** — full EwE 6 parity. Group and fleet mediation modify the Ecosim derivative directly. Landings mediation multipliers are computed but applied post-simulation (the derivative doesn't distinguish landings from total catch).
5. **Thread through params dict** — mediation is injected as `params_dict["_mediation"]` in `rsim_run()`, avoiding signature changes to `deriv_vector()`, `integrate_rk4()`, and `integrate_ab()`. This follows the existing pattern where params is a dict carrying all simulation state.
6. **Keyword-only `mediation` on `rsim_run()`** — uses `*` separator to prevent breaking existing positional callers.
7. **Standalone `read_mediation()`** — composable; users call it separately from `read_ewemdb()`, same pattern as `read_timeseries()`.
8. **X-Y points with linear interpolation** — lossless round-trip with EwE 6 databases, simple implementation, generalizable beyond the fixed 9-point EwE format.
9. **Multiple mediation links on same simulation link → multiplicative** — when two mediators affect the same pred-prey link, their multipliers are multiplied together.
10. **Deprecate existing `mediation_function()` in ecosim_deriv.py** — superseded by `MediationShape.evaluate()`, kept for backward compatibility. Remove dead `_Mediation = params.get("Mediation", {})` line.
