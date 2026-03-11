# Phase 1: Time Series & Calibration Pipeline — Design Spec

**Date:** 2026-03-11
**Status:** Approved
**Scope:** Add EwE time series support, observed data loading, and sum-of-squares fitting to PyPath

## 1. Data Structures

**New module:** `packages/pypath/src/pypath/core/timeseries.py`

### DatType Constants

```python
DATTYPE_REL_BIOMASS = 0
DATTYPE_ABS_BIOMASS = 1
DATTYPE_FISHING_MORTALITY = 2
DATTYPE_EFFORT = 3
DATTYPE_CATCH = 6
DATTYPE_FORCED_BIOMASS = -1
```

**Out of scope for Phase 1:** DatType 4 (total mortality Z) and DatType 5 (average weight). These can be added later without breaking changes.

### EweTimeSeries Dataclass

```python
@dataclass
class EweTimeSeries:
    series_id: int
    name: str
    dat_type: int              # DatType constant
    group_idx: int | None      # None for fleet-level series
    fleet_idx: int | None      # None for group-level series
    values: np.ndarray         # shape (n_timesteps,), NaN for missing
    weight: float = 1.0        # SS weighting factor
    dataset_id: int = 0
```

### EweTimeSeriesCollection

Container with filtered views:

- `observed_biomass` — series with `dat_type` in {0, 1}
- `observed_catch` — series with `dat_type == 6`
- `forced_biomass` — series with `dat_type == -1`
- `forced_effort` — series with `dat_type == 3`
- `to_observed_dict(n_timesteps: int)` — returns `{group_idx: np.array}` for backward compatibility with existing `EcosimOptimizer`. NaN values in the source series are preserved in the output arrays. The arrays are padded/truncated to `n_timesteps` length to match `EcosimOptimizer`'s expectation that observed data length equals `len(self.years)`.

**Uniform timestep handling:** All series within a collection are padded with NaN to the length of the longest series. The collection stores a `n_timesteps` attribute reflecting this uniform length. Loaders are responsible for performing this alignment on construction.

## 2. I/O Layer

### EwE Database Reading

Extend `io/ewemdb.py` with `read_timeseries(db_path) -> EweTimeSeriesCollection`.

**Source tables:**

| Table | Key Columns | Purpose |
|---|---|---|
| `EcosimTimeSeries` | TimeSeriesID, Name, DatType, GroupID, FleetID, DatasetID, WtType, PoolColor | Series metadata |
| `EcosimTimeSeriesValues` | TimeSeriesID, TimeStep, Value | Data points |
| `EcosimTimeSeriesDataset` | DatasetID, DatasourceName, Enabled | Dataset grouping |
| `EcosimTimeSeriesSeason` | TimeSeriesID, Season, Value | Seasonal patterns |

These 4 tables must be added to `io/_ewe_schema.py` with full column definitions (names, types, constraints) before implementation. Missing tables in a database produce an empty collection (graceful degradation for older databases).

### CSV Loading

**New module:** `io/timeseries_csv.py`

`load_timeseries_csv(path, format="ewe") -> EweTimeSeriesCollection`

- `"ewe"` format: header row with series names, DatType row, then timestep rows (matches EwE CSV export)
- `"simple"` format: columns = `time, group, value, dat_type`

### Convenience Loader

```python
def load_timeseries(path: str | Path) -> EweTimeSeriesCollection:
```

Dispatches based on file extension: `.eweaccdb`/`.ewemdb`/`.accdb` → `read_timeseries()`, `.csv` → `load_timeseries_csv()`. Defined in `core/timeseries.py`.

### Export

Extend `io/ewe_writer.py` (and its backends `_access_writer.py` / `_csv_bundle_writer.py`) with `write_timeseries(collection, db_path)` — inserts into the 4 schema tables.

CSV export via `collection.to_dataframe().to_csv()`.

### Integration

`read_timeseries(db_path)` is a standalone function — users call it separately from `read_ewemdb()`. No modifications to `read_ewemdb()` or `RpathParams` are needed. The usage pattern is:

```python
rpath = read_ewemdb("model.eweaccdb")
ts = read_timeseries("model.eweaccdb")  # separate call, same db
```

## 3. Applying Driver Series

**New function in `timeseries.py`:**

```python
def apply_timeseries_drivers(scenario, collection: EweTimeSeriesCollection) -> None:
```

Maps driver series to specific scenario dataclass fields:

| DatType | Target Field | Effect |
|---|---|---|
| `-1` (forced biomass) | `scenario.forcing.ForcedBio[group, timestep]` | Overrides dynamics for that group |
| `3` (effort) | `scenario.fishing.ForcedEffort[fleet, timestep]` | Scales fleet catchability |
| `2` (fishing mortality) | `scenario.fishing.ForcedFRate[group, timestep]` | Direct F override |

Interpolates if time series timesteps don't align with simulation step count. Warns on unknown groups/fleets. Raises `ValueError` on negative forced biomass.

**Usage pattern:**

```python
scenario = rsim_scenario(rpath, params)
ts = load_timeseries("model.eweaccdb")
apply_timeseries_drivers(scenario, ts)
result = rsim_run(scenario, years=50)
```

No changes to `rsim_run()` — it already reads `ForcedBio`/`ForcedEffort`/`ForcedFRate` from the scenario.

## 4. Calibration API

### fit_to_timeseries()

```python
def fit_to_timeseries(
    rpath,
    params,
    timeseries: EweTimeSeriesCollection | dict,
    *,
    fit_vv: bool = True,
    fit_pp: bool = False,
    fit_groups: list[int] | None = None,    # None = all with observed data
    vv_bounds: tuple = (1.0, 100.0),
    pp_bounds: tuple = (0.0, 2.0),
    method: str = "differential_evolution",  # or "minimize" for L-BFGS-B
    max_iterations: int = 1000,
    verbose: bool = False,
) -> CalibrationResult:
```

### CalibrationResult Dataclass

```python
@dataclass
class CalibrationResult:
    best_vv: np.ndarray          # fitted vulnerability values per pred-prey link
    best_pp: np.ndarray | None   # fitted primary production anomaly (if fit_pp)
    ss: float                    # final sum-of-squares
    ss_by_group: dict[int, float]  # SS contribution per group
    n_iterations: int
    converged: bool
    fitted_scenario: dict        # ready-to-use scenario with best params
    link_map: list[tuple[int, int]]  # (prey_idx, pred_idx) for each entry in best_vv
```

The `link_map` field maps each index in `best_vv` to the corresponding `(prey_idx, pred_idx)` pair, so callers can interpret which VV value corresponds to which ecological interaction.

### Objective Function (EwE-standard SS)

```
SS = Σ_series Σ_t  weight_i * ((log(predicted/observed))^2)
```

Log-ratio as in EwE 6 — handles relative biomass naturally (scale-free). Each series' `weight` from `EweTimeSeries.weight` scales its contribution. Series with `dat_type=0` (relative biomass) are rescaled: `predicted_scaled = predicted * (mean_observed / mean_predicted)`. Timesteps where either predicted or observed is NaN are skipped.

### Fitting Flow

1. Convert `timeseries` to `EweTimeSeriesCollection` if dict passed (backward compat)
2. Apply driver series to scenario
3. Build parameter vector from initial VV (and PP if requested), only for links connected to groups with observed data
4. On each iteration: inject params → `rsim_run()` → extract predicted → compute SS
5. Return `CalibrationResult` with best parameters, a pre-built scenario, and link metadata

### Parameter Mapping

VV values live on pred-prey links, not groups. `fit_groups` filters to links where either predator or prey has observed data. A mapping array translates between the flat optimizer vector and the link indices in the scenario. The `link_map` in `CalibrationResult` preserves this mapping for interpretation.

## 5. Testing Strategy

### Unit Tests (`test_timeseries.py`)

- `EweTimeSeries` and `EweTimeSeriesCollection` construction, filtering, `to_observed_dict()`
- DatType constants match expected values
- `apply_timeseries_drivers()` fills correct scenario arrays (`scenario.fishing.ForcedEffort`, `scenario.fishing.ForcedFRate`, `scenario.forcing.ForcedBio`)
- Validation: warns on unknown groups, errors on negative forced biomass
- NaN padding: series with different lengths are padded to uniform length

### Integration Tests (`test_calibration.py`)

- Build a small 3-group model (producer → consumer → predator)
- Generate synthetic observed data by running Ecosim with known VV, adding noise
- Fit with `fit_to_timeseries()` — assert recovered VV is within 20% of true values
- Assert SS decreases from initial to fitted
- Test with dict input (backward compat) and `EweTimeSeriesCollection` input
- Test `fit_pp=True` path
- Verify `link_map` entries correspond to actual pred-prey links

### I/O Tests (`test_timeseries_io.py`)

- Round-trip: create collection → write to CSV → read back → assert equal
- CSV format parsing (both EwE and simple formats)
- Database reading: `@pytest.mark.integration` only (requires Access driver)
- Empty/missing tables → empty collection (graceful degradation)

### Test Markers

- Calibration integration tests: `@pytest.mark.slow`
- Database I/O tests: `@pytest.mark.integration`
- Unit tests: no markers (fast)

## 6. Files Changed/Created

| File | Action |
|---|---|
| `src/pypath/core/timeseries.py` | **Create** — data structures, constants, `load_timeseries()`, `apply_timeseries_drivers()` |
| `src/pypath/core/calibration.py` | **Create** — `fit_to_timeseries()`, `CalibrationResult` |
| `src/pypath/io/timeseries_csv.py` | **Create** — CSV loading |
| `src/pypath/io/ewemdb.py` | **Modify** — add `read_timeseries()` |
| `src/pypath/io/_ewe_schema.py` | **Modify** — add 4 time series table definitions |
| `src/pypath/io/ewe_writer.py` | **Modify** — add `write_timeseries()` |
| `src/pypath/io/_access_writer.py` | **Modify** — time series table writing |
| `src/pypath/io/_csv_bundle_writer.py` | **Modify** — time series CSV bundle writing |
| `src/pypath/core/__init__.py` | **Modify** — export `timeseries`, `calibration` |
| `src/pypath/io/__init__.py` | **Modify** — export `read_timeseries`, `load_timeseries_csv` |
| `tests/test_timeseries.py` | **Create** — unit tests |
| `tests/test_calibration.py` | **Create** — integration tests |
| `tests/test_timeseries_io.py` | **Create** — I/O tests |

### Public API Import Paths

```python
from pypath.core.timeseries import (
    EweTimeSeries, EweTimeSeriesCollection, load_timeseries,
    apply_timeseries_drivers, DATTYPE_REL_BIOMASS, DATTYPE_CATCH, ...
)
from pypath.core.calibration import fit_to_timeseries, CalibrationResult
from pypath.io.ewemdb import read_timeseries
from pypath.io.timeseries_csv import load_timeseries_csv
```

## 7. Design Decisions

1. **Standalone `fit_to_timeseries()`** rather than extending `EcosimOptimizer` — cleaner separation, EwE-standard SS objective, existing optimizer remains untouched for users who depend on it.
2. **VV + PP + configurable additional parameters** — matches EwE 6 calibration scope without over-engineering.
3. **Separate `read_timeseries()` and `apply_timeseries_drivers()`** — composable functions; users can load once, apply to multiple scenarios. No modifications to `read_ewemdb()` or `RpathParams`.
4. **Accept both `EweTimeSeriesCollection` and raw dict** — backward compatibility with existing scripts like `calibrate_lt2022.py`.
5. **Log-ratio SS objective** — matches EwE 6 standard, scale-free for relative biomass.
6. **Uniform timestep padding** — all series in a collection are NaN-padded to the longest series length, simplifying downstream iteration.
7. **`link_map` in `CalibrationResult`** — provides (prey, pred) index pairs so fitted VV values can be interpreted without re-deriving the mapping.
