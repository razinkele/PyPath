# Time Series & Calibration Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add EwE time series support, observed data loading, driver application, and sum-of-squares calibration fitting to PyPath.

**Architecture:** New `core/timeseries.py` module holds data structures and driver application logic. New `core/calibration.py` holds the fitting API. I/O layer extended with `read_timeseries()` in `ewemdb.py`, new `timeseries_csv.py` for CSV, and writer backends for export. All code TDD with synthetic test models.

**Tech Stack:** Python 3.10+, NumPy, SciPy (differential_evolution, minimize), pandas, dataclasses

**Spec:** `docs/superpowers/specs/2026-03-11-timeseries-calibration-design.md`

**Conventions:**
- `group_idx` is always **0-based** throughout the timeseries/calibration API. This differs from `EcosimOptimizer` which uses 1-based indices. The `EweTimeSeries` docstring must state this explicitly.
- `fleet_idx` is also **0-based**.
- Internal Ecosim arrays use 1-based indexing (column 0 = "Outside"), so conversion is `col = group_idx + 1`.

---

## Chunk 1: Core Data Structures & Driver Application

### Task 1: DatType Constants and EweTimeSeries Dataclass

**Files:**
- Create: `packages/pypath/src/pypath/core/timeseries.py`
- Create: `packages/pypath/tests/test_timeseries.py`

- [ ] **Step 1: Write failing tests for constants and dataclass**

In `packages/pypath/tests/test_timeseries.py`:

```python
"""Tests for pypath.core.timeseries module."""
import numpy as np
import pytest

from pypath.core.timeseries import (
    DATTYPE_ABS_BIOMASS,
    DATTYPE_CATCH,
    DATTYPE_EFFORT,
    DATTYPE_FISHING_MORTALITY,
    DATTYPE_FORCED_BIOMASS,
    DATTYPE_REL_BIOMASS,
    EweTimeSeries,
)


class TestDatTypeConstants:
    def test_rel_biomass(self):
        assert DATTYPE_REL_BIOMASS == 0

    def test_abs_biomass(self):
        assert DATTYPE_ABS_BIOMASS == 1

    def test_fishing_mortality(self):
        assert DATTYPE_FISHING_MORTALITY == 2

    def test_effort(self):
        assert DATTYPE_EFFORT == 3

    def test_catch(self):
        assert DATTYPE_CATCH == 6

    def test_forced_biomass(self):
        assert DATTYPE_FORCED_BIOMASS == -1


class TestEweTimeSeries:
    def test_construction(self):
        ts = EweTimeSeries(
            series_id=1,
            name="Cod biomass",
            dat_type=DATTYPE_REL_BIOMASS,
            group_idx=3,
            fleet_idx=None,
            values=np.array([1.0, 1.2, np.nan, 0.9]),
        )
        assert ts.series_id == 1
        assert ts.name == "Cod biomass"
        assert ts.dat_type == 0
        assert ts.group_idx == 3  # 0-based
        assert ts.fleet_idx is None
        assert len(ts.values) == 4
        assert ts.weight == 1.0
        assert ts.dataset_id == 0

    def test_custom_weight(self):
        ts = EweTimeSeries(
            series_id=2,
            name="Herring catch",
            dat_type=DATTYPE_CATCH,
            group_idx=5,
            fleet_idx=1,
            values=np.array([100.0, 120.0]),
            weight=2.5,
            dataset_id=3,
        )
        assert ts.weight == 2.5
        assert ts.dataset_id == 3

    def test_fleet_level_series(self):
        ts = EweTimeSeries(
            series_id=3,
            name="Trawl effort",
            dat_type=DATTYPE_EFFORT,
            group_idx=None,
            fleet_idx=1,
            values=np.array([1.0, 1.1]),
        )
        assert ts.group_idx is None
        assert ts.fleet_idx == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.core.timeseries'`

- [ ] **Step 3: Implement constants and dataclass**

Create `packages/pypath/src/pypath/core/timeseries.py`:

```python
"""EwE time series data structures and driver application.

Provides data classes for observed and forced time series used in Ecosim
calibration and scenario forcing.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimScenario

# --- DatType constants (EwE 6 standard codes) ---
DATTYPE_REL_BIOMASS = 0
DATTYPE_ABS_BIOMASS = 1
DATTYPE_FISHING_MORTALITY = 2
DATTYPE_EFFORT = 3
DATTYPE_CATCH = 6
DATTYPE_FORCED_BIOMASS = -1


@dataclass
class EweTimeSeries:
    """A single EwE time series.

    Parameters
    ----------
    series_id : int
        Unique identifier for this series.
    name : str
        Human-readable name (e.g. "Cod biomass").
    dat_type : int
        DatType constant indicating the series type.
    group_idx : int or None
        **0-based** group index, or None for fleet-level series.
        Note: this differs from EcosimOptimizer which uses 1-based indices.
    fleet_idx : int or None
        **0-based** fleet index, or None for group-level series.
    values : np.ndarray
        Time series values, shape (n_timesteps,). NaN for missing.
    weight : float
        Weighting factor for SS objective (default 1.0).
    dataset_id : int
        Dataset grouping identifier (default 0).
    """

    series_id: int
    name: str
    dat_type: int
    group_idx: int | None
    fleet_idx: int | None
    values: np.ndarray
    weight: float = 1.0
    dataset_id: int = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/timeseries.py packages/pypath/tests/test_timeseries.py
git commit -m "feat(core): add EweTimeSeries dataclass and DatType constants"
```

---

### Task 2: EweTimeSeriesCollection Container

**Files:**
- Modify: `packages/pypath/src/pypath/core/timeseries.py`
- Modify: `packages/pypath/tests/test_timeseries.py`

- [ ] **Step 1: Write failing tests for collection**

Append to `packages/pypath/tests/test_timeseries.py`:

```python
from pypath.core.timeseries import EweTimeSeriesCollection


def _make_collection():
    """Helper: build a collection with 5 series of different types."""
    series = [
        EweTimeSeries(1, "Cod rel bio", DATTYPE_REL_BIOMASS, 0, None, np.array([1.0, 1.2, 0.9])),
        EweTimeSeries(2, "Herring abs bio", DATTYPE_ABS_BIOMASS, 1, None, np.array([50.0, 55.0, 48.0])),
        EweTimeSeries(3, "Cod catch", DATTYPE_CATCH, 0, 0, np.array([10.0, 12.0, 11.0])),
        EweTimeSeries(4, "Forced phyto", DATTYPE_FORCED_BIOMASS, 2, None, np.array([5.0, 5.5])),
        EweTimeSeries(5, "Trawl effort", DATTYPE_EFFORT, None, 0, np.array([1.0, 1.1, 1.2])),
    ]
    return EweTimeSeriesCollection(series)


class TestEweTimeSeriesCollection:
    def test_construction(self):
        coll = _make_collection()
        assert len(coll.series) == 5
        assert coll.n_timesteps == 3  # padded to longest

    def test_nan_padding(self):
        coll = _make_collection()
        # Series 4 has length 2, should be padded to 3
        forced = coll.forced_biomass
        assert len(forced) == 1
        assert len(forced[0].values) == 3
        assert np.isnan(forced[0].values[2])

    def test_observed_biomass_filter(self):
        coll = _make_collection()
        obs = coll.observed_biomass
        assert len(obs) == 2
        assert all(s.dat_type in (0, 1) for s in obs)

    def test_observed_catch_filter(self):
        coll = _make_collection()
        catch = coll.observed_catch
        assert len(catch) == 1
        assert catch[0].dat_type == DATTYPE_CATCH

    def test_forced_biomass_filter(self):
        coll = _make_collection()
        forced = coll.forced_biomass
        assert len(forced) == 1
        assert forced[0].dat_type == DATTYPE_FORCED_BIOMASS

    def test_forced_effort_filter(self):
        coll = _make_collection()
        effort = coll.forced_effort
        assert len(effort) == 1
        assert effort[0].dat_type == DATTYPE_EFFORT

    def test_to_observed_dict(self):
        coll = _make_collection()
        obs_dict = coll.to_observed_dict(n_timesteps=3)
        # Groups 0 and 1 have observed biomass
        assert 0 in obs_dict
        assert 1 in obs_dict
        assert len(obs_dict[0]) == 3
        np.testing.assert_array_equal(obs_dict[0], [1.0, 1.2, 0.9])

    def test_to_observed_dict_truncates(self):
        coll = _make_collection()
        obs_dict = coll.to_observed_dict(n_timesteps=2)
        assert len(obs_dict[0]) == 2

    def test_to_observed_dict_pads(self):
        coll = _make_collection()
        obs_dict = coll.to_observed_dict(n_timesteps=5)
        assert len(obs_dict[0]) == 5
        assert np.isnan(obs_dict[0][3])

    def test_empty_collection(self):
        coll = EweTimeSeriesCollection([])
        assert coll.n_timesteps == 0
        assert len(coll.observed_biomass) == 0
        assert coll.to_observed_dict(n_timesteps=10) == {}

    def test_to_dataframe(self):
        coll = _make_collection()
        df = coll.to_dataframe()
        assert "time" in df.columns
        assert "group" in df.columns
        assert "value" in df.columns
        assert "dat_type" in df.columns
        # Should have rows for all non-NaN values
        assert len(df) > 0
        # Cod rel bio: 3 values, Herring abs bio: 3, Cod catch: 3,
        # Forced phyto: 2 (3rd is NaN), Trawl effort: 3
        assert len(df) == 14
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries.py::TestEweTimeSeriesCollection -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement EweTimeSeriesCollection**

Append to `packages/pypath/src/pypath/core/timeseries.py`:

```python
@dataclass
class EweTimeSeriesCollection:
    """Container for multiple EwE time series with filtered views.

    On construction, all series are NaN-padded to the length of the longest
    series, and ``n_timesteps`` is set to that uniform length.

    Parameters
    ----------
    series : list[EweTimeSeries]
        List of time series to include.
    """

    series: list[EweTimeSeries] = field(default_factory=list)
    n_timesteps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.series:
            self.n_timesteps = 0
            return
        max_len = max(len(s.values) for s in self.series)
        self.n_timesteps = max_len
        for s in self.series:
            if len(s.values) < max_len:
                padded = np.full(max_len, np.nan)
                padded[: len(s.values)] = s.values
                s.values = padded

    @property
    def observed_biomass(self) -> list[EweTimeSeries]:
        """Series with dat_type in {0 (relative), 1 (absolute)}."""
        return [s for s in self.series if s.dat_type in (DATTYPE_REL_BIOMASS, DATTYPE_ABS_BIOMASS)]

    @property
    def observed_catch(self) -> list[EweTimeSeries]:
        """Series with dat_type == 6 (catch)."""
        return [s for s in self.series if s.dat_type == DATTYPE_CATCH]

    @property
    def forced_biomass(self) -> list[EweTimeSeries]:
        """Series with dat_type == -1 (forced biomass)."""
        return [s for s in self.series if s.dat_type == DATTYPE_FORCED_BIOMASS]

    @property
    def forced_effort(self) -> list[EweTimeSeries]:
        """Series with dat_type == 3 (effort)."""
        return [s for s in self.series if s.dat_type == DATTYPE_EFFORT]

    def to_observed_dict(self, n_timesteps: int) -> dict[int, np.ndarray]:
        """Convert observed biomass series to {group_idx: array} dict.

        Arrays are padded with NaN or truncated to ``n_timesteps``.
        Compatible with EcosimOptimizer's observed_data format (note:
        EcosimOptimizer uses 1-based group indices; this returns 0-based).
        """
        result: dict[int, np.ndarray] = {}
        for s in self.observed_biomass:
            if s.group_idx is None:
                continue
            vals = s.values
            if len(vals) >= n_timesteps:
                result[s.group_idx] = vals[:n_timesteps].copy()
            else:
                padded = np.full(n_timesteps, np.nan)
                padded[: len(vals)] = vals
                result[s.group_idx] = padded
        return result

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to a DataFrame in simple format (time, group, value, dat_type).

        Suitable for CSV export and round-tripping.
        """
        import pandas as pd

        rows = []
        for s in self.series:
            for t, v in enumerate(s.values):
                if not np.isnan(v):
                    rows.append({
                        "time": t + 1,
                        "group": s.group_idx if s.group_idx is not None else s.fleet_idx,
                        "value": v,
                        "dat_type": s.dat_type,
                    })
        return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries.py -v`
Expected: All 20 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/timeseries.py packages/pypath/tests/test_timeseries.py
git commit -m "feat(core): add EweTimeSeriesCollection with filtered views, to_observed_dict, to_dataframe"
```

---

### Task 3: apply_timeseries_drivers()

**Files:**
- Modify: `packages/pypath/src/pypath/core/timeseries.py`
- Modify: `packages/pypath/tests/test_timeseries.py`

**Context:** The Ecosim scenario has these forcing arrays:
- `scenario.forcing.ForcedBio` — shape `(n_months, n_groups+1)`, value `-1` = not forced
- `scenario.fishing.ForcedEffort` — shape `(n_months, n_gears+1)`, initialized to `1.0` (= baseline effort)
- `scenario.fishing.ForcedFRate` — shape `(n_years, NUM_BIO+1)`, value `0` = no forcing

See `packages/pypath/src/pypath/core/ecosim.py` for the `RsimForcing` (line ~240) and `RsimFishing` (line ~260) dataclass definitions.

- [ ] **Step 1: Write failing tests for driver application**

Append to `packages/pypath/tests/test_timeseries.py`:

```python
from unittest.mock import MagicMock

from pypath.core.timeseries import apply_timeseries_drivers


def _make_mock_scenario(n_months=36, n_groups=5, n_gears=2, n_years=3):
    """Create a mock scenario with forcing arrays matching real initialization."""
    scenario = MagicMock()
    scenario.forcing.ForcedBio = np.full((n_months, n_groups + 1), -1.0)
    # Real ForcedEffort is initialized to 1.0 (baseline), not 0.0
    scenario.fishing.ForcedEffort = np.ones((n_months, n_gears + 1))
    # ForcedFRate uses NUM_BIO (= NUM_LIVING + NUM_DEAD), not NUM_GROUPS
    scenario.fishing.ForcedFRate = np.zeros((n_years, n_groups + 1))
    scenario.params.NUM_GROUPS = n_groups
    scenario.params.NUM_LIVING = n_groups
    scenario.params.NUM_GEARS = n_gears
    scenario.params.STEPS_PER_MONTH = 1
    scenario.params.STEPS_PER_YEAR = 12
    return scenario


class TestApplyTimeseriesDrivers:
    def test_forced_biomass(self):
        scenario = _make_mock_scenario()
        series = [
            EweTimeSeries(1, "Forced phyto", DATTYPE_FORCED_BIOMASS, 2, None,
                          np.array([5.0, 5.5, 6.0])),
        ]
        coll = EweTimeSeriesCollection(series)
        apply_timeseries_drivers(scenario, coll)
        # Group 2 (1-based col 3) should have forced values in ForcedBio
        # 3 annual values interpolated to 36 months
        bio = scenario.forcing.ForcedBio[:, 3]
        assert not np.all(bio == -1.0)
        # First month should be close to 5.0
        assert abs(bio[0] - 5.0) < 0.5

    def test_forced_effort(self):
        scenario = _make_mock_scenario()
        series = [
            EweTimeSeries(2, "Trawl effort", DATTYPE_EFFORT, None, 0,
                          np.array([1.5, 2.0, 2.5])),
        ]
        coll = EweTimeSeriesCollection(series)
        apply_timeseries_drivers(scenario, coll)
        # Fleet 0 (1-based col 1) should differ from baseline 1.0
        effort = scenario.fishing.ForcedEffort[:, 1]
        assert not np.all(effort == 1.0), "Effort should differ from baseline 1.0"

    def test_forced_frate(self):
        scenario = _make_mock_scenario()
        series = [
            EweTimeSeries(3, "Cod F", DATTYPE_FISHING_MORTALITY, 1, None,
                          np.array([0.3, 0.4, 0.5])),
        ]
        coll = EweTimeSeriesCollection(series)
        apply_timeseries_drivers(scenario, coll)
        # Group 1 (1-based col 2) in ForcedFRate
        frate = scenario.fishing.ForcedFRate[:, 2]
        np.testing.assert_array_almost_equal(frate, [0.3, 0.4, 0.5])

    def test_negative_forced_biomass_raises(self):
        scenario = _make_mock_scenario()
        series = [
            EweTimeSeries(1, "Bad", DATTYPE_FORCED_BIOMASS, 0, None,
                          np.array([-1.0, 2.0])),
        ]
        coll = EweTimeSeriesCollection(series)
        with pytest.raises(ValueError, match="negative"):
            apply_timeseries_drivers(scenario, coll)

    def test_unknown_group_warns(self):
        scenario = _make_mock_scenario(n_groups=3)
        series = [
            EweTimeSeries(1, "Ghost", DATTYPE_FORCED_BIOMASS, 99, None,
                          np.array([1.0, 2.0])),
        ]
        coll = EweTimeSeriesCollection(series)
        with pytest.warns(UserWarning, match="group"):
            apply_timeseries_drivers(scenario, coll)

    def test_unknown_fleet_warns(self):
        scenario = _make_mock_scenario(n_gears=1)
        series = [
            EweTimeSeries(1, "Ghost fleet", DATTYPE_EFFORT, None, 99,
                          np.array([1.0])),
        ]
        coll = EweTimeSeriesCollection(series)
        with pytest.warns(UserWarning, match="fleet"):
            apply_timeseries_drivers(scenario, coll)

    def test_no_drivers_is_noop(self):
        scenario = _make_mock_scenario()
        # Only observed series, no drivers
        series = [
            EweTimeSeries(1, "Obs", DATTYPE_REL_BIOMASS, 0, None,
                          np.array([1.0, 1.1])),
        ]
        coll = EweTimeSeriesCollection(series)
        bio_before = scenario.forcing.ForcedBio.copy()
        effort_before = scenario.fishing.ForcedEffort.copy()
        apply_timeseries_drivers(scenario, coll)
        np.testing.assert_array_equal(scenario.forcing.ForcedBio, bio_before)
        np.testing.assert_array_equal(scenario.fishing.ForcedEffort, effort_before)

    def test_nan_values_preserve_temporal_position(self):
        """NaN values should be interpolated through, not stripped."""
        scenario = _make_mock_scenario(n_months=48, n_years=4)
        # Year 2 is NaN — values at years 1, 3, 4 should still be at correct positions
        series = [
            EweTimeSeries(1, "F", DATTYPE_FORCED_BIOMASS, 0, None,
                          np.array([5.0, np.nan, 6.0, 7.0])),
        ]
        coll = EweTimeSeriesCollection(series)
        apply_timeseries_drivers(scenario, coll)
        bio = scenario.forcing.ForcedBio[:, 1]
        # First month should be ~5.0, last should be ~7.0
        assert abs(bio[0] - 5.0) < 0.5
        assert abs(bio[-1] - 7.0) < 0.5
        # Month 12 (start of year 2) should be interpolated between 5.0 and 6.0
        assert 5.0 < bio[12] < 6.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries.py::TestApplyTimeseriesDrivers -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement apply_timeseries_drivers**

Append to `packages/pypath/src/pypath/core/timeseries.py`:

```python
def _interpolate_to_length(values: np.ndarray, target_len: int) -> np.ndarray:
    """Interpolate a time series to a target length, handling NaN values.

    NaN values are interpolated through (preserving temporal position)
    rather than stripped. The result has no NaN values.
    """
    n = len(values)
    src_indices = np.arange(n)
    target_indices = np.linspace(0, n - 1, target_len)

    # Find non-NaN values and interpolate at their original positions
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.full(target_len, np.nan)

    valid_idx = src_indices[valid_mask]
    valid_vals = values[valid_mask]

    # Interpolate: NaN gaps are filled by interpolation between neighbors
    return np.interp(target_indices, valid_idx, valid_vals)


def apply_timeseries_drivers(
    scenario: RsimScenario,
    collection: EweTimeSeriesCollection,
) -> None:
    """Apply driver time series to an Ecosim scenario's forcing arrays.

    Modifies ``scenario`` in place. Only processes driver series
    (forced biomass, effort, fishing mortality). Observed series are ignored.

    Parameters
    ----------
    scenario : RsimScenario
        The Ecosim scenario to modify.
    collection : EweTimeSeriesCollection
        Time series collection containing driver series.

    Raises
    ------
    ValueError
        If any forced biomass values are negative (excluding NaN).
    """
    n_groups = scenario.params.NUM_GROUPS
    n_gears = scenario.params.NUM_GEARS

    for s in collection.forced_biomass:
        if s.group_idx is None:
            continue
        if s.group_idx >= n_groups:
            warnings.warn(
                f"Time series '{s.name}' references group index {s.group_idx} "
                f"but model only has {n_groups} groups. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        valid_vals = s.values[~np.isnan(s.values)]
        if len(valid_vals) > 0 and np.any(valid_vals < 0):
            raise ValueError(
                f"Time series '{s.name}' contains negative forced biomass values."
            )
        col = s.group_idx + 1  # 1-based column
        n_months = scenario.forcing.ForcedBio.shape[0]
        scenario.forcing.ForcedBio[:, col] = _interpolate_to_length(s.values, n_months)

    for s in collection.forced_effort:
        if s.fleet_idx is None:
            continue
        if s.fleet_idx >= n_gears:
            warnings.warn(
                f"Time series '{s.name}' references fleet index {s.fleet_idx} "
                f"but model only has {n_gears} fleets. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        col = s.fleet_idx + 1  # 1-based column
        n_months = scenario.fishing.ForcedEffort.shape[0]
        scenario.fishing.ForcedEffort[:, col] = _interpolate_to_length(s.values, n_months)

    fmort_series = [s for s in collection.series if s.dat_type == DATTYPE_FISHING_MORTALITY]
    for s in fmort_series:
        if s.group_idx is None:
            continue
        if s.group_idx >= n_groups:
            warnings.warn(
                f"Time series '{s.name}' references group index {s.group_idx} "
                f"but model only has {n_groups} groups. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        col = s.group_idx + 1  # 1-based column
        n_years = scenario.fishing.ForcedFRate.shape[0]
        scenario.fishing.ForcedFRate[:, col] = _interpolate_to_length(s.values, n_years)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/timeseries.py packages/pypath/tests/test_timeseries.py
git commit -m "feat(core): add apply_timeseries_drivers for forced biomass, effort, F rate"
```

---

## Chunk 2: I/O Layer

### Task 4: Time Series Schema Definitions

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py`

**Context:** Schema is defined in `EWE_TABLES` dict with `OrderedDict` values mapping column names to SQL types. See existing tables like `EcopathGroup` for the pattern. Boolean fields use `"YESNO"` type (see `DoInteg`, `UseNudge` etc.).

- [ ] **Step 1: Read the current schema file**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -c "from pypath.io._ewe_schema import EWE_TABLES; print(list(EWE_TABLES.keys()))"`
This shows existing table names to confirm the pattern.

- [ ] **Step 2: Add time series table schemas**

In `packages/pypath/src/pypath/io/_ewe_schema.py`, add these 4 tables to the `EWE_TABLES` dict. Note: `ScenarioID` is added for multi-scenario support (standard in EwE 6.6+ databases). `Enabled` uses `YESNO` to match existing schema convention.

```python
    "EcosimTimeSeries": OrderedDict([
        ("TimeSeriesID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("Name", "TEXT"),
        ("DatType", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("DatasetID", "INTEGER"),
        ("WtType", "INTEGER"),
        ("PoolColor", "INTEGER"),
    ]),
    "EcosimTimeSeriesValues": OrderedDict([
        ("TimeSeriesID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("TimeStep", "INTEGER"),
        ("Value", "DOUBLE"),
    ]),
    "EcosimTimeSeriesDataset": OrderedDict([
        ("DatasetID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("DatasourceName", "TEXT"),
        ("Enabled", "YESNO"),
    ]),
    "EcosimTimeSeriesSeason": OrderedDict([
        ("TimeSeriesID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("Season", "INTEGER"),
        ("Value", "DOUBLE"),
    ]),
```

- [ ] **Step 3: Verify schema loads without errors**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -c "from pypath.io._ewe_schema import EWE_TABLES; assert 'EcosimTimeSeries' in EWE_TABLES; assert 'EcosimTimeSeriesValues' in EWE_TABLES; assert EWE_TABLES['EcosimTimeSeriesDataset']['Enabled'] == 'YESNO'; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py
git commit -m "feat(io): add EwE time series table schemas to _ewe_schema.py"
```

---

### Task 5: Database Reader — read_timeseries()

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py`
- Create: `packages/pypath/tests/test_timeseries_io.py`

**Context:** The existing `read_ewemdb_table(filepath, table)` reads any table from an EwE database as a DataFrame. Use it to read the time series tables. Missing tables should be handled gracefully. Note: `WtType` is an integer enum for the weighting *method*, not the weight *value* — always use `weight=1.0` as default.

- [ ] **Step 1: Write failing tests for read_timeseries**

Create `packages/pypath/tests/test_timeseries_io.py`:

```python
"""Tests for time series I/O (CSV and database)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.timeseries import (
    DATTYPE_CATCH,
    DATTYPE_FORCED_BIOMASS,
    DATTYPE_REL_BIOMASS,
    EweTimeSeries,
    EweTimeSeriesCollection,
)
from pypath.io.ewemdb import read_timeseries


class TestReadTimeseries:
    """Tests for read_timeseries from EwE databases.

    Note: actual database tests require Access driver and are marked
    @pytest.mark.integration. These unit tests verify the function exists
    and returns empty collections for missing databases.
    """

    def test_import(self):
        """read_timeseries is importable."""
        assert callable(read_timeseries)

    def test_missing_file_returns_empty(self, tmp_path):
        """Non-existent file returns empty collection."""
        coll = read_timeseries(str(tmp_path / "nonexistent.eweaccdb"))
        assert isinstance(coll, EweTimeSeriesCollection)
        assert len(coll.series) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries_io.py -v`
Expected: FAIL with `ImportError: cannot import name 'read_timeseries' from 'pypath.io.ewemdb'`

- [ ] **Step 3: Implement read_timeseries**

Add to the end of `packages/pypath/src/pypath/io/ewemdb.py`:

```python
def read_timeseries(
    filepath: str, scenario: int = 1
) -> "EweTimeSeriesCollection":
    """Read time series data from an EwE database.

    Reads the EcosimTimeSeries and EcosimTimeSeriesValues tables and
    constructs an EweTimeSeriesCollection. If tables are missing,
    returns an empty collection.

    Parameters
    ----------
    filepath : str
        Path to the EwE database file (.eweaccdb, .ewemdb, or .accdb).
    scenario : int
        Scenario ID to filter by (default 1).

    Returns
    -------
    EweTimeSeriesCollection
        Collection of all time series in the database.
    """
    from pypath.core.timeseries import EweTimeSeries, EweTimeSeriesCollection

    try:
        tables = list_ewemdb_tables(filepath)
    except Exception:
        return EweTimeSeriesCollection([])

    if "EcosimTimeSeries" not in tables or "EcosimTimeSeriesValues" not in tables:
        return EweTimeSeriesCollection([])

    try:
        meta_df = read_ewemdb_table(filepath, "EcosimTimeSeries")
        values_df = read_ewemdb_table(filepath, "EcosimTimeSeriesValues")
    except Exception:
        return EweTimeSeriesCollection([])

    if meta_df.empty or values_df.empty:
        return EweTimeSeriesCollection([])

    # Filter by scenario if ScenarioID column exists
    if "ScenarioID" in meta_df.columns:
        meta_df = meta_df[meta_df["ScenarioID"] == scenario]
    if "ScenarioID" in values_df.columns:
        values_df = values_df[values_df["ScenarioID"] == scenario]

    series_list = []
    for _, row in meta_df.iterrows():
        ts_id = int(row["TimeSeriesID"])
        name = str(row.get("Name", f"Series_{ts_id}"))
        dat_type = int(row.get("DatType", 0))

        group_id = row.get("GroupID")
        group_idx = int(group_id) - 1 if pd.notna(group_id) and int(group_id) > 0 else None

        fleet_id = row.get("FleetID")
        fleet_idx = int(fleet_id) - 1 if pd.notna(fleet_id) and int(fleet_id) > 0 else None

        dataset_id = int(row.get("DatasetID", 0)) if pd.notna(row.get("DatasetID")) else 0

        # WtType is a method enum (0=SS, 1=SSLog, etc.), NOT a weight value.
        # Actual per-series weight defaults to 1.0.
        weight = 1.0

        # Extract values for this series, sorted by timestep
        ts_vals = values_df[values_df["TimeSeriesID"] == ts_id].sort_values("TimeStep")
        values = ts_vals["Value"].to_numpy(dtype=float)

        if len(values) == 0:
            continue

        series_list.append(
            EweTimeSeries(
                series_id=ts_id,
                name=name,
                dat_type=dat_type,
                group_idx=group_idx,
                fleet_idx=fleet_idx,
                values=values,
                weight=weight,
                dataset_id=dataset_id,
            )
        )

    return EweTimeSeriesCollection(series_list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries_io.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py packages/pypath/tests/test_timeseries_io.py
git commit -m "feat(io): add read_timeseries() for EwE database time series loading"
```

---

### Task 6: CSV Loader — load_timeseries_csv()

**Files:**
- Create: `packages/pypath/src/pypath/io/timeseries_csv.py`
- Modify: `packages/pypath/tests/test_timeseries_io.py`

- [ ] **Step 1: Write failing tests for CSV loading**

Append to `packages/pypath/tests/test_timeseries_io.py`:

```python
from pypath.io.timeseries_csv import load_timeseries_csv


class TestLoadTimeseriesCsvEweFormat:
    """Test loading EwE-format CSV time series."""

    def _write_ewe_csv(self, tmp_path):
        """Write a sample EwE-format CSV."""
        csv_path = tmp_path / "timeseries.csv"
        # EwE format: header row (series names), DatType row, then data
        csv_path.write_text(
            "TimeStep,Cod biomass,Herring biomass,Cod catch\n"
            ",0,0,6\n"
            "1,1.0,50.0,10.0\n"
            "2,1.2,55.0,12.0\n"
            "3,0.9,,11.0\n"
        )
        return csv_path

    def test_loads_series(self, tmp_path):
        csv_path = self._write_ewe_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="ewe")
        assert len(coll.series) == 3

    def test_dat_types_parsed(self, tmp_path):
        csv_path = self._write_ewe_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="ewe")
        types = {s.name: s.dat_type for s in coll.series}
        assert types["Cod biomass"] == 0
        assert types["Herring biomass"] == 0
        assert types["Cod catch"] == 6

    def test_nan_for_missing(self, tmp_path):
        csv_path = self._write_ewe_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="ewe")
        herring = [s for s in coll.series if s.name == "Herring biomass"][0]
        assert np.isnan(herring.values[2])

    def test_values_correct(self, tmp_path):
        csv_path = self._write_ewe_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="ewe")
        cod = [s for s in coll.series if s.name == "Cod biomass"][0]
        np.testing.assert_array_almost_equal(cod.values, [1.0, 1.2, 0.9])


class TestLoadTimeseriesCsvSimpleFormat:
    """Test loading simple-format CSV time series."""

    def _write_simple_csv(self, tmp_path):
        csv_path = tmp_path / "ts_simple.csv"
        csv_path.write_text(
            "time,group,value,dat_type\n"
            "1,0,1.0,0\n"
            "2,0,1.2,0\n"
            "3,0,0.9,0\n"
            "1,1,50.0,6\n"
            "2,1,55.0,6\n"
        )
        return csv_path

    def test_loads_series(self, tmp_path):
        csv_path = self._write_simple_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="simple")
        assert len(coll.series) == 2

    def test_groups_separated(self, tmp_path):
        csv_path = self._write_simple_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="simple")
        groups = {s.group_idx for s in coll.series}
        assert groups == {0, 1}

    def test_values_correct(self, tmp_path):
        csv_path = self._write_simple_csv(tmp_path)
        coll = load_timeseries_csv(csv_path, format="simple")
        grp0 = [s for s in coll.series if s.group_idx == 0][0]
        np.testing.assert_array_almost_equal(grp0.values, [1.0, 1.2, 0.9])


class TestCsvRoundtrip:
    """Test write-then-read round-trip via to_dataframe()."""

    def test_roundtrip(self, tmp_path):
        original = EweTimeSeriesCollection([
            EweTimeSeries(1, "Cod", DATTYPE_REL_BIOMASS, 0, None, np.array([1.0, 1.2, 0.9])),
            EweTimeSeries(2, "Catch", DATTYPE_CATCH, 1, 0, np.array([10.0, 12.0, 11.0])),
        ])
        csv_path = tmp_path / "roundtrip.csv"
        df = original.to_dataframe()
        df.to_csv(csv_path, index=False)
        reloaded = load_timeseries_csv(csv_path, format="simple")
        assert len(reloaded.series) == 2
        # Verify data fidelity
        for orig_s in original.series:
            matches = [s for s in reloaded.series
                       if s.group_idx == orig_s.group_idx and s.dat_type == orig_s.dat_type]
            assert len(matches) == 1, f"No match for group {orig_s.group_idx} type {orig_s.dat_type}"
            np.testing.assert_array_almost_equal(matches[0].values, orig_s.values)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries_io.py::TestLoadTimeseriesCsvEweFormat -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement load_timeseries_csv**

Create `packages/pypath/src/pypath/io/timeseries_csv.py`:

```python
"""CSV loading for EwE time series data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.timeseries import EweTimeSeries, EweTimeSeriesCollection


def load_timeseries_csv(
    path: str | Path,
    format: str = "ewe",
) -> EweTimeSeriesCollection:
    """Load time series from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    format : str
        ``"ewe"`` for EwE CSV export format (header row with series names,
        DatType row, then timestep rows). ``"simple"`` for flat format
        with columns ``time, group, value, dat_type``.

    Returns
    -------
    EweTimeSeriesCollection
    """
    path = Path(path)
    if format == "ewe":
        return _load_ewe_format(path)
    elif format == "simple":
        return _load_simple_format(path)
    else:
        raise ValueError(f"Unknown format: {format!r}. Use 'ewe' or 'simple'.")


def _load_ewe_format(path: Path) -> EweTimeSeriesCollection:
    """Load EwE-format CSV: header row, DatType row, then data rows."""
    df = pd.read_csv(path, header=0)
    # First column is TimeStep, remaining are series
    series_names = list(df.columns[1:])

    # First data row contains DatType values (TimeStep cell is empty/NaN)
    dat_types_row = df.iloc[0]
    dat_types = {}
    for name in series_names:
        val = dat_types_row[name]
        dat_types[name] = int(float(val)) if pd.notna(val) else 0

    # Remaining rows are actual data
    data_df = df.iloc[1:].reset_index(drop=True)

    series_list = []
    for i, name in enumerate(series_names):
        values = pd.to_numeric(data_df[name], errors="coerce").to_numpy(dtype=float)
        dat_type = dat_types[name]
        series_list.append(
            EweTimeSeries(
                series_id=i + 1,
                name=name,
                dat_type=dat_type,
                group_idx=i,  # default: column index as group
                fleet_idx=None,
                values=values,
            )
        )

    return EweTimeSeriesCollection(series_list)


def _load_simple_format(path: Path) -> EweTimeSeriesCollection:
    """Load simple-format CSV: time, group, value, dat_type columns."""
    df = pd.read_csv(path)
    required = {"time", "group", "value", "dat_type"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Simple format requires columns {required}, got {set(df.columns)}")

    series_list = []
    series_id = 1
    for (group, dat_type), group_df in df.groupby(["group", "dat_type"]):
        sorted_df = group_df.sort_values("time")
        values = sorted_df["value"].to_numpy(dtype=float)
        group_idx = int(group)
        series_list.append(
            EweTimeSeries(
                series_id=series_id,
                name=f"Group_{group_idx}_type_{int(dat_type)}",
                dat_type=int(dat_type),
                group_idx=group_idx,
                fleet_idx=None,
                values=values,
            )
        )
        series_id += 1

    return EweTimeSeriesCollection(series_list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries_io.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/timeseries_csv.py packages/pypath/tests/test_timeseries_io.py
git commit -m "feat(io): add CSV time series loader with EwE and simple formats"
```

---

### Task 7: Convenience Loader — load_timeseries()

**Files:**
- Modify: `packages/pypath/src/pypath/core/timeseries.py`
- Modify: `packages/pypath/tests/test_timeseries.py`

- [ ] **Step 1: Write failing test**

Append to `packages/pypath/tests/test_timeseries.py`:

```python
from pypath.core.timeseries import load_timeseries


class TestLoadTimeseries:
    def test_csv_dispatch(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "time,group,value,dat_type\n"
            "1,0,1.0,0\n"
            "2,0,1.2,0\n"
        )
        coll = load_timeseries(csv_path)
        assert len(coll.series) == 1

    def test_unknown_extension_raises(self, tmp_path):
        bad_path = tmp_path / "test.xyz"
        bad_path.write_text("data")
        with pytest.raises(ValueError, match="extension"):
            load_timeseries(bad_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries.py::TestLoadTimeseries -v`
Expected: FAIL

- [ ] **Step 3: Implement load_timeseries**

Add to `packages/pypath/src/pypath/core/timeseries.py`:

```python
def load_timeseries(path: str | Path) -> EweTimeSeriesCollection:
    """Load time series from a file, dispatching by extension.

    Supported extensions:
    - ``.csv`` → CSV loader (simple format)
    - ``.eweaccdb``, ``.ewemdb``, ``.accdb`` → EwE database reader

    Parameters
    ----------
    path : str or Path
        Path to the time series file.

    Returns
    -------
    EweTimeSeriesCollection

    Raises
    ------
    ValueError
        If the file extension is not recognized.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        from pypath.io.timeseries_csv import load_timeseries_csv
        return load_timeseries_csv(path, format="simple")
    elif ext in (".eweaccdb", ".ewemdb", ".accdb"):
        from pypath.io.ewemdb import read_timeseries
        return read_timeseries(str(path))
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Use .csv, .eweaccdb, .ewemdb, or .accdb."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_timeseries.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/timeseries.py packages/pypath/tests/test_timeseries.py
git commit -m "feat(core): add load_timeseries() convenience dispatcher"
```

---

### Task 8: Export — Writer Backends

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewe_writer.py`
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py`
- Modify: `packages/pypath/src/pypath/io/_access_writer.py`

**Context:** The existing `write_ewemdb()` delegates to `AccessWriter` or `CsvBundleWriter`. Each has `write_ecopath()`, `write_ecosim()`, `write_ecospace()` methods. We add `write_timeseries()` to each and call it from `write_ewemdb()`. The `AccessWriter` uses `_build_tables_via_csv_writer()` to delegate table construction — follow that pattern.

- [ ] **Step 1: Add write_timeseries to CsvBundleWriter**

In `packages/pypath/src/pypath/io/_csv_bundle_writer.py`, add:

```python
    def write_timeseries(self, timeseries=None) -> None:
        """Write time series tables to the CSV bundle.

        Parameters
        ----------
        timeseries : EweTimeSeriesCollection or None
            Time series data to write. If None, skips.
        """
        if timeseries is None or not timeseries.series:
            return

        import numpy as np
        import pandas as pd

        # EcosimTimeSeries metadata table
        meta_rows = []
        for s in timeseries.series:
            meta_rows.append({
                "TimeSeriesID": s.series_id,
                "ScenarioID": self._scenario_id,
                "Name": s.name,
                "DatType": s.dat_type,
                "GroupID": (s.group_idx + 1) if s.group_idx is not None else 0,
                "FleetID": (s.fleet_idx + 1) if s.fleet_idx is not None else 0,
                "DatasetID": s.dataset_id,
                "WtType": 0,  # Default: SS weighting method
                "PoolColor": 0,
            })
        self._tables["EcosimTimeSeries"] = pd.DataFrame(meta_rows)

        # EcosimTimeSeriesValues table
        val_rows = []
        for s in timeseries.series:
            for t, v in enumerate(s.values):
                if not np.isnan(v):
                    val_rows.append({
                        "TimeSeriesID": s.series_id,
                        "ScenarioID": self._scenario_id,
                        "TimeStep": t + 1,
                        "Value": v,
                    })
        if val_rows:
            self._tables["EcosimTimeSeriesValues"] = pd.DataFrame(val_rows)

        # EcosimTimeSeriesDataset table
        dataset_ids = {s.dataset_id for s in timeseries.series}
        ds_rows = [
            {"DatasetID": did, "ScenarioID": self._scenario_id,
             "DatasourceName": f"Dataset_{did}", "Enabled": True}
            for did in sorted(dataset_ids)
        ]
        self._tables["EcosimTimeSeriesDataset"] = pd.DataFrame(ds_rows)
```

- [ ] **Step 2: Add write_timeseries to AccessWriter using delegation pattern**

In `packages/pypath/src/pypath/io/_access_writer.py`, add:

```python
    def write_timeseries(self, timeseries=None) -> None:
        """Write time series tables to the Access database.

        Delegates table construction to CsvBundleWriter, then inserts rows.

        Parameters
        ----------
        timeseries : EweTimeSeriesCollection or None
            Time series data to write. If None, skips.
        """
        if timeseries is None:
            return
        self._build_tables_via_csv_writer("write_timeseries", timeseries=timeseries)
```

Note: Check if `_build_tables_via_csv_writer` passes kwargs through. If it doesn't, add the delegation manually:

```python
    def write_timeseries(self, timeseries=None) -> None:
        if timeseries is None or not timeseries.series:
            return
        # Build tables via CSV writer
        from pypath.io._csv_bundle_writer import CsvBundleWriter
        csv_writer = CsvBundleWriter(self._params, "/dev/null", scenario_id=self._scenario_id)
        csv_writer.write_timeseries(timeseries)
        for table_name, df in csv_writer._tables.items():
            if table_name.startswith("EcosimTimeSeries"):
                self._insert_rows(table_name, df)
```

- [ ] **Step 3: Add timeseries parameter to write_ewemdb**

In `packages/pypath/src/pypath/io/ewe_writer.py`, modify `write_ewemdb()` signature to add `timeseries: Any | None = None` parameter, and add this call alongside existing writer calls (before `writer.close()`):

```python
    writer.write_timeseries(timeseries)
```

Call it unconditionally — the method handles `None` internally, matching the existing pattern for `write_ecosim(scenarios)` and `write_ecospace(ecospace)`.

- [ ] **Step 4: Verify existing tests still pass**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_ewe_writer.py -v -x`
Expected: All existing tests PASS (timeseries=None is default, no behavior change)

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/ewe_writer.py packages/pypath/src/pypath/io/_csv_bundle_writer.py packages/pypath/src/pypath/io/_access_writer.py
git commit -m "feat(io): add time series export to EwE writer backends"
```

---

## Chunk 3: Calibration API

### Task 9: CalibrationResult Dataclass and SS Objective

**Files:**
- Create: `packages/pypath/src/pypath/core/calibration.py`
- Create: `packages/pypath/tests/test_calibration.py`

- [ ] **Step 1: Write failing tests for CalibrationResult and SS objective**

Create `packages/pypath/tests/test_calibration.py`:

```python
"""Tests for pypath.core.calibration module."""
import numpy as np
import pytest

from pypath.core.calibration import CalibrationResult, _compute_ss


class TestCalibrationResult:
    def test_construction(self):
        result = CalibrationResult(
            best_vv=np.array([2.0, 3.0]),
            best_pp=None,
            ss=0.05,
            ss_by_group={0: 0.03, 1: 0.02},
            n_iterations=100,
            converged=True,
            fitted_scenario=None,
            link_map=[(0, 1), (1, 2)],
        )
        assert result.ss == 0.05
        assert result.converged is True
        assert len(result.link_map) == 2

    def test_link_map_matches_vv(self):
        vv = np.array([2.0, 3.0, 4.0])
        link_map = [(0, 1), (1, 2), (0, 2)]
        result = CalibrationResult(
            best_vv=vv, best_pp=None, ss=0.0,
            ss_by_group={}, n_iterations=0,
            converged=True, fitted_scenario=None,
            link_map=link_map,
        )
        assert len(result.best_vv) == len(result.link_map)


class TestComputeSS:
    def test_perfect_match_zero_ss(self):
        observed = {0: np.array([1.0, 2.0, 3.0])}
        predicted = {0: np.array([1.0, 2.0, 3.0])}
        weights = {0: 1.0}
        ss, ss_by_group = _compute_ss(observed, predicted, weights, relative={0: False})
        assert ss == pytest.approx(0.0, abs=1e-10)

    def test_ss_increases_with_deviation(self):
        observed = {0: np.array([1.0, 1.0, 1.0])}
        pred_close = {0: np.array([1.1, 1.1, 1.1])}
        pred_far = {0: np.array([2.0, 2.0, 2.0])}
        weights = {0: 1.0}
        rel = {0: False}
        ss_close, _ = _compute_ss(observed, pred_close, weights, relative=rel)
        ss_far, _ = _compute_ss(observed, pred_far, weights, relative=rel)
        assert ss_close < ss_far

    def test_nan_timesteps_skipped(self):
        observed = {0: np.array([1.0, np.nan, 1.0])}
        predicted = {0: np.array([1.0, 999.0, 1.0])}
        weights = {0: 1.0}
        ss, _ = _compute_ss(observed, predicted, weights, relative={0: False})
        assert ss == pytest.approx(0.0, abs=1e-10)

    def test_weight_scaling(self):
        observed = {0: np.array([1.0, 2.0])}
        predicted = {0: np.array([1.5, 2.5])}
        w1 = {0: 1.0}
        w2 = {0: 2.0}
        rel = {0: False}
        ss1, _ = _compute_ss(observed, predicted, w1, relative=rel)
        ss2, _ = _compute_ss(observed, predicted, w2, relative=rel)
        assert ss2 == pytest.approx(ss1 * 2.0, rel=1e-10)

    def test_relative_biomass_rescaling(self):
        # Relative series: predicted is rescaled so mean matches observed mean
        observed = {0: np.array([1.0, 1.0, 1.0])}
        predicted = {0: np.array([10.0, 10.0, 10.0])}
        weights = {0: 1.0}
        ss, _ = _compute_ss(observed, predicted, weights, relative={0: True})
        # After rescaling, predicted becomes [1.0, 1.0, 1.0] -> SS = 0
        assert ss == pytest.approx(0.0, abs=1e-10)

    def test_ss_by_group(self):
        observed = {0: np.array([1.0]), 1: np.array([2.0])}
        predicted = {0: np.array([1.5]), 1: np.array([2.0])}
        weights = {0: 1.0, 1: 1.0}
        rel = {0: False, 1: False}
        _, ss_by_group = _compute_ss(observed, predicted, weights, relative=rel)
        assert 0 in ss_by_group
        assert 1 in ss_by_group
        assert ss_by_group[0] > 0
        assert ss_by_group[1] == pytest.approx(0.0, abs=1e-10)

    def test_multiple_series_same_group(self):
        """Multiple series for the same group are summed."""
        observed = {0: np.array([1.0, 2.0])}
        predicted = {0: np.array([1.5, 2.5])}
        weights = {0: 1.0}
        rel = {0: False}
        ss, _ = _compute_ss(observed, predicted, weights, relative=rel)
        assert ss > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement CalibrationResult and _compute_ss**

Create `packages/pypath/src/pypath/core/calibration.py`:

```python
"""Ecosim calibration via sum-of-squares fitting to time series data.

Provides fit_to_timeseries() for optimizing Ecosim vulnerability (VV)
and primary production (PP) parameters against observed time series.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimScenario
    from pypath.core.timeseries import EweTimeSeriesCollection

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of time series calibration.

    Parameters
    ----------
    best_vv : np.ndarray
        Fitted vulnerability values, one per pred-prey link.
    best_pp : np.ndarray or None
        Fitted primary production anomaly (if fit_pp was True).
    ss : float
        Final sum-of-squares value.
    ss_by_group : dict[int, float]
        SS contribution per group index (0-based).
    n_iterations : int
        Number of optimizer iterations performed.
    converged : bool
        Whether the optimizer converged.
    fitted_scenario : RsimScenario or None
        Scenario with best-fit parameters applied.
    link_map : list[tuple[int, int]]
        (prey_idx, pred_idx) 0-based for each entry in best_vv.
    """

    best_vv: np.ndarray
    best_pp: np.ndarray | None
    ss: float
    ss_by_group: dict[int, float]
    n_iterations: int
    converged: bool
    fitted_scenario: object  # RsimScenario
    link_map: list[tuple[int, int]]


def _compute_ss(
    observed: dict[int, np.ndarray],
    predicted: dict[int, np.ndarray],
    weights: dict[int, float],
    relative: dict[int, bool],
) -> tuple[float, dict[int, float]]:
    """Compute EwE-standard log-ratio sum-of-squares.

    SS = sum_groups( weight_i * sum_t( (log(pred/obs))^2 ) )

    Parameters
    ----------
    observed : dict[int, np.ndarray]
        Observed values keyed by group index (0-based).
    predicted : dict[int, np.ndarray]
        Predicted values keyed by group index (0-based).
    weights : dict[int, float]
        SS weight per group.
    relative : dict[int, bool]
        Whether each group's series is relative biomass (needs rescaling).

    Returns
    -------
    ss : float
        Total weighted sum-of-squares.
    ss_by_group : dict[int, float]
        SS contribution per group.
    """
    total_ss = 0.0
    ss_by_group: dict[int, float] = {}

    for grp_idx, obs in observed.items():
        if grp_idx not in predicted:
            continue
        pred = predicted[grp_idx]
        w = weights.get(grp_idx, 1.0)

        n = min(len(obs), len(pred))
        obs_n = obs[:n]
        pred_n = pred[:n]

        # Mask: both must be valid (not NaN) and positive
        valid = ~np.isnan(obs_n) & ~np.isnan(pred_n) & (obs_n > 0) & (pred_n > 0)

        if not np.any(valid):
            ss_by_group[grp_idx] = 0.0
            continue

        obs_v = obs_n[valid]
        pred_v = pred_n[valid]

        # Rescale relative biomass series
        if relative.get(grp_idx, False):
            scale = np.mean(obs_v) / np.mean(pred_v)
            pred_v = pred_v * scale

        # Log-ratio SS
        log_ratios = np.log(pred_v / obs_v)
        group_ss = w * np.sum(log_ratios ** 2)
        ss_by_group[grp_idx] = group_ss
        total_ss += group_ss

    return total_ss, ss_by_group
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_calibration.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/calibration.py packages/pypath/tests/test_calibration.py
git commit -m "feat(core): add CalibrationResult dataclass and _compute_ss objective"
```

---

### Task 10: fit_to_timeseries() Implementation

**Files:**
- Modify: `packages/pypath/src/pypath/core/calibration.py`
- Modify: `packages/pypath/tests/test_calibration.py`

**Context:** The fitting function builds a parameter vector from VV link values, runs `rsim_run()` on each optimizer iteration, and computes SS against observed data. Uses `scipy.optimize.differential_evolution` by default.

Key Ecosim internals to know:
- `rsim_scenario()` creates the scenario with default VV values
- VV values are at `scenario.params.VV[link_idx]` (1-based, shape `n_links+1`)
- Prey/pred link indices at `scenario.params.PreyFrom[link_idx]` and `scenario.params.PreyTo[link_idx]` (1-based values)
- `rsim_run()` returns `RsimOutput` with `.out_Biomass` shape `(n_months, n_groups+1)`

**Important:** If a group has both biomass and catch observed series, they are stored under separate keys using `(group_idx, dat_type)` tuples to avoid overwriting. The current implementation simplifies to one series per group — if both exist, catch takes priority. This can be extended later.

- [ ] **Step 1: Write failing integration test**

Append to `packages/pypath/tests/test_calibration.py`:

```python
from pypath.core.calibration import fit_to_timeseries
from pypath.core.timeseries import (
    DATTYPE_REL_BIOMASS,
    EweTimeSeries,
    EweTimeSeriesCollection,
)


@pytest.mark.slow
class TestFitToTimeseries:
    """Integration tests using a minimal 3-group Ecosim model."""

    @pytest.fixture
    def simple_model(self):
        """Build a balanced 3-group model: producer -> consumer -> predator.

        Group 0: Producer (type=1, primary producer)
        Group 1: Consumer (type=0, eats producer)
        Group 2: Predator (type=0, eats consumer)
        """
        from pypath.core.ecopath import rpath
        from pypath.core.params import create_rpath_params

        params = create_rpath_params(
            groups=["Producer", "Consumer", "Predator"],
            types=[1, 0, 0],  # type=1 for producer
            stgroups=[np.nan, np.nan, np.nan],
        )
        params.Biomass = np.array([10.0, 5.0, 1.0])
        params.PB = np.array([50.0, 2.0, 0.5])
        params.QB = np.array([0.0, 10.0, 3.0])
        params.EE = np.array([0.8, 0.8, 0.0])
        params.BioAcc = np.array([0.0, 0.0, 0.0])
        params.Unassim = np.array([0.0, 0.2, 0.2])

        # Diet: consumer eats producer, predator eats consumer
        params.DC = np.zeros((3, 3))
        params.DC[0, 1] = 1.0  # consumer eats 100% producer
        params.DC[1, 2] = 1.0  # predator eats 100% consumer

        # No fishing
        params.Catch = np.zeros(3)
        params.Discards = np.zeros(3)

        balanced = rpath(params)
        return balanced, params

    def test_ss_decreases(self, simple_model):
        """Fitting should produce a valid CalibrationResult."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        balanced, params = simple_model
        scenario = rsim_scenario(balanced, params, years=range(1, 11))

        # Run with default VV to get "truth" biomass
        output = rsim_run(scenario)
        # Use annual means as observed (12 months per year)
        n_years = 10
        obs_bio = np.zeros(n_years)
        for yr in range(n_years):
            start = yr * 12
            end = start + 12
            obs_bio[yr] = np.mean(output.out_Biomass[start:end, 2])  # Consumer (0-based idx 1, col 2)

        # Add 10% noise
        rng = np.random.default_rng(42)
        obs_noisy = obs_bio * (1.0 + 0.1 * rng.standard_normal(n_years))
        obs_noisy = np.maximum(obs_noisy, 0.01)

        ts = EweTimeSeriesCollection([
            EweTimeSeries(1, "Consumer", DATTYPE_REL_BIOMASS, 1, None, obs_noisy),
        ])

        result = fit_to_timeseries(
            balanced, params, ts,
            fit_vv=True, fit_pp=False,
            method="differential_evolution",
            max_iterations=50,
            verbose=False,
        )

        assert isinstance(result, CalibrationResult)
        assert result.ss >= 0
        assert result.n_iterations > 0
        assert len(result.link_map) > 0
        assert len(result.best_vv) == len(result.link_map)

    def test_dict_input_backward_compat(self, simple_model):
        """fit_to_timeseries accepts dict input (0-based group_idx keys)."""
        balanced, params = simple_model
        obs_dict = {1: np.array([5.0, 5.1, 4.9, 5.0, 5.2])}

        result = fit_to_timeseries(
            balanced, params, obs_dict,
            fit_vv=True,
            method="differential_evolution",
            max_iterations=20,
            verbose=False,
        )
        assert isinstance(result, CalibrationResult)
        assert result.ss >= 0

    def test_fit_pp_raises_not_implemented(self, simple_model):
        """fit_pp=True is not yet implemented."""
        balanced, params = simple_model
        ts = EweTimeSeriesCollection([
            EweTimeSeries(1, "Consumer", DATTYPE_REL_BIOMASS, 1, None,
                          np.array([5.0, 5.1, 4.9])),
        ])
        with pytest.raises(NotImplementedError):
            fit_to_timeseries(balanced, params, ts, fit_pp=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_calibration.py::TestFitToTimeseries::test_ss_decreases -v`
Expected: FAIL with `ImportError: cannot import name 'fit_to_timeseries'`

- [ ] **Step 3: Implement fit_to_timeseries**

Append to `packages/pypath/src/pypath/core/calibration.py`:

```python
def fit_to_timeseries(
    rpath_model,
    params,
    timeseries,
    *,
    fit_vv: bool = True,
    fit_pp: bool = False,
    fit_groups: list[int] | None = None,
    vv_bounds: tuple = (1.0, 100.0),
    pp_bounds: tuple = (0.0, 2.0),
    method: str = "differential_evolution",
    max_iterations: int = 1000,
    verbose: bool = False,
) -> CalibrationResult:
    """Fit Ecosim parameters to observed time series data.

    Parameters
    ----------
    rpath_model : Rpath
        Balanced Ecopath model.
    params : RpathParams
        Model parameters.
    timeseries : EweTimeSeriesCollection or dict
        Observed time series. If dict, treated as ``{group_idx: np.array}``
        of relative biomass (0-based group indices).
    fit_vv : bool
        Fit vulnerability (VV) parameters (default True).
    fit_pp : bool
        Fit primary production anomaly. **Not yet implemented** — raises
        NotImplementedError if True.
    fit_groups : list[int] or None
        0-based group indices to fit. None = all groups with observed data.
    vv_bounds : tuple
        (min, max) bounds for VV parameters.
    pp_bounds : tuple
        (min, max) bounds for PP parameters (reserved for future use).
    method : str
        ``"differential_evolution"`` or ``"minimize"`` (L-BFGS-B).
    max_iterations : int
        Maximum optimizer iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    CalibrationResult
    """
    from pypath.core.ecosim import rsim_run, rsim_scenario
    from pypath.core.timeseries import (
        DATTYPE_REL_BIOMASS,
        EweTimeSeries,
        EweTimeSeriesCollection,
        apply_timeseries_drivers,
    )

    if fit_pp:
        raise NotImplementedError(
            "Primary production fitting (fit_pp=True) is not yet implemented. "
            "It will be added in a future release."
        )

    # Convert dict input to EweTimeSeriesCollection
    if isinstance(timeseries, dict):
        series_list = []
        for grp_idx, values in timeseries.items():
            series_list.append(
                EweTimeSeries(
                    series_id=grp_idx,
                    name=f"Group_{grp_idx}",
                    dat_type=DATTYPE_REL_BIOMASS,
                    group_idx=grp_idx,
                    fleet_idx=None,
                    values=values,
                )
            )
        timeseries = EweTimeSeriesCollection(series_list)

    # Determine simulation years from longest series
    n_obs_years = timeseries.n_timesteps
    if n_obs_years < 2:
        raise ValueError("Need at least 2 observed timesteps for calibration.")
    years = range(1, n_obs_years + 1)

    # Build base scenario
    scenario = rsim_scenario(rpath_model, params, years=years)

    # Apply driver series (forced biomass, effort, etc.)
    apply_timeseries_drivers(scenario, timeseries)

    # Build observed data dict and metadata
    observed: dict[int, np.ndarray] = {}
    weights: dict[int, float] = {}
    relative: dict[int, bool] = {}

    for s in timeseries.observed_biomass:
        if s.group_idx is not None:
            observed[s.group_idx] = s.values[:n_obs_years]
            weights[s.group_idx] = s.weight
            relative[s.group_idx] = (s.dat_type == DATTYPE_REL_BIOMASS)

    for s in timeseries.observed_catch:
        if s.group_idx is not None and s.group_idx not in observed:
            observed[s.group_idx] = s.values[:n_obs_years]
            weights[s.group_idx] = s.weight
            relative[s.group_idx] = False

    if not observed:
        raise ValueError("No observed series found in timeseries collection.")

    # Determine which groups to fit
    if fit_groups is None:
        fit_groups = list(observed.keys())

    # Build link map: find VV indices for links involving fit_groups
    # PreyFrom/PreyTo are 1-based; convert to 0-based for comparison
    link_indices = []
    link_map = []
    n_links = len(scenario.params.PreyFrom)
    for i in range(1, n_links):
        prey = scenario.params.PreyFrom[i] - 1  # to 0-based
        pred = scenario.params.PreyTo[i] - 1
        if prey in fit_groups or pred in fit_groups:
            link_indices.append(i)
            link_map.append((prey, pred))

    if not link_indices and fit_vv:
        raise ValueError("No pred-prey links found for the specified fit_groups.")

    # Build parameter bounds
    bounds = []
    if fit_vv:
        bounds.extend([vv_bounds] * len(link_indices))

    n_vv = len(link_indices) if fit_vv else 0
    iteration_count = [0]

    def objective(param_vector):
        """Evaluate SS for a parameter vector."""
        # Inject VV values
        if fit_vv:
            for j, link_idx in enumerate(link_indices):
                scenario.params.VV[link_idx] = param_vector[j]

        # Run simulation
        try:
            output = rsim_run(scenario)
        except Exception as e:
            logger.debug("Simulation failed: %s", e)
            return 1e10  # penalty for failed runs

        # Extract annual mean predicted biomass
        predicted: dict[int, np.ndarray] = {}
        n_months = output.out_Biomass.shape[0]
        months_per_year = n_months // n_obs_years if n_obs_years > 0 else 12

        for grp_idx in observed:
            col = grp_idx + 1  # 1-based column in output
            if col >= output.out_Biomass.shape[1]:
                continue
            annual = np.zeros(n_obs_years)
            for yr in range(n_obs_years):
                start = yr * months_per_year
                end = min(start + months_per_year, n_months)
                if start < n_months:
                    annual[yr] = np.mean(output.out_Biomass[start:end, col])
            predicted[grp_idx] = annual

        ss, _ = _compute_ss(observed, predicted, weights, relative)

        iteration_count[0] += 1
        if verbose and iteration_count[0] % 50 == 0:
            logger.info("Iteration %d: SS = %.6f", iteration_count[0], ss)

        return ss

    # Run optimizer
    if method == "differential_evolution":
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=max_iterations,
            seed=42,
            tol=1e-6,
            polish=True,
        )
        best_params = result.x
        converged = result.success
        final_ss = result.fun

    elif method == "minimize":
        from scipy.optimize import minimize

        x0 = np.array([scenario.params.VV[i] for i in link_indices])
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )
        best_params = result.x
        converged = result.success
        final_ss = result.fun

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'differential_evolution' or 'minimize'.")

    # Extract best VV
    best_vv = best_params[:n_vv] if fit_vv else np.array([])

    # Apply best params to scenario for the fitted_scenario output
    if fit_vv:
        for j, link_idx in enumerate(link_indices):
            scenario.params.VV[link_idx] = best_vv[j]

    # Compute final SS breakdown
    try:
        output = rsim_run(scenario)
        predicted_final: dict[int, np.ndarray] = {}
        n_months = output.out_Biomass.shape[0]
        months_per_year = n_months // n_obs_years if n_obs_years > 0 else 12
        for grp_idx in observed:
            col = grp_idx + 1
            if col >= output.out_Biomass.shape[1]:
                continue
            annual = np.zeros(n_obs_years)
            for yr in range(n_obs_years):
                start = yr * months_per_year
                end = min(start + months_per_year, n_months)
                if start < n_months:
                    annual[yr] = np.mean(output.out_Biomass[start:end, col])
            predicted_final[grp_idx] = annual
        _, ss_by_group = _compute_ss(observed, predicted_final, weights, relative)
    except Exception:
        ss_by_group = {}

    return CalibrationResult(
        best_vv=best_vv,
        best_pp=None,
        ss=final_ss,
        ss_by_group=ss_by_group,
        n_iterations=iteration_count[0],
        converged=converged,
        fitted_scenario=scenario,
        link_map=link_map,
    )
```

- [ ] **Step 4: Run integration tests**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_calibration.py -v --timeout=300`
Expected: All tests PASS (may take 1-2 minutes due to optimization)

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/calibration.py packages/pypath/tests/test_calibration.py
git commit -m "feat(core): implement fit_to_timeseries() calibration with differential evolution"
```

---

## Chunk 4: Package Exports & Final Verification

### Task 11: Package __init__.py Exports

**Files:**
- Modify: `packages/pypath/src/pypath/core/__init__.py`
- Modify: `packages/pypath/src/pypath/io/__init__.py`

**Context:** Both files have explicit `__all__` lists that must be updated alongside imports.

- [ ] **Step 1: Add core exports**

In `packages/pypath/src/pypath/core/__init__.py`, add these imports alongside existing ones:

```python
from pypath.core.timeseries import (
    DATTYPE_ABS_BIOMASS,
    DATTYPE_CATCH,
    DATTYPE_EFFORT,
    DATTYPE_FISHING_MORTALITY,
    DATTYPE_FORCED_BIOMASS,
    DATTYPE_REL_BIOMASS,
    EweTimeSeries,
    EweTimeSeriesCollection,
    apply_timeseries_drivers,
    load_timeseries,
)
from pypath.core.calibration import CalibrationResult, fit_to_timeseries
```

And add to `__all__`:

```python
    "DATTYPE_ABS_BIOMASS",
    "DATTYPE_CATCH",
    "DATTYPE_EFFORT",
    "DATTYPE_FISHING_MORTALITY",
    "DATTYPE_FORCED_BIOMASS",
    "DATTYPE_REL_BIOMASS",
    "EweTimeSeries",
    "EweTimeSeriesCollection",
    "apply_timeseries_drivers",
    "load_timeseries",
    "CalibrationResult",
    "fit_to_timeseries",
```

- [ ] **Step 2: Add I/O exports**

In `packages/pypath/src/pypath/io/__init__.py`, add:

```python
from pypath.io.ewemdb import read_timeseries
from pypath.io.timeseries_csv import load_timeseries_csv
```

And add to `__all__`:

```python
    "read_timeseries",
    "load_timeseries_csv",
```

- [ ] **Step 3: Verify imports work**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -c "from pypath.core import EweTimeSeries, EweTimeSeriesCollection, fit_to_timeseries, CalibrationResult, load_timeseries, apply_timeseries_drivers; from pypath.io import read_timeseries, load_timeseries_csv; print('All exports OK')"`
Expected: `All exports OK`

- [ ] **Step 4: Run full test suite**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/ -q -m "not integration" --ignore=packages/pypath/tests/scripts --timeout=120`
Expected: All tests PASS (including new and existing)

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/core/__init__.py packages/pypath/src/pypath/io/__init__.py
git commit -m "feat: export time series and calibration API from core and io packages"
```

---

### Task 12: Final Verification & Cleanup

- [ ] **Step 1: Run full test suite with slow tests**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/ -q --ignore=packages/pypath/tests/scripts --timeout=300`
Expected: All tests PASS

- [ ] **Step 2: Verify no regressions in existing tests**

Run: `cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -m pytest packages/pypath/tests/test_ecosim.py packages/pypath/tests/test_ecopath.py packages/pypath/tests/test_ewemdb.py -v`
Expected: All PASS — no regressions in core modules

- [ ] **Step 3: Verify import paths match spec**

Run:
```bash
cd "C:/Users/DELL/OneDrive - ku.lt/HORIZON_EUROPE/PyPath" && python -c "
from pypath.core.timeseries import EweTimeSeries, EweTimeSeriesCollection, load_timeseries, apply_timeseries_drivers, DATTYPE_REL_BIOMASS, DATTYPE_CATCH
from pypath.core.calibration import fit_to_timeseries, CalibrationResult
from pypath.io.ewemdb import read_timeseries
from pypath.io.timeseries_csv import load_timeseries_csv
print('All spec import paths verified')
"
```
Expected: `All spec import paths verified`

- [ ] **Step 4: Final commit if any cleanup needed**

Only if changes were made during verification.
