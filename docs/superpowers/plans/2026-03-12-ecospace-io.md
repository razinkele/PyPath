# Ecospace I/O Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read Ecospace scenario configuration from EwE databases and construct `EcospaceParams` ready for `rsim_run_spatial()`.

**Architecture:** `read_ecospace()` in `io/ewemdb.py` reads multiple Ecospace tables, builds a fallback grid or maps onto a user-provided grid, populates group/habitat/fleet params, and returns `EcospaceReadResult` containing `EcospaceParams` plus metadata. `EcospaceGrid` gains optional `cell_metadata` for round-tripping.

**Tech Stack:** numpy, pandas, scipy.sparse, dataclasses. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-12-ecospace-io-design.md`

---

## Chunk 1: Schema, Grid Metadata, and Fallback Grid Builder

### Task 1: Add cell_metadata to EcospaceGrid

**Files:**
- Modify: `packages/pypath/src/pypath/spatial/ecospace_params.py`
- Create: `packages/pypath/tests/test_ecospace_io.py`

- [ ] **Step 1: Write failing test for cell_metadata**

Create `packages/pypath/tests/test_ecospace_io.py`:

```python
"""Tests for Ecospace I/O (read_ecospace + schema)."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from pypath.spatial.ecospace_params import EcospaceGrid


class TestEcospaceGridCellMetadata:
    def _make_grid(self, n=3, cell_metadata=None):
        """Helper to build a simple 1D grid."""
        adj = scipy.sparse.csr_matrix(np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]))
        return EcospaceGrid(
            n_patches=n,
            patch_ids=np.arange(n),
            patch_areas=np.ones(n),
            patch_centroids=np.column_stack([np.arange(n), np.zeros(n)]),
            adjacency_matrix=adj,
            edge_lengths={(0, 1): 1.0, (1, 2): 1.0},
            cell_metadata=cell_metadata,
        )

    def test_cell_metadata_default_none(self):
        grid = self._make_grid()
        assert grid.cell_metadata is None

    def test_cell_metadata_with_dataframe(self):
        meta = pd.DataFrame({
            "row": [0, 0, 1],
            "col": [0, 1, 0],
            "depth": [10.0, 20.0, 15.0],
            "habitat_type_id": [0, 0, 1],
        })
        grid = self._make_grid(cell_metadata=meta)
        assert grid.cell_metadata is not None
        assert len(grid.cell_metadata) == 3
        assert "habitat_type_id" in grid.cell_metadata.columns

    def test_validation_still_works_with_metadata(self):
        meta = pd.DataFrame({"row": [0], "col": [0]})
        # Should raise because n_patches=3 but only 1-patch grid data
        with pytest.raises(ValueError):
            EcospaceGrid(
                n_patches=3,
                patch_ids=np.array([0]),  # wrong length
                patch_areas=np.ones(3),
                patch_centroids=np.zeros((3, 2)),
                adjacency_matrix=scipy.sparse.csr_matrix((3, 3)),
                edge_lengths={},
                cell_metadata=meta,
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io.py::TestEcospaceGridCellMetadata -v`
Expected: FAIL (no `cell_metadata` parameter)

- [ ] **Step 3: Add cell_metadata field to EcospaceGrid**

In `packages/pypath/src/pypath/spatial/ecospace_params.py`, add the field after `geometry`:

```python
    geometry: Optional[object] = None  # gpd.GeoDataFrame when available
    cell_metadata: Optional["pd.DataFrame"] = None  # row/col/depth/habitat_type_id per patch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io.py::TestEcospaceGridCellMetadata -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/spatial/ecospace_params.py packages/pypath/tests/test_ecospace_io.py
git commit -m "feat(spatial): add cell_metadata field to EcospaceGrid"
```

---

### Task 2: Update EwE schema tables

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py`
- Modify: `packages/pypath/tests/test_ecospace_io.py`

- [ ] **Step 1: Write schema tests**

Append to `packages/pypath/tests/test_ecospace_io.py`:

```python
class TestEcospaceSchema:
    def test_ecospace_scenario_has_grid_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenario"]
        assert tbl["Inrow"] == "INTEGER"
        assert tbl["Incol"] == "INTEGER"
        assert tbl["CellLength"] == "DOUBLE"
        assert tbl["MinLon"] == "DOUBLE"
        assert tbl["MinLat"] == "DOUBLE"
        # New columns
        assert tbl["PredictEffort"] == "YESNO"
        assert tbl["IFDPower"] == "DOUBLE"
        assert tbl["DepthMap"] == "LONGBINARY"

    def test_ecospace_scenario_group_has_all_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioGroup"]
        assert tbl["Mvel"] == "DOUBLE"
        assert tbl["IsAdvected"] == "YESNO"
        assert tbl["BarrierAvoidanceWeight"] == "DOUBLE"
        # New columns
        assert tbl["CapacityMap"] == "LONGBINARY"
        assert tbl["CapacityCalType"] == "INTEGER"
        assert tbl["KMoveFit"] == "DOUBLE"

    def test_ecospace_scenario_habitat_has_map(self):
        from pypath.io._ewe_schema import EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioHabitat"]
        assert tbl["HabitatName"] == "TEXT"
        assert tbl["HabitatMap"] == "LONGBINARY"

    def test_ecospace_fleet_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcospaceScenarioFleet" in EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioFleet"]
        assert tbl["FleetID"] == "INTEGER"
        assert tbl["EcopathFleetID"] == "INTEGER"
        assert tbl["EffPower"] == "DOUBLE"
        assert tbl["SEMult"] == "DOUBLE"
        assert tbl["PortMap"] == "LONGBINARY"

    def test_ecospace_group_habitat_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcospaceScenarioGroupHabitat" in EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioGroupHabitat"]
        assert tbl["GroupID"] == "INTEGER"
        assert tbl["HabitatID"] == "INTEGER"
        assert tbl["Preference"] == "DOUBLE"

    def test_ecospace_capacity_drivers_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcospaceScenarioCapacityDrivers" in EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioCapacityDrivers"]
        assert tbl["GroupID"] == "INTEGER"
        assert tbl["VarDBID"] == "INTEGER"
        assert tbl["ShapeID"] == "INTEGER"

    def test_ecospace_driver_layer_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES
        assert "EcospaceScenarioDriverLayer" in EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioDriverLayer"]
        assert tbl["LayerName"] == "TEXT"
        assert tbl["LayerMAP"] == "LONGBINARY"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io.py::TestEcospaceSchema -v`
Expected: FAIL (missing tables/columns)

- [ ] **Step 3: Update existing schema tables and add new ones**

Read `packages/pypath/src/pypath/io/_ewe_schema.py`. Find the Ecospace tables section (line 259).

Update `EcospaceScenario` (line 261) to include all columns from the real DB:

Replace the existing `EcospaceScenario` OrderedDict with:
```python
    "EcospaceScenario": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("ScenarioName", "TEXT"),
            ("Description", "TEXT"),
            ("Author", "TEXT"),
            ("Contact", "TEXT"),
            ("LastSaved", "TEXT"),
            ("EcosimScenarioID", "INTEGER"),
            ("Inrow", "INTEGER"),
            ("Incol", "INTEGER"),
            ("CellLength", "DOUBLE"),
            ("TimeStep", "DOUBLE"),
            ("PredictEffort", "YESNO"),
            ("IFDPower", "DOUBLE"),
            ("TotalTime", "DOUBLE"),
            ("ModelType", "INTEGER"),
            ("NumThreads", "INTEGER"),
            ("NumPacketsMultiplier", "DOUBLE"),
            ("AdjustSpace", "YESNO"),
            ("UseExact", "YESNO"),
            ("Tolerance", "DOUBLE"),
            ("MinLon", "DOUBLE"),
            ("MinLat", "DOUBLE"),
            ("DepthMap", "LONGBINARY"),
            ("RelPPMap", "LONGBINARY"),
            ("RelCinMap", "LONGBINARY"),
            ("DepthAMap", "LONGBINARY"),
            ("LastSavedVersion", "TEXT"),
            ("NumRegions", "INTEGER"),
            ("RegionMap", "LONGBINARY"),
            ("CellSize", "DOUBLE"),
            ("UseEffortDistrThreshold", "YESNO"),
            ("EffortDistrThreshold", "DOUBLE"),
            ("ExclusionMap", "LONGBINARY"),
            ("AssumeSquareCells", "YESNO"),
            ("CoordinateSystemWKT", "TEXT"),
            ("FlowMap", "LONGBINARY"),
            ("FitResponseType", "INTEGER"),
            ("Q10DriverMap", "LONGBINARY"),
            ("UseSpinup", "YESNO"),
            ("SpinupYears", "INTEGER"),
            ("CellAreaMap", "LONGBINARY"),
            ("NumEffortZones", "INTEGER"),
            ("EffortZoneMap", "LONGBINARY"),
            ("UsePenaltySearch", "YESNO"),
            ("NoFishWeight", "DOUBLE"),
            ("PenaltyPower", "DOUBLE"),
            ("FirstPenaltyMonth", "INTEGER"),
        ]
    ),
```

Update `EcospaceScenarioGroup` (line 281) to add new columns:
```python
    "EcospaceScenarioGroup": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("EcopathGroupID", "INTEGER"),
            ("Mvel", "DOUBLE"),
            ("RelMoveBad", "DOUBLE"),
            ("RelVulBad", "DOUBLE"),
            ("IsAdvected", "YESNO"),
            ("IsMigratory", "YESNO"),
            ("BarrierAvoidanceWeight", "DOUBLE"),
            ("CapacityMap", "LONGBINARY"),
            ("CapacityCalType", "INTEGER"),
            ("InMigAreaMovement", "DOUBLE"),
            ("OtherMortMap", "LONGBINARY"),
            ("KMoveFit", "DOUBLE"),
            ("FTarget", "DOUBLE"),
        ]
    ),
```

Update `EcospaceScenarioHabitat` (line 294) to add HabitatMap:
```python
    "EcospaceScenarioHabitat": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("HabitatID", "INTEGER"),
            ("HabitatName", "TEXT"),
            ("Sequence", "INTEGER"),
            ("HabitatMap", "LONGBINARY"),
        ]
    ),
```

After the `EcospaceScenarioMPAPatch` table (line 325), add the 4 new tables:
```python
    "EcospaceScenarioFleet": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("FleetID", "INTEGER"),
            ("EcopathFleetID", "INTEGER"),
            ("EffPower", "DOUBLE"),
            ("PortMap", "LONGBINARY"),
            ("SailCostMap", "LONGBINARY"),
            ("SEMult", "DOUBLE"),
        ]
    ),
    "EcospaceScenarioGroupHabitat": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("HabitatID", "INTEGER"),
            ("Preference", "DOUBLE"),
        ]
    ),
    "EcospaceScenarioCapacityDrivers": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("GroupID", "INTEGER"),
            ("VarDBID", "INTEGER"),
            ("ShapeID", "INTEGER"),
            ("Target", "INTEGER"),
        ]
    ),
    "EcospaceScenarioDriverLayer": OrderedDict(
        [
            ("ScenarioID", "INTEGER"),
            ("LayerID", "INTEGER"),
            ("Sequence", "INTEGER"),
            ("LayerName", "TEXT"),
            ("LayerDescription", "TEXT"),
            ("LayerMAP", "LONGBINARY"),
            ("LayerUnits", "TEXT"),
        ]
    ),
```

- [ ] **Step 4: Run schema tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io.py::TestEcospaceSchema -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py packages/pypath/tests/test_ecospace_io.py
git commit -m "feat(io): update Ecospace schema with full column set and add new tables"
```

---

### Task 3: Fallback grid builder helper

**Files:**
- Modify: `packages/pypath/tests/test_ecospace_io.py`
- Modify: `packages/pypath/src/pypath/io/ewemdb.py`

- [ ] **Step 1: Write fallback grid tests**

Append to `packages/pypath/tests/test_ecospace_io.py`:

```python
class TestBuildFallbackGrid:
    def test_builds_correct_patch_count(self):
        from pypath.io.ewemdb import _build_fallback_grid
        grid = _build_fallback_grid(n_rows=3, n_cols=4, cell_length=10.0)
        assert grid.n_patches == 12

    def test_cell_areas(self):
        from pypath.io.ewemdb import _build_fallback_grid
        grid = _build_fallback_grid(n_rows=2, n_cols=2, cell_length=5.0)
        np.testing.assert_array_equal(grid.patch_areas, 25.0)

    def test_rook_adjacency(self):
        from pypath.io.ewemdb import _build_fallback_grid
        # 2x3 grid:
        #  0  1  2
        #  3  4  5
        grid = _build_fallback_grid(n_rows=2, n_cols=3, cell_length=1.0)
        adj = grid.adjacency_matrix.toarray()
        # Patch 0: neighbors 1 (right), 3 (below)
        assert adj[0, 1] == 1
        assert adj[0, 3] == 1
        assert adj[0, 2] == 0  # not diagonal
        # Patch 4 (center): neighbors 1, 3, 5
        assert adj[4, 1] == 1
        assert adj[4, 3] == 1
        assert adj[4, 5] == 1
        assert adj[4, 0] == 0  # not diagonal

    def test_centroids_with_origin(self):
        from pypath.io.ewemdb import _build_fallback_grid
        grid = _build_fallback_grid(
            n_rows=2, n_cols=2, cell_length=10.0,
            min_lon=20.0, min_lat=55.0,
        )
        # Patch 0 at row=0,col=0 -> centroid at (20+5, 55+5) = (25, 60)
        # (lon = min_lon + col*cell + cell/2, lat = min_lat + row*cell + cell/2)
        assert grid.patch_centroids[0, 0] == pytest.approx(25.0)
        assert grid.patch_centroids[0, 1] == pytest.approx(60.0)

    def test_cell_metadata_populated(self):
        from pypath.io.ewemdb import _build_fallback_grid
        grid = _build_fallback_grid(n_rows=2, n_cols=3, cell_length=1.0)
        assert grid.cell_metadata is not None
        assert len(grid.cell_metadata) == 6
        assert "row" in grid.cell_metadata.columns
        assert "col" in grid.cell_metadata.columns

    def test_single_cell_grid(self):
        from pypath.io.ewemdb import _build_fallback_grid
        grid = _build_fallback_grid(n_rows=1, n_cols=1, cell_length=1.0)
        assert grid.n_patches == 1
        assert grid.adjacency_matrix.nnz == 0  # no neighbors

    def test_edge_lengths_equal_cell_length(self):
        from pypath.io.ewemdb import _build_fallback_grid
        grid = _build_fallback_grid(n_rows=2, n_cols=2, cell_length=7.5)
        # All edges should have length = cell_length
        for edge_len in grid.edge_lengths.values():
            assert edge_len == pytest.approx(7.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io.py::TestBuildFallbackGrid -v`
Expected: FAIL (no `_build_fallback_grid`)

- [ ] **Step 3: Implement _build_fallback_grid**

Add to `packages/pypath/src/pypath/io/ewemdb.py`, before `read_mpa_config` (find a good spot near the end of the file):

```python
def _build_fallback_grid(
    n_rows: int,
    n_cols: int,
    cell_length: float,
    min_lon: float = 0.0,
    min_lat: float = 0.0,
) -> "EcospaceGrid":
    """Build a regular raster grid from EwE scenario dimensions.

    Creates square cells in a row-major layout. All cells are treated as
    water (no land exclusion). Adjacency uses rook neighborhood (shared edges,
    no diagonals).

    Parameters
    ----------
    n_rows : int
        Number of grid rows.
    n_cols : int
        Number of grid columns.
    cell_length : float
        Cell edge length in km.
    min_lon : float
        Longitude of grid origin (lower-left corner).
    min_lat : float
        Latitude of grid origin (lower-left corner).

    Returns
    -------
    EcospaceGrid
        Grid with n_rows * n_cols patches.
    """
    import scipy.sparse
    from pypath.spatial.ecospace_params import EcospaceGrid

    n_patches = n_rows * n_cols
    cell_area = cell_length ** 2

    # Patch IDs, areas
    patch_ids = np.arange(n_patches)
    patch_areas = np.full(n_patches, cell_area)

    # Centroids: row-major layout
    # patch_idx = row * n_cols + col
    rows_arr = np.arange(n_patches) // n_cols
    cols_arr = np.arange(n_patches) % n_cols
    lon = min_lon + (cols_arr + 0.5) * cell_length
    lat = min_lat + (rows_arr + 0.5) * cell_length
    centroids = np.column_stack([lon, lat])

    # Rook adjacency and edge lengths
    row_idx = []
    col_idx = []
    edge_lengths = {}
    for p in range(n_patches):
        r, c = divmod(p, n_cols)
        # Right neighbor
        if c + 1 < n_cols:
            q = r * n_cols + (c + 1)
            row_idx.extend([p, q])
            col_idx.extend([q, p])
            edge_lengths[(min(p, q), max(p, q))] = cell_length
        # Below neighbor
        if r + 1 < n_rows:
            q = (r + 1) * n_cols + c
            row_idx.extend([p, q])
            col_idx.extend([q, p])
            edge_lengths[(min(p, q), max(p, q))] = cell_length

    data = np.ones(len(row_idx), dtype=int)
    adjacency = scipy.sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(n_patches, n_patches)
    )

    # Cell metadata for round-tripping
    meta = pd.DataFrame({
        "row": rows_arr,
        "col": cols_arr,
        "depth": np.zeros(n_patches),
        "habitat_type_id": np.zeros(n_patches, dtype=int),
    })

    return EcospaceGrid(
        n_patches=n_patches,
        patch_ids=patch_ids,
        patch_areas=patch_areas,
        patch_centroids=centroids,
        adjacency_matrix=adjacency,
        edge_lengths=edge_lengths,
        cell_metadata=meta,
    )
```

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io.py::TestBuildFallbackGrid -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py packages/pypath/tests/test_ecospace_io.py
git commit -m "feat(io): add fallback grid builder for Ecospace raster grids"
```

---

## Chunk 2: read_ecospace() and Tests

### Task 4: EcospaceReadResult dataclass and read_ecospace()

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py`
- Modify: `packages/pypath/tests/test_ecospace_io.py`

- [ ] **Step 1: Write read_ecospace unit tests**

Append to `packages/pypath/tests/test_ecospace_io.py`:

```python
from unittest.mock import patch


def _mock_scenario_df(n_rows=3, n_cols=3, cell_length=10.0):
    """Create a mock EcospaceScenario DataFrame."""
    return pd.DataFrame([{
        "ScenarioID": 1,
        "ScenarioName": "Test",
        "Description": "Test scenario",
        "Inrow": n_rows,
        "Incol": n_cols,
        "CellLength": cell_length,
        "CellSize": cell_length ** 2,
        "MinLon": 20.0,
        "MinLat": 55.0,
        "TotalTime": 10.0,
        "TimeStep": 1.0,
    }])


def _mock_group_df(n_groups=2):
    """Create a mock EcospaceScenarioGroup DataFrame."""
    rows = []
    for i in range(n_groups):
        rows.append({
            "ScenarioID": 1,
            "GroupID": i + 1,
            "EcopathGroupID": i + 1,
            "Mvel": 0.5 * (i + 1),
            "RelMoveBad": 0.5,
            "RelVulBad": 0.5,
            "IsAdvected": True if i == 0 else False,
            "IsMigratory": False,
            "BarrierAvoidanceWeight": 0.0,
        })
    return pd.DataFrame(rows)


def _mock_habitat_df():
    """Create a mock EcospaceScenarioHabitat DataFrame."""
    return pd.DataFrame([
        {"ScenarioID": 1, "HabitatID": 1, "HabitatName": "Rocky", "Sequence": 1},
        {"ScenarioID": 1, "HabitatID": 2, "HabitatName": "Sandy", "Sequence": 2},
    ])


def _mock_group_habitat_df(n_groups=2):
    """Create mock group-habitat preferences."""
    rows = []
    for g in range(1, n_groups + 1):
        rows.append({"ScenarioID": 1, "GroupID": g, "HabitatID": 1, "Preference": 0.8})
        rows.append({"ScenarioID": 1, "GroupID": g, "HabitatID": 2, "Preference": 0.3})
    return pd.DataFrame(rows)


def _mock_fleet_df():
    """Create mock fleet spatial params."""
    return pd.DataFrame([
        {"ScenarioID": 1, "FleetID": 1, "EcopathFleetID": 1,
         "EffPower": 1.0, "SEMult": 1.0},
    ])


def _mock_capacity_df():
    """Create mock capacity driver assignments."""
    return pd.DataFrame([
        {"ScenarioID": 1, "GroupID": 1, "VarDBID": 1, "ShapeID": 1, "Target": 0},
    ])


class TestReadEcospace:
    def _read_with_mocks(self, table_map, n_groups=2, grid=None):
        from pypath.io.ewemdb import read_ecospace
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                return read_ecospace(
                    "fake.eweaccdb", n_groups=n_groups, grid=grid,
                )

    def test_builds_fallback_grid(self):
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=3),
            "EcospaceScenarioGroup": _mock_group_df(2),
        }
        result = self._read_with_mocks(table_map, n_groups=2)
        assert result.ecospace.grid.n_patches == 6
        assert result.ecospace.habitat_preference.shape == (2, 6)
        assert result.ecospace.habitat_capacity.shape == (2, 6)
        np.testing.assert_array_equal(result.ecospace.habitat_capacity, 1.0)

    def test_group_params_mapped(self):
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=2),
            "EcospaceScenarioGroup": _mock_group_df(2),
        }
        result = self._read_with_mocks(table_map, n_groups=2)
        # Mvel -> dispersal_rate: group 0 = 0.5, group 1 = 1.0
        assert result.ecospace.dispersal_rate[0] == pytest.approx(0.5)
        assert result.ecospace.dispersal_rate[1] == pytest.approx(1.0)
        # IsAdvected: group 0 = True, group 1 = False
        assert result.ecospace.advection_enabled[0] == True
        assert result.ecospace.advection_enabled[1] == False
        # gravity_strength defaults to 0.0
        np.testing.assert_array_equal(result.ecospace.gravity_strength, 0.0)

    def test_habitat_preference_from_group_habitat(self):
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=2),
            "EcospaceScenarioGroup": _mock_group_df(2),
            "EcospaceScenarioHabitat": _mock_habitat_df(),
            "EcospaceScenarioGroupHabitat": _mock_group_habitat_df(2),
        }
        result = self._read_with_mocks(table_map, n_groups=2)
        # Without habitat map, all patches get default habitat type (first = 0)
        # So all patches should have preference for habitat 0 = 0.8
        assert result.ecospace.habitat_preference[0, 0] == pytest.approx(0.8)
        assert result.ecospace.habitat_preference[1, 0] == pytest.approx(0.8)

    def test_uses_provided_grid(self):
        from pypath.spatial import create_1d_grid
        user_grid = create_1d_grid(n_patches=5, spacing=1.0)

        table_map = {
            "EcospaceScenario": _mock_scenario_df(),
            "EcospaceScenarioGroup": _mock_group_df(2),
        }
        result = self._read_with_mocks(table_map, n_groups=2, grid=user_grid)
        assert result.ecospace.grid.n_patches == 5
        assert result.ecospace.habitat_preference.shape == (2, 5)

    def test_missing_optional_tables_defaults(self):
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=2),
        }
        result = self._read_with_mocks(table_map, n_groups=2)
        # No group table -> zero dispersal, no advection
        np.testing.assert_array_equal(result.ecospace.dispersal_rate, 0.0)
        np.testing.assert_array_equal(result.ecospace.advection_enabled, False)
        # No habitat tables -> uniform preference
        np.testing.assert_array_equal(result.ecospace.habitat_preference, 1.0)
        # No fleet/capacity tables -> None
        assert result.fleet_info is None
        assert result.capacity_drivers is None

    def test_missing_ecospace_scenario_raises(self):
        from pypath.io.ewemdb import read_ecospace, EwEDatabaseError
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=["SomeOtherTable"]):
            with pytest.raises(EwEDatabaseError):
                read_ecospace("fake.eweaccdb", n_groups=2)

    def test_scenario_metadata_populated(self):
        table_map = {
            "EcospaceScenario": _mock_scenario_df(),
        }
        result = self._read_with_mocks(table_map, n_groups=2)
        assert result.scenario_meta["ScenarioName"] == "Test"
        assert result.scenario_meta["Description"] == "Test scenario"
        assert result.scenario_meta["Inrow"] == 3
        assert result.scenario_meta["Incol"] == 3

    def test_fleet_info_populated(self):
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=2),
            "EcospaceScenarioFleet": _mock_fleet_df(),
        }
        result = self._read_with_mocks(table_map, n_groups=2)
        assert result.fleet_info is not None
        assert len(result.fleet_info) == 1
        assert result.fleet_info.iloc[0]["EffPower"] == 1.0

    def test_capacity_drivers_populated(self):
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=2),
            "EcospaceScenarioCapacityDrivers": _mock_capacity_df(),
        }
        result = self._read_with_mocks(table_map, n_groups=2)
        assert result.capacity_drivers is not None
        assert len(result.capacity_drivers) == 1

    def test_habitat_types_dict(self):
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=2),
            "EcospaceScenarioHabitat": _mock_habitat_df(),
        }
        result = self._read_with_mocks(table_map, n_groups=2)
        # 1-based -> 0-based: {0: "Rocky", 1: "Sandy"}
        assert result.habitat_types[0] == "Rocky"
        assert result.habitat_types[1] == "Sandy"

    def test_index_conversion_1based_to_0based(self):
        group_df = pd.DataFrame([{
            "ScenarioID": 1, "GroupID": 3, "EcopathGroupID": 3,
            "Mvel": 2.5, "RelMoveBad": 0.5, "RelVulBad": 0.5,
            "IsAdvected": False, "IsMigratory": False,
            "BarrierAvoidanceWeight": 0.0,
        }])
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=2),
            "EcospaceScenarioGroup": group_df,
        }
        # n_groups=3, GroupID=3 -> index 2
        result = self._read_with_mocks(table_map, n_groups=3)
        assert result.ecospace.dispersal_rate[2] == pytest.approx(2.5)
        assert result.ecospace.dispersal_rate[0] == 0.0  # not in table -> default

    def test_yesno_boolean_conversion(self):
        """YESNO columns from Access DB may arrive as str, int, or bool."""
        for adv_val, expected in [("Yes", True), ("yes", True), (1, True),
                                   ("No", False), (0, False), (False, False)]:
            group_df = pd.DataFrame([{
                "ScenarioID": 1, "GroupID": 1, "EcopathGroupID": 1,
                "Mvel": 1.0, "RelMoveBad": 0.5, "RelVulBad": 0.5,
                "IsAdvected": adv_val, "IsMigratory": False,
                "BarrierAvoidanceWeight": 0.0,
            }])
            table_map = {
                "EcospaceScenario": _mock_scenario_df(n_rows=1, n_cols=1),
                "EcospaceScenarioGroup": group_df,
            }
            result = self._read_with_mocks(table_map, n_groups=1)
            assert result.ecospace.advection_enabled[0] == expected, (
                f"IsAdvected={adv_val!r} should map to {expected}"
            )

    def test_group_beyond_n_groups_ignored(self):
        """Groups with IDs beyond n_groups are skipped."""
        group_df = pd.DataFrame([
            {"ScenarioID": 1, "GroupID": 1, "EcopathGroupID": 1,
             "Mvel": 1.0, "RelMoveBad": 0.5, "RelVulBad": 0.5,
             "IsAdvected": False, "IsMigratory": False,
             "BarrierAvoidanceWeight": 0.0},
            {"ScenarioID": 1, "GroupID": 99, "EcopathGroupID": 99,
             "Mvel": 5.0, "RelMoveBad": 0.5, "RelVulBad": 0.5,
             "IsAdvected": False, "IsMigratory": False,
             "BarrierAvoidanceWeight": 0.0},
        ])
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=1, n_cols=1),
            "EcospaceScenarioGroup": group_df,
        }
        result = self._read_with_mocks(table_map, n_groups=2)
        assert result.ecospace.dispersal_rate[0] == pytest.approx(1.0)
        assert result.ecospace.dispersal_rate[1] == 0.0  # default, GroupID 99 skipped

    def test_zero_groups_handled(self):
        """n_groups=0 should not crash."""
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=1, n_cols=1),
        }
        result = self._read_with_mocks(table_map, n_groups=0)
        assert result.ecospace.habitat_preference.shape == (0, 1)
        assert result.ecospace.dispersal_rate.shape == (0,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io.py::TestReadEcospace -v`
Expected: FAIL (no `read_ecospace`)

- [ ] **Step 3: Implement EcospaceReadResult and read_ecospace()**

Add to `packages/pypath/src/pypath/io/ewemdb.py`:

```python
@dataclass
class EcospaceReadResult:
    """Result of reading Ecospace configuration from an EwE database.

    Attributes
    ----------
    ecospace : EcospaceParams
        Spatial parameters ready for rsim_run_spatial().
    habitat_types : dict
        Mapping of 0-based habitat ID to habitat name.
    fleet_info : pd.DataFrame or None
        Fleet spatial params (EffPower, SEMult, etc.).
    capacity_drivers : pd.DataFrame or None
        Capacity driver assignments.
    scenario_meta : dict
        Scenario-level metadata (name, description, grid dims, etc.).
    """

    ecospace: "EcospaceParams"
    habitat_types: dict
    fleet_info: "Optional[pd.DataFrame]"
    capacity_drivers: "Optional[pd.DataFrame]"
    scenario_meta: dict


def read_ecospace(
    db_path: str,
    n_groups: int,
    scenario_id: int = 1,
    grid: "Optional[EcospaceGrid]" = None,
) -> EcospaceReadResult:
    """Read Ecospace configuration from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.
    n_groups : int
        Number of living + dead groups (from Ecopath model).
    scenario_id : int
        Scenario ID to filter by (default 1).
    grid : EcospaceGrid, optional
        User-provided spatial grid. If None, a regular grid is constructed
        from Inrow/Incol/CellLength in EcospaceScenario.

    Returns
    -------
    EcospaceReadResult

    Raises
    ------
    EwEDatabaseError
        If EcospaceScenario table is missing.
    """
    from pypath.spatial.ecospace_params import EcospaceParams

    tables = list_ewemdb_tables(db_path)

    # 1. Read EcospaceScenario (required)
    if "EcospaceScenario" not in tables:
        raise EwEDatabaseError("EcospaceScenario table not found in database")

    scenario_df = read_ewemdb_table(db_path, "EcospaceScenario")
    scenario_df = scenario_df[scenario_df["ScenarioID"] == scenario_id]
    if len(scenario_df) == 0:
        raise EwEDatabaseError(
            f"No EcospaceScenario with ScenarioID={scenario_id}"
        )
    scenario_row = scenario_df.iloc[0]

    scenario_meta = {}
    for col in ["ScenarioName", "Description", "Inrow", "Incol",
                 "CellLength", "CellSize", "MinLon", "MinLat",
                 "TotalTime", "TimeStep"]:
        if col in scenario_row.index:
            scenario_meta[col] = scenario_row[col]

    # 2. Grid construction
    if grid is None:
        n_rows = int(scenario_row.get("Inrow", 1))
        n_cols = int(scenario_row.get("Incol", 1))
        cell_length = float(scenario_row.get("CellLength", 1.0))
        min_lon = float(scenario_row.get("MinLon", 0.0))
        min_lat = float(scenario_row.get("MinLat", 0.0))
        logger.warning(
            "No grid provided; building fallback %dx%d grid. "
            "Land/water distinction not available without basemap.",
            n_rows, n_cols,
        )
        grid = _build_fallback_grid(n_rows, n_cols, cell_length, min_lon, min_lat)

    n_patches = grid.n_patches

    # 3. Read habitat types
    habitat_types = {}  # 0-based ID -> name
    if "EcospaceScenarioHabitat" in tables:
        try:
            hab_df = read_ewemdb_table(db_path, "EcospaceScenarioHabitat")
            hab_df = hab_df[hab_df["ScenarioID"] == scenario_id]
            for _, row in hab_df.iterrows():
                hid = int(row["HabitatID"]) - 1  # 1-based -> 0-based
                name = str(row.get("HabitatName", f"Habitat{hid}"))
                habitat_types[hid] = name
        except Exception as e:
            logger.warning("Failed to read EcospaceScenarioHabitat: %s", e)

    # 4. Build habitat_preference [n_groups, n_patches]
    habitat_preference = np.ones((n_groups, n_patches))
    if "EcospaceScenarioGroupHabitat" in tables and habitat_types:
        try:
            gh_df = read_ewemdb_table(db_path, "EcospaceScenarioGroupHabitat")
            gh_df = gh_df[gh_df["ScenarioID"] == scenario_id]

            # Build group -> {habitat_id -> preference} lookup
            group_hab_pref = {}
            for _, row in gh_df.iterrows():
                gid = int(row["GroupID"]) - 1  # 0-based
                hid = int(row["HabitatID"]) - 1  # 0-based
                pref = float(row.get("Preference", 1.0))
                if gid < n_groups:
                    group_hab_pref.setdefault(gid, {})[hid] = pref

            # Map preferences to patches via habitat type
            if grid.cell_metadata is not None and "habitat_type_id" in grid.cell_metadata.columns:
                patch_hab_types = grid.cell_metadata["habitat_type_id"].values
            else:
                # No basemap -> all patches get first habitat type (0)
                patch_hab_types = np.zeros(n_patches, dtype=int)

            for gid, hab_prefs in group_hab_pref.items():
                for p in range(n_patches):
                    hab_type = int(patch_hab_types[p])
                    if hab_type in hab_prefs:
                        habitat_preference[gid, p] = hab_prefs[hab_type]
        except Exception as e:
            logger.warning("Failed to read EcospaceScenarioGroupHabitat: %s", e)

    # habitat_capacity defaults to 1.0 (binary maps out of scope)
    habitat_capacity = np.ones((n_groups, n_patches))

    # 5. Read group spatial params
    dispersal_rate = np.zeros(n_groups)
    advection_enabled = np.zeros(n_groups, dtype=bool)
    gravity_strength = np.zeros(n_groups)

    if "EcospaceScenarioGroup" in tables:
        try:
            grp_df = read_ewemdb_table(db_path, "EcospaceScenarioGroup")
            grp_df = grp_df[grp_df["ScenarioID"] == scenario_id]
            for _, row in grp_df.iterrows():
                gid = int(row["GroupID"]) - 1  # 0-based
                if 0 <= gid < n_groups:
                    dispersal_rate[gid] = float(row.get("Mvel", 0.0))
                    is_adv = row.get("IsAdvected", False)
                    if isinstance(is_adv, str):
                        is_adv = is_adv.lower() in ("yes", "true", "1")
                    elif isinstance(is_adv, (int, float)):
                        is_adv = bool(is_adv)
                    advection_enabled[gid] = is_adv
                else:
                    logger.warning(
                        "EcospaceScenarioGroup GroupID=%d beyond n_groups=%d, skipped",
                        gid + 1, n_groups,
                    )
        except Exception as e:
            logger.warning("Failed to read EcospaceScenarioGroup: %s", e)

    # 6. Read fleet info
    fleet_info = None
    if "EcospaceScenarioFleet" in tables:
        try:
            fleet_df = read_ewemdb_table(db_path, "EcospaceScenarioFleet")
            fleet_df = fleet_df[fleet_df["ScenarioID"] == scenario_id]
            # Drop binary map columns for the returned DataFrame
            drop_cols = [c for c in fleet_df.columns if c.endswith("Map")]
            fleet_info = fleet_df.drop(columns=drop_cols, errors="ignore")
        except Exception as e:
            logger.warning("Failed to read EcospaceScenarioFleet: %s", e)

    # 7. Read capacity drivers
    capacity_drivers = None
    if "EcospaceScenarioCapacityDrivers" in tables:
        try:
            cap_df = read_ewemdb_table(db_path, "EcospaceScenarioCapacityDrivers")
            cap_df = cap_df[cap_df["ScenarioID"] == scenario_id]
            if len(cap_df) > 0:
                capacity_drivers = cap_df
        except Exception as e:
            logger.warning("Failed to read EcospaceScenarioCapacityDrivers: %s", e)

    # 8. Build EcospaceParams
    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=habitat_preference,
        habitat_capacity=habitat_capacity,
        dispersal_rate=dispersal_rate,
        advection_enabled=advection_enabled,
        gravity_strength=gravity_strength,
    )

    return EcospaceReadResult(
        ecospace=ecospace,
        habitat_types=habitat_types,
        fleet_info=fleet_info,
        capacity_drivers=capacity_drivers,
        scenario_meta=scenario_meta,
    )
```

Add `from dataclasses import dataclass` to the imports at the top of ewemdb.py if not already there.

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io.py::TestReadEcospace -v --tb=short`
Expected: All PASSED

- [ ] **Step 5: Run all tests so far**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py packages/pypath/tests/test_ecospace_io.py
git commit -m "feat(io): add read_ecospace() for reading Ecospace scenarios from EwE databases"
```

---

### Task 5: Integration test

**Files:**
- Create: `packages/pypath/tests/test_ecospace_io_integration.py`

- [ ] **Step 1: Write integration test**

Create `packages/pypath/tests/test_ecospace_io_integration.py`:

```python
"""Integration tests for Ecospace I/O with spatial Ecosim."""
import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import patch

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import rsim_run_spatial


def _make_model():
    """Create a balanced 3-group model."""
    params = create_rpath_params(
        groups=["Producer", "Consumer", "Det", "Fleet"],
        types=[1, 0, 2, 3],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 100.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 20.0
    params.model.loc[1, "QB"] = 60.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[2, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model["Detritus"] = 1.0
    params.model.loc[2, "Detritus"] = 0.0
    params.model.loc[3, "Detritus"] = 0.0
    params.diet["Consumer"] = [1.0, 0.0, 0.0, 0.0]
    params.model.loc[1, "Fleet"] = 0.5
    return params


@pytest.mark.slow
class TestEcospaceIOIntegration:
    def test_read_ecospace_runs_spatial_sim(self):
        """EcospaceParams from read_ecospace can run a spatial simulation."""
        from pypath.io.ewemdb import read_ecospace

        scenario_df = pd.DataFrame([{
            "ScenarioID": 1, "ScenarioName": "IntTest",
            "Description": "", "Inrow": 2, "Incol": 2,
            "CellLength": 10.0, "CellSize": 100.0,
            "MinLon": 0.0, "MinLat": 0.0,
            "TotalTime": 2.0, "TimeStep": 1.0,
        }])
        group_df = pd.DataFrame([
            {"ScenarioID": 1, "GroupID": 1, "EcopathGroupID": 1,
             "Mvel": 1.0, "RelMoveBad": 0.5, "RelVulBad": 0.5,
             "IsAdvected": False, "IsMigratory": False,
             "BarrierAvoidanceWeight": 0.0},
            {"ScenarioID": 1, "GroupID": 2, "EcopathGroupID": 2,
             "Mvel": 0.5, "RelMoveBad": 0.5, "RelVulBad": 0.5,
             "IsAdvected": False, "IsMigratory": False,
             "BarrierAvoidanceWeight": 0.0},
            {"ScenarioID": 1, "GroupID": 3, "EcopathGroupID": 3,
             "Mvel": 0.0, "RelMoveBad": 0.5, "RelVulBad": 0.5,
             "IsAdvected": False, "IsMigratory": False,
             "BarrierAvoidanceWeight": 0.0},
        ])
        table_map = {
            "EcospaceScenario": scenario_df,
            "EcospaceScenarioGroup": group_df,
        }

        # n_groups=3: Producer + Consumer + Det (living + dead, excludes Fleet)
        with patch("pypath.io.ewemdb.list_ewemdb_tables",
                    return_value=list(table_map.keys())):
            with patch("pypath.io.ewemdb.read_ewemdb_table",
                       side_effect=lambda path, tbl: table_map[tbl]):
                eco_result = read_ecospace("fake.eweaccdb", n_groups=3)

        # Build Ecopath/Ecosim model
        params = _make_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 3))

        # Run spatial sim with read_ecospace result
        result = rsim_run_spatial(scenario, ecospace=eco_result.ecospace)

        # Verify valid results: shape = (n_months+1, n_groups+1, n_patches)
        assert result.out_Biomass_spatial.shape[1] == 4  # 3 groups + 1 (1-based)
        assert result.out_Biomass_spatial.shape[2] == 4  # 2x2 grid
        assert np.all(np.isfinite(result.out_Biomass_spatial))
        assert np.all(result.out_Biomass_spatial >= 0)
```

- [ ] **Step 2: Run integration test**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io_integration.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/tests/test_ecospace_io_integration.py
git commit -m "test(io): add Ecospace I/O integration test with spatial Ecosim"
```

---

### Task 6: Package exports

**Files:**
- Modify: `packages/pypath/src/pypath/io/__init__.py`

- [ ] **Step 1: Add exports**

Read `packages/pypath/src/pypath/io/__init__.py`. Add `read_ecospace` and `EcospaceReadResult` to the ewemdb import block:

```python
from pypath.io.ewemdb import (
    ...
    read_mpa_config,
    read_ecospace,
    EcospaceReadResult,
    ...
)
```

Add to `__all__`:
```python
    "read_ecospace",
    "EcospaceReadResult",
```

- [ ] **Step 2: Verify imports**

Run: `conda run -n shiny python -c "from pypath.io import read_ecospace, EcospaceReadResult; print('OK')"`

- [ ] **Step 3: Run all new tests**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_ecospace_io.py packages/pypath/tests/test_ecospace_io_integration.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 4: Run existing tests for regression**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_spatial_ecosim_integration.py packages/pypath/tests/test_spatial_integration.py -v --tb=short`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/__init__.py
git commit -m "feat(api): export read_ecospace and EcospaceReadResult from io package"
```
