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


class TestEcospaceSchema:
    def test_ecospace_scenario_has_grid_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenario"]
        assert tbl["Inrow"] == "INTEGER"
        assert tbl["Incol"] == "INTEGER"
        assert tbl["CellLength"] == "DOUBLE"
        assert tbl["MinLon"] == "DOUBLE"
        assert tbl["MinLat"] == "DOUBLE"
        assert tbl["PredictEffort"] == "YESNO"
        assert tbl["IFDPower"] == "DOUBLE"
        assert tbl["DepthMap"] == "LONGBINARY"

    def test_ecospace_scenario_group_has_all_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioGroup"]
        assert tbl["Mvel"] == "DOUBLE"
        assert tbl["IsAdvected"] == "YESNO"
        assert tbl["BarrierAvoidanceWeight"] == "DOUBLE"
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
        for edge_len in grid.edge_lengths.values():
            assert edge_len == pytest.approx(7.5)
