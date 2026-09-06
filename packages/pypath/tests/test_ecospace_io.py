"""Tests for Ecospace I/O (read_ecospace + schema)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from pypath.spatial.ecospace_params import EcospaceGrid


def _mock_scenario_df(n_rows=3, n_cols=3, cell_length=10.0):
    return pd.DataFrame(
        [
            {
                "ScenarioID": 1,
                "ScenarioName": "Test",
                "Description": "Test scenario",
                "Inrow": n_rows,
                "Incol": n_cols,
                "CellLength": cell_length,
                "CellSize": cell_length**2,
                "MinLon": 20.0,
                "MinLat": 55.0,
                "TotalTime": 10.0,
                "TimeStep": 1.0,
            }
        ]
    )


def _mock_group_df(n_groups=2):
    rows = []
    for i in range(n_groups):
        rows.append(
            {
                "ScenarioID": 1,
                "GroupID": i + 1,
                "EcopathGroupID": i + 1,
                "Mvel": 0.5 * (i + 1),
                "RelMoveBad": 0.5,
                "RelVulBad": 0.5,
                "IsAdvected": True if i == 0 else False,
                "IsMigratory": False,
                "BarrierAvoidanceWeight": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _mock_habitat_df():
    return pd.DataFrame(
        [
            {"ScenarioID": 1, "HabitatID": 1, "HabitatName": "Rocky", "Sequence": 1},
            {"ScenarioID": 1, "HabitatID": 2, "HabitatName": "Sandy", "Sequence": 2},
        ]
    )


def _mock_group_habitat_df(n_groups=2):
    rows = []
    for g in range(1, n_groups + 1):
        rows.append({"ScenarioID": 1, "GroupID": g, "HabitatID": 1, "Preference": 0.8})
        rows.append({"ScenarioID": 1, "GroupID": g, "HabitatID": 2, "Preference": 0.3})
    return pd.DataFrame(rows)


def _mock_fleet_df():
    return pd.DataFrame(
        [
            {
                "ScenarioID": 1,
                "FleetID": 1,
                "EcopathFleetID": 1,
                "EffPower": 1.0,
                "SEMult": 1.0,
            },
        ]
    )


def _mock_capacity_df():
    return pd.DataFrame(
        [
            {"ScenarioID": 1, "GroupID": 1, "VarDBID": 1, "ShapeID": 1, "Target": 0},
        ]
    )


class TestEcospaceGridCellMetadata:
    def _make_grid(self, n=3, cell_metadata=None):
        """Helper to build a simple 1D grid."""
        adj = scipy.sparse.csr_matrix(
            np.array(
                [
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                ]
            )
        )
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
        meta = pd.DataFrame(
            {
                "row": [0, 0, 1],
                "col": [0, 1, 0],
                "depth": [10.0, 20.0, 15.0],
                "habitat_type_id": [0, 0, 1],
            }
        )
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
            n_rows=2,
            n_cols=2,
            cell_length=10.0,
            min_lon=20.0,
            min_lat=55.0,
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


class TestReadEcospace:
    def _read_with_mocks(self, table_map, n_groups=2, grid=None):
        from pypath.io.ewemdb import read_ecospace

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                return read_ecospace("fake.eweaccdb", n_groups=n_groups, grid=grid)

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
        assert result.ecospace.dispersal_rate[0] == pytest.approx(0.5)
        assert result.ecospace.dispersal_rate[1] == pytest.approx(1.0)
        assert result.ecospace.advection_enabled[0]
        assert not result.ecospace.advection_enabled[1]
        np.testing.assert_array_equal(result.ecospace.gravity_strength, 0.0)

    def test_habitat_preference_from_group_habitat(self):
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=2),
            "EcospaceScenarioGroup": _mock_group_df(2),
            "EcospaceScenarioHabitat": _mock_habitat_df(),
            "EcospaceScenarioGroupHabitat": _mock_group_habitat_df(2),
        }
        result = self._read_with_mocks(table_map, n_groups=2)
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
        table_map = {"EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=2)}
        result = self._read_with_mocks(table_map, n_groups=2)
        np.testing.assert_array_equal(result.ecospace.dispersal_rate, 0.0)
        np.testing.assert_array_equal(result.ecospace.advection_enabled, False)
        np.testing.assert_array_equal(result.ecospace.habitat_preference, 1.0)
        assert result.fleet_info is None
        assert result.capacity_drivers is None

    def test_missing_ecospace_scenario_raises(self):
        from pypath.io.ewemdb import EwEDatabaseError, read_ecospace

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=["SomeOtherTable"]
        ):
            with pytest.raises(EwEDatabaseError):
                read_ecospace("fake.eweaccdb", n_groups=2)

    def test_scenario_metadata_populated(self):
        table_map = {"EcospaceScenario": _mock_scenario_df()}
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
        assert result.habitat_types[0] == "Rocky"
        assert result.habitat_types[1] == "Sandy"

    def test_index_conversion_1based_to_0based(self):
        group_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "GroupID": 3,
                    "EcopathGroupID": 3,
                    "Mvel": 2.5,
                    "RelMoveBad": 0.5,
                    "RelVulBad": 0.5,
                    "IsAdvected": False,
                    "IsMigratory": False,
                    "BarrierAvoidanceWeight": 0.0,
                }
            ]
        )
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=2, n_cols=2),
            "EcospaceScenarioGroup": group_df,
        }
        result = self._read_with_mocks(table_map, n_groups=3)
        assert result.ecospace.dispersal_rate[2] == pytest.approx(2.5)
        assert result.ecospace.dispersal_rate[0] == 0.0

    def test_yesno_boolean_conversion(self):
        for adv_val, expected in [
            ("Yes", True),
            ("yes", True),
            (1, True),
            ("No", False),
            (0, False),
            (False, False),
        ]:
            group_df = pd.DataFrame(
                [
                    {
                        "ScenarioID": 1,
                        "GroupID": 1,
                        "EcopathGroupID": 1,
                        "Mvel": 1.0,
                        "RelMoveBad": 0.5,
                        "RelVulBad": 0.5,
                        "IsAdvected": adv_val,
                        "IsMigratory": False,
                        "BarrierAvoidanceWeight": 0.0,
                    }
                ]
            )
            table_map = {
                "EcospaceScenario": _mock_scenario_df(n_rows=1, n_cols=1),
                "EcospaceScenarioGroup": group_df,
            }
            result = self._read_with_mocks(table_map, n_groups=1)
            assert result.ecospace.advection_enabled[0] == expected, (
                f"IsAdvected={adv_val!r} should map to {expected}"
            )

    def test_group_beyond_n_groups_ignored(self):
        group_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "GroupID": 1,
                    "EcopathGroupID": 1,
                    "Mvel": 1.0,
                    "RelMoveBad": 0.5,
                    "RelVulBad": 0.5,
                    "IsAdvected": False,
                    "IsMigratory": False,
                    "BarrierAvoidanceWeight": 0.0,
                },
                {
                    "ScenarioID": 1,
                    "GroupID": 99,
                    "EcopathGroupID": 99,
                    "Mvel": 5.0,
                    "RelMoveBad": 0.5,
                    "RelVulBad": 0.5,
                    "IsAdvected": False,
                    "IsMigratory": False,
                    "BarrierAvoidanceWeight": 0.0,
                },
            ]
        )
        table_map = {
            "EcospaceScenario": _mock_scenario_df(n_rows=1, n_cols=1),
            "EcospaceScenarioGroup": group_df,
        }
        result = self._read_with_mocks(table_map, n_groups=2)
        assert result.ecospace.dispersal_rate[0] == pytest.approx(1.0)
        assert result.ecospace.dispersal_rate[1] == 0.0

    def test_zero_groups_handled(self):
        table_map = {"EcospaceScenario": _mock_scenario_df(n_rows=1, n_cols=1)}
        result = self._read_with_mocks(table_map, n_groups=0)
        assert result.ecospace.habitat_preference.shape == (0, 1)
        assert result.ecospace.dispersal_rate.shape == (0,)
