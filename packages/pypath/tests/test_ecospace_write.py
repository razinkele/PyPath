"""Tests for advanced Ecospace I/O (schema, reader, writer, MPA)."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from pypath.io._ewe_schema import EWE_TABLES


class TestSchema:
    """Schema definition tests for Ecospace tables."""

    def test_new_ecospace_tables_exist(self):
        """All 7 new Ecospace tables exist in EWE_TABLES."""
        new_tables = [
            "EcospaceScenarioGroupMigration",
            "EcospaceScenarioMonth",
            "EcospaceScenarioWeightLayer",
            "EcospaceScenarioDataConnection",
            "EcospaceScenarioDataConnectionDisabled",
            "EcospaceScenarioDriverDisabled",
            "EcospaceScenarioHabitatFishery",
        ]
        for table in new_tables:
            assert table in EWE_TABLES, f"{table} missing from EWE_TABLES"

        assert len(EWE_TABLES["EcospaceScenarioGroupMigration"]) == 4
        assert EWE_TABLES["EcospaceScenarioGroupMigration"]["Map"] == "LONGBINARY"

        assert len(EWE_TABLES["EcospaceScenarioMonth"]) == 7
        assert EWE_TABLES["EcospaceScenarioMonth"]["WindXVelMap"] == "LONGBINARY"
        assert EWE_TABLES["EcospaceScenarioMonth"]["UpwellingMap"] == "LONGBINARY"

        assert len(EWE_TABLES["EcospaceScenarioWeightLayer"]) == 7
        assert EWE_TABLES["EcospaceScenarioWeightLayer"]["Weight"] == "DOUBLE"
        assert EWE_TABLES["EcospaceScenarioWeightLayer"]["LayerMap"] == "LONGBINARY"

        assert len(EWE_TABLES["EcospaceScenarioDataConnection"]) == 13
        assert EWE_TABLES["EcospaceScenarioDataConnection"]["DatasetGUID"] == "TEXT"
        assert EWE_TABLES["EcospaceScenarioDataConnection"]["Scale"] == "DOUBLE"

        assert len(EWE_TABLES["EcospaceScenarioDataConnectionDisabled"]) == 3
        assert EWE_TABLES["EcospaceScenarioDataConnectionDisabled"]["Varname"] == "TEXT"

        assert len(EWE_TABLES["EcospaceScenarioDriverDisabled"]) == 3
        assert EWE_TABLES["EcospaceScenarioDriverDisabled"]["Target"] == "TEXT"

        assert len(EWE_TABLES["EcospaceScenarioHabitatFishery"]) == 3
        assert EWE_TABLES["EcospaceScenarioHabitatFishery"]["HabitatID"] == "INTEGER"

    def test_mpa_patch_removed(self):
        """EcospaceScenarioMPAPatch does not exist in real EwE 6.6+ databases."""
        assert "EcospaceScenarioMPAPatch" not in EWE_TABLES


from pypath.io.ewemdb import EcospaceReadResult, read_ecospace, EwEDatabaseError


def _mock_ecospace_tables():
    """Build mock table DataFrames for read_ecospace() testing."""
    return {
        "EcospaceScenario": pd.DataFrame([{
            "ScenarioID": 1, "ScenarioName": "Test", "Description": "",
            "Inrow": 3, "Incol": 3, "CellLength": 1.0, "CellSize": 1.0,
            "MinLon": 0.0, "MinLat": 0.0, "TotalTime": 10, "TimeStep": 1.0,
        }]),
        "EcospaceScenarioDriverLayer": pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Sequence": 1,
            "LayerName": "Temperature", "LayerDescription": "SST",
            "LayerMAP": b"\x00\x01\x02", "LayerUnits": "C",
        }]),
        "EcospaceScenarioGroupMigration": pd.DataFrame([{
            "ScenarioID": 1, "GroupID": 1, "MonthID": 1, "Map": b"\x10\x20",
        }]),
        "EcospaceScenarioMonth": pd.DataFrame([{
            "ScenarioID": 1, "MonthID": 1,
            "WindXVelMap": b"\x01", "WindYVelMap": b"\x02",
            "AdvectionXVelMap": b"\x03", "AdvectionYVelMap": b"\x04",
            "UpwellingMap": b"\x05",
        }]),
        "EcospaceScenarioWeightLayer": pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Sequence": 1,
            "Name": "Weight1", "Description": "test",
            "Weight": 0.5, "LayerMap": b"\xAA",
        }]),
        "EcospaceScenarioDataConnection": pd.DataFrame([{
            "ScenarioID": 1, "VarName": "SST", "LayerID": 1, "Sequence": 1,
            "DatasetGUID": "abc-123", "DatasetTypeName": "NetCDF",
            "DatasetCfg": "{}", "ConverterTypeName": "Linear",
            "ConverterCfg": "{}", "Scale": 1.0, "ScaleType": 0,
            "CustomDateStart": "", "CustomDateEnd": "",
        }]),
        "EcospaceScenarioDataConnectionDisabled": pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Varname": "SST",
        }]),
        "EcospaceScenarioDriverDisabled": pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Target": "group1",
        }]),
        "EcospaceScenarioHabitatFishery": pd.DataFrame([{
            "ScenarioID": 1, "FleetID": 1, "HabitatID": 1,
        }]),
        "EcospaceScenarioFleet": pd.DataFrame([{
            "ScenarioID": 1, "FleetID": 1, "EcopathFleetID": 1,
            "EffPower": 1.0, "PortMap": b"\xFF", "SailCostMap": b"\xEE",
            "SEMult": 1.0,
        }]),
    }


def _mock_read_table(db_path, table):
    """Mock read_ewemdb_table that returns test DataFrames."""
    tables = _mock_ecospace_tables()
    if table in tables:
        return tables[table]
    raise EwEDatabaseError(f"Table {table} not found")


def _mock_list_tables(db_path):
    """Mock list_ewemdb_tables that returns all test table names."""
    return list(_mock_ecospace_tables().keys())


class TestReader:
    """read_ecospace() reader extension tests."""

    @patch("pypath.io.ewemdb.list_ewemdb_tables", side_effect=_mock_list_tables)
    @patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read_table)
    def test_new_fields_populated(self, mock_read, mock_list):
        """New EcospaceReadResult fields are populated from mock DB."""
        result = read_ecospace("dummy.eweaccdb", n_groups=3)
        assert result.driver_layers is not None
        assert result.migration_maps is not None
        assert result.monthly_maps is not None
        assert result.weight_layers is not None
        assert result.data_connections is not None
        assert result.disabled_connections is not None
        assert result.disabled_drivers is not None
        assert result.habitat_fishery is not None

    @patch("pypath.io.ewemdb.list_ewemdb_tables", return_value=["EcospaceScenario"])
    @patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read_table)
    def test_missing_tables_return_none(self, mock_read, mock_list):
        """Missing optional tables return None, not error."""
        result = read_ecospace("dummy.eweaccdb", n_groups=3)
        assert result.driver_layers is None
        assert result.migration_maps is None
        assert result.monthly_maps is None
        assert result.weight_layers is None
        assert result.data_connections is None
        assert result.disabled_connections is None
        assert result.disabled_drivers is None
        assert result.habitat_fishery is None

    @patch("pypath.io.ewemdb.list_ewemdb_tables", side_effect=_mock_list_tables)
    @patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read_table)
    def test_binary_map_columns_preserved(self, mock_read, mock_list):
        """Binary map columns are preserved as raw bytes."""
        result = read_ecospace("dummy.eweaccdb", n_groups=3)
        assert result.driver_layers.iloc[0]["LayerMAP"] == b"\x00\x01\x02"
        assert result.migration_maps.iloc[0]["Map"] == b"\x10\x20"
        assert result.monthly_maps.iloc[0]["WindXVelMap"] == b"\x01"

    @patch("pypath.io.ewemdb.list_ewemdb_tables", side_effect=_mock_list_tables)
    @patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read_table)
    def test_fleet_map_columns_preserved(self, mock_read, mock_list):
        """Fleet PortMap and SailCostMap binary columns are now preserved."""
        result = read_ecospace("dummy.eweaccdb", n_groups=3)
        assert result.fleet_info is not None
        assert "PortMap" in result.fleet_info.columns
        assert "SailCostMap" in result.fleet_info.columns
        assert result.fleet_info.iloc[0]["PortMap"] == b"\xFF"


from pypath.io._csv_bundle_writer import CsvBundleWriter
from pypath.core.params import create_rpath_params


def _make_test_params():
    """Build minimal RpathParams for writer tests."""
    params = create_rpath_params(
        groups=["Fish", "Zooplankton", "Detritus"],
        types=[0, 0, 2],
        stgroups=None,
    )
    params.model.loc["Fish", "Biomass"] = 10.0
    params.model.loc["Fish", "PB"] = 0.5
    params.model.loc["Fish", "QB"] = 2.0
    params.model.loc["Fish", "EE"] = 0.9
    params.model.loc["Zooplankton", "Biomass"] = 50.0
    params.model.loc["Zooplankton", "PB"] = 20.0
    params.model.loc["Zooplankton", "QB"] = 40.0
    params.model.loc["Zooplankton", "EE"] = 0.8
    params.model.loc["Detritus", "Biomass"] = 100.0
    params.model.loc["Detritus", "DetInput"] = 0.0
    params.diet.loc["Fish", "Zooplankton"] = 0.8
    params.diet.loc["Fish", "Detritus"] = 0.2
    params.diet.loc["Zooplankton", "Detritus"] = 1.0
    return params


def _make_ecospace_read_result():
    """Build an EcospaceReadResult with all fields populated for writer tests."""
    from pypath.spatial.ecospace_params import EcospaceParams, EcospaceGrid

    grid = EcospaceGrid.from_regular_grid(bounds=(0, 0, 3, 3), nx=3, ny=3)
    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((3, 9)),
        habitat_capacity=np.ones((3, 9)),
        dispersal_rate=np.array([1.0, 2.0, 0.0]),
        advection_enabled=np.array([True, False, False]),
        gravity_strength=np.zeros(3),
    )

    return EcospaceReadResult(
        ecospace=ecospace,
        habitat_types={0: "Reef", 1: "Sand"},
        fleet_info=pd.DataFrame([{
            "ScenarioID": 1, "FleetID": 1, "EcopathFleetID": 1,
            "EffPower": 1.0, "PortMap": b"\xFF", "SailCostMap": b"\xEE",
            "SEMult": 1.0,
        }]),
        capacity_drivers=pd.DataFrame([{
            "ScenarioID": 1, "GroupID": 1, "VarDBID": 1,
            "ShapeID": 1, "Target": 1,
        }]),
        scenario_meta={"ScenarioName": "Test"},
        driver_layers=pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Sequence": 1,
            "LayerName": "Temp", "LayerDescription": "SST",
            "LayerMAP": b"\x00\x01", "LayerUnits": "C",
        }]),
        migration_maps=pd.DataFrame([{
            "ScenarioID": 1, "GroupID": 1, "MonthID": 1, "Map": b"\x10",
        }]),
        monthly_maps=pd.DataFrame([{
            "ScenarioID": 1, "MonthID": 1,
            "WindXVelMap": b"\x01", "WindYVelMap": b"\x02",
            "AdvectionXVelMap": b"\x03", "AdvectionYVelMap": b"\x04",
            "UpwellingMap": b"\x05",
        }]),
        weight_layers=pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Sequence": 1,
            "Name": "W1", "Description": "test",
            "Weight": 0.5, "LayerMap": b"\xAA",
        }]),
        data_connections=pd.DataFrame([{
            "ScenarioID": 1, "VarName": "SST", "LayerID": 1, "Sequence": 1,
            "DatasetGUID": "abc", "DatasetTypeName": "NetCDF",
            "DatasetCfg": "{}", "ConverterTypeName": "Linear",
            "ConverterCfg": "{}", "Scale": 1.0, "ScaleType": 0,
            "CustomDateStart": "", "CustomDateEnd": "",
        }]),
        disabled_connections=pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Varname": "SST",
        }]),
        disabled_drivers=pd.DataFrame([{
            "ScenarioID": 1, "LayerID": 1, "Target": "group1",
        }]),
        habitat_fishery=pd.DataFrame([{
            "ScenarioID": 1, "FleetID": 1, "HabitatID": 1,
        }]),
    )


class TestWriter:
    """write_ecospace() writer extension tests."""

    def test_habitat_tables_written(self):
        """Habitat types and preferences written to correct tables."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        assert "EcospaceScenarioHabitat" in writer._tables
        hab_df = writer._tables["EcospaceScenarioHabitat"]
        assert len(hab_df) == 2  # 2 habitat types
        assert set(hab_df["HabitatName"]) == {"Reef", "Sand"}
        # 0-based keys -> 1-based HabitatID
        assert set(hab_df["HabitatID"]) == {1, 2}

        assert "EcospaceScenarioGroupHabitat" in writer._tables

    def test_fleet_info_written(self):
        """Fleet info DataFrame written including binary map columns."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        assert "EcospaceScenarioFleet" in writer._tables
        fleet_df = writer._tables["EcospaceScenarioFleet"]
        assert len(fleet_df) == 1
        assert "PortMap" in fleet_df.columns

    def test_dataframe_passthrough_tables(self):
        """All DataFrame passthrough tables written correctly."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        passthrough = {
            "EcospaceScenarioGroupMigration": "migration_maps",
            "EcospaceScenarioMonth": "monthly_maps",
            "EcospaceScenarioWeightLayer": "weight_layers",
            "EcospaceScenarioDataConnection": "data_connections",
            "EcospaceScenarioDataConnectionDisabled": "disabled_connections",
            "EcospaceScenarioDriverDisabled": "disabled_drivers",
            "EcospaceScenarioHabitatFishery": "habitat_fishery",
            "EcospaceScenarioDriverLayer": "driver_layers",
            "EcospaceScenarioCapacityDrivers": "capacity_drivers",
        }
        for table_name in passthrough:
            assert table_name in writer._tables, f"{table_name} not written"

    def test_migration_map_binary_preserved(self):
        """Binary map bytes round-trip through writer."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        mig_df = writer._tables["EcospaceScenarioGroupMigration"]
        assert mig_df.iloc[0]["Map"] == b"\x10"

    def test_monthly_maps_5_binary_columns(self):
        """EcospaceScenarioMonth has all 5 binary map columns."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        month_df = writer._tables["EcospaceScenarioMonth"]
        for col in ["WindXVelMap", "WindYVelMap", "AdvectionXVelMap",
                     "AdvectionYVelMap", "UpwellingMap"]:
            assert col in month_df.columns

    def test_empty_result_writes_no_extra_tables(self):
        """EcospaceReadResult with all None fields writes only base tables."""
        params = _make_test_params()
        from pypath.spatial.ecospace_params import EcospaceParams, EcospaceGrid
        grid = EcospaceGrid.from_regular_grid(bounds=(0, 0, 3, 3), nx=3, ny=3)
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((3, 9)),
            habitat_capacity=np.ones((3, 9)),
            dispersal_rate=np.array([1.0, 2.0, 0.0]),
            advection_enabled=np.array([True, False, False]),
            gravity_strength=np.zeros(3),
        )
        result = EcospaceReadResult(
            ecospace=ecospace,
            habitat_types={},
            fleet_info=None,
            capacity_drivers=None,
            scenario_meta={},
        )
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        assert "EcospaceScenario" in writer._tables
        assert "EcospaceScenarioGroup" in writer._tables
        assert "EcospaceScenarioGroupMigration" not in writer._tables
        assert "EcospaceScenarioMonth" not in writer._tables

    def test_existing_base_tables_unchanged(self):
        """Existing EcospaceScenario and EcospaceScenarioGroup logic preserved."""
        params = _make_test_params()
        result = _make_ecospace_read_result()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_ecospace(result)

        assert "EcospaceScenario" in writer._tables
        scen_df = writer._tables["EcospaceScenario"]
        assert "Inrow" in scen_df.columns

        assert "EcospaceScenarioGroup" in writer._tables
        grp_df = writer._tables["EcospaceScenarioGroup"]
        assert len(grp_df) == 3  # 3 groups


from pypath.io._access_writer import AccessWriter


class TestAccessWriter:
    """Access writer table list tests."""

    def test_ecospace_tables_count(self):
        """_ECOSPACE_TABLES has all 16 Ecospace tables."""
        assert len(AccessWriter._ECOSPACE_TABLES) == 16

    def test_ecospace_tables_children_before_parents(self):
        """Children tables are listed before parent tables for safe clearing."""
        tables = AccessWriter._ECOSPACE_TABLES
        # EcospaceScenario must be last (parent of all)
        assert tables[-1] == "EcospaceScenario"
        # EcospaceScenarioGroup must be second-to-last
        assert tables[-2] == "EcospaceScenarioGroup"
        # MPA child before MPA parent
        mpa_fishery_idx = tables.index("EcospaceScenarioMPAFishery")
        mpa_idx = tables.index("EcospaceScenarioMPA")
        assert mpa_fishery_idx < mpa_idx


from pypath.spatial.mpa import MPAConfig, MPAZone


class TestMPAWriter:
    """write_mpa() tests."""

    def test_mpa_config_written(self):
        """MPAConfig converted to EcospaceScenarioMPA + MPA Fishery tables."""
        params = _make_test_params()
        zones = [
            MPAZone(
                mpa_id=1, name="Reserve A", patches=[0, 1],
                start_month=1, excluded_fleets=[0, 2],
            ),
            MPAZone(
                mpa_id=2, name="Reserve B", patches=[3],
                start_month=6, excluded_fleets=[1],
            ),
        ]
        mpa_config = MPAConfig(zones=zones)

        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_mpa(mpa_config=mpa_config)

        # Check MPA table
        assert "EcospaceScenarioMPA" in writer._tables
        mpa_df = writer._tables["EcospaceScenarioMPA"]
        assert len(mpa_df) == 2
        assert mpa_df.iloc[0]["MPAname"] == "Reserve A"
        assert mpa_df.iloc[1]["MPAmonth"] == 6

        # Check MPA Fishery table (0-based fleet -> 1-based FleetID)
        assert "EcospaceScenarioMPAFishery" in writer._tables
        fish_df = writer._tables["EcospaceScenarioMPAFishery"]
        # Zone 1 has 2 excluded fleets [0,2], Zone 2 has 1 excluded fleet [1]
        assert len(fish_df) == 3
        assert set(fish_df["FleetID"]) == {1, 2, 3}  # 0-based -> 1-based
        assert all(fish_df["Excluded"] == True)  # noqa: E712

    def test_empty_mpa_config(self):
        """Empty MPAConfig writes no tables (no error)."""
        params = _make_test_params()
        writer = CsvBundleWriter(params, "/tmp/test.csv", scenario_id=1)
        writer.write_mpa(mpa_config=MPAConfig(zones=[]))

        assert "EcospaceScenarioMPA" not in writer._tables
        assert "EcospaceScenarioMPAFishery" not in writer._tables
