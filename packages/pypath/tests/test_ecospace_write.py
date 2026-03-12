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
