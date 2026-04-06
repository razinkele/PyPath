"""I/O tests for Ecotracer."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


class TestEcotracerSchema:
    def test_scenario_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcotracerScenario" in EWE_TABLES
        tbl = EWE_TABLES["EcotracerScenario"]
        assert tbl["ScenarioID"] == "INTEGER"
        assert tbl["Czero"] == "DOUBLE"
        assert tbl["Cinflow"] == "DOUBLE"

    def test_group_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcotracerScenarioGroup" in EWE_TABLES
        tbl = EWE_TABLES["EcotracerScenarioGroup"]
        assert tbl["EcopathGroupID"] == "INTEGER"
        assert tbl["CassimProp"] == "DOUBLE"
        assert tbl["CmetabolismRate"] == "DOUBLE"


class TestReadEcotracer:
    def test_reads_scenario_defaults(self):
        from pypath.io.ewemdb import read_ecotracer

        sc_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "Czero": 0.5,
                    "Cinflow": 0.1,
                    "Cdecay": 0.05,
                }
            ]
        )
        gp_df = pd.DataFrame(columns=["ScenarioID", "EcopathGroupID"])

        table_map = {
            "EcotracerScenario": sc_df,
            "EcotracerScenarioGroup": gp_df,
        }
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                params = read_ecotracer("fake.eweaccdb", 3)

        np.testing.assert_array_equal(params.czero, 0.5)
        np.testing.assert_array_equal(params.cimmig, 0.1)
        np.testing.assert_array_equal(params.cdecay, 0.05)

    def test_reads_group_overrides(self):
        from pypath.io.ewemdb import read_ecotracer

        sc_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "Czero": 0.0,
                    "Cinflow": 0.0,
                    "Cdecay": 0.0,
                }
            ]
        )
        gp_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "EcopathGroupID": 1,
                    "Czero": 1.0,
                    "Cimmig": None,
                    "Cenv": 0.2,
                    "Cdecay": 0.1,
                    "CassimProp": 0.9,
                    "CmetabolismRate": 0.03,
                },
                {
                    "ScenarioID": 1,
                    "EcopathGroupID": 2,
                    "Czero": 0.0,
                    "Cimmig": 0.05,
                    "Cenv": None,
                    "Cdecay": None,
                    "CassimProp": None,
                    "CmetabolismRate": None,
                },
            ]
        )

        table_map = {
            "EcotracerScenario": sc_df,
            "EcotracerScenarioGroup": gp_df,
        }
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                params = read_ecotracer("fake.eweaccdb", 3)

        assert params.czero[0] == 1.0  # group 1 → idx 0
        assert params.cenv[0] == 0.2
        assert params.cassim[0] == 0.9
        assert params.cmetab[0] == 0.03
        assert params.cimmig[1] == 0.05  # group 2 → idx 1

    def test_missing_tables_returns_default(self):
        from pypath.io.ewemdb import read_ecotracer

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=["SomeOtherTable"]
        ):
            params = read_ecotracer("fake.eweaccdb", 3)

        np.testing.assert_array_equal(params.czero, 0.0)
        np.testing.assert_array_equal(params.cassim, 1.0)

    def test_db_exception_returns_default(self):
        from pypath.io.ewemdb import read_ecotracer

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", side_effect=Exception("No driver")
        ):
            params = read_ecotracer("fake.eweaccdb", 3)

        assert params.czero.shape == (3,)
