"""I/O tests for pedigree functions."""

from unittest.mock import patch

import pandas as pd
import pytest


class TestPedigreeSchema:
    def test_pedigree_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "Pedigree" in EWE_TABLES

    def test_pedigree_table_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        tbl = EWE_TABLES["Pedigree"]
        assert tbl["LevelID"] == "INTEGER"
        assert tbl["VarName"] == "TEXT"
        assert tbl["IndexValue"] == "DOUBLE"
        assert tbl["Confidence"] == "DOUBLE"

    def test_group_pedigree_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcopathGroupPedigree" in EWE_TABLES
        tbl = EWE_TABLES["EcopathGroupPedigree"]
        assert "GroupID" in tbl
        assert "VarName" in tbl
        assert "LevelID" in tbl

    def test_sample_tables_exist(self):
        from pypath.io._ewe_schema import EWE_TABLES

        for name in [
            "EcopathSample",
            "EcopathGroupSample",
            "EcopathDietCompSample",
            "EcopathGroupCatchSample",
        ]:
            assert name in EWE_TABLES


class TestReadPedigree:
    def test_reads_pedigree_levels(self):
        from pypath.io.ewemdb import read_pedigree

        ped_df = pd.DataFrame(
            [
                {
                    "LevelID": 6,
                    "LevelName": "Guesstimate",
                    "VarName": "PBInput",
                    "Sequence": 1,
                    "IndexValue": 0.1,
                    "Confidence": 70.0,
                    "LevelColor": 0,
                    "Description": "",
                },
                {
                    "LevelID": 7,
                    "LevelName": "Other model",
                    "VarName": "PBInput",
                    "Sequence": 2,
                    "IndexValue": 0.2,
                    "Confidence": 60.0,
                    "LevelColor": 0,
                    "Description": "",
                },
            ]
        )
        gp_df = pd.DataFrame(
            [
                {"GroupID": 1, "VarName": "PBInput", "LevelID": 6},
            ]
        )

        table_map = {
            "Pedigree": ped_df,
            "EcopathGroupPedigree": gp_df,
        }
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                config, group_ped = read_pedigree("fake.eweaccdb")

        assert config.level_to_cv["PBInput"][6] == 0.1
        assert config.level_to_cv["PBInput"][7] == 0.2
        assert len(group_ped) == 1
        assert group_ped.iloc[0]["CV"] == pytest.approx(0.1)

    def test_missing_tables_returns_empty(self):
        from pypath.io.ewemdb import read_pedigree

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=["SomeOtherTable"]
        ):
            config, group_ped = read_pedigree("fake.eweaccdb")

        assert config.level_to_cv == {}
        assert len(group_ped) == 0

    def test_db_exception_returns_empty(self):
        from pypath.io.ewemdb import read_pedigree

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", side_effect=Exception("No driver")
        ):
            config, group_ped = read_pedigree("fake.eweaccdb")

        assert len(group_ped) == 0
