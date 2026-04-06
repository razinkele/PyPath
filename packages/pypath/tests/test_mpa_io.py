"""I/O tests for MPA support."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


class TestMPASchema:
    def test_mpa_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcospaceScenarioMPA" in EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioMPA"]
        assert tbl["MPAID"] == "INTEGER"
        assert tbl["MPAname"] == "TEXT"
        assert tbl["MPAmonth"] == "INTEGER"

    def test_mpa_fishery_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcospaceScenarioMPAFishery" in EWE_TABLES
        tbl = EWE_TABLES["EcospaceScenarioMPAFishery"]
        assert tbl["MPAID"] == "INTEGER"
        assert tbl["FleetID"] == "INTEGER"
        assert tbl["Excluded"] == "YESNO"

    def test_mpa_patch_table_removed(self):
        """EcospaceScenarioMPAPatch doesn't exist in real EwE 6.6+ databases."""
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcospaceScenarioMPAPatch" not in EWE_TABLES


class TestReadMPAConfig:
    def test_reads_basic_mpa(self):
        from pypath.io.ewemdb import read_mpa_config

        mpa_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "MPAID": 1,
                    "Sequence": 1,
                    "MPAname": "Reserve",
                    "MPAmonth": 0,
                },
            ]
        )
        patch_df = pd.DataFrame(
            [
                {"ScenarioID": 1, "MPAID": 1, "PatchID": 1},
                {"ScenarioID": 1, "MPAID": 1, "PatchID": 2},
                {"ScenarioID": 1, "MPAID": 1, "PatchID": 3},
            ]
        )
        table_map = {
            "EcospaceScenarioMPA": mpa_df,
            "EcospaceScenarioMPAPatch": patch_df,
        }
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                cfg = read_mpa_config(
                    "fake.eweaccdb",
                    n_patches=5,
                    fleet_ids=[1],
                )

        assert len(cfg.zones) == 1
        assert cfg.zones[0].name == "Reserve"
        assert cfg.zones[0].patches == [0, 1, 2]  # 1-based -> 0-based
        assert cfg.zones[0].excluded_fleets is None  # no fishery table -> no-take
        assert cfg.zones[0].start_month == 0

    def test_reads_fleet_exclusions(self):
        from pypath.io.ewemdb import read_mpa_config

        mpa_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "MPAID": 1,
                    "Sequence": 1,
                    "MPAname": "Selective",
                    "MPAmonth": 6,
                },
            ]
        )
        patch_df = pd.DataFrame(
            [
                {"ScenarioID": 1, "MPAID": 1, "PatchID": 1},
            ]
        )
        fish_df = pd.DataFrame(
            [
                {"ScenarioID": 1, "MPAID": 1, "FleetID": 1, "Excluded": True},
                {"ScenarioID": 1, "MPAID": 1, "FleetID": 2, "Excluded": False},
            ]
        )
        table_map = {
            "EcospaceScenarioMPA": mpa_df,
            "EcospaceScenarioMPAPatch": patch_df,
            "EcospaceScenarioMPAFishery": fish_df,
        }
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                cfg = read_mpa_config(
                    "fake.eweaccdb",
                    n_patches=5,
                    fleet_ids=[1, 2],
                )

        assert cfg.zones[0].excluded_fleets == [0]  # FleetID 1 -> 0-based idx 0
        assert cfg.zones[0].start_month == 6

    def test_converts_patch_ids_to_0based(self):
        from pypath.io.ewemdb import read_mpa_config

        mpa_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "MPAID": 1,
                    "Sequence": 1,
                    "MPAname": "Test",
                    "MPAmonth": 0,
                },
            ]
        )
        patch_df = pd.DataFrame(
            [
                {"ScenarioID": 1, "MPAID": 1, "PatchID": 5},
            ]
        )
        table_map = {
            "EcospaceScenarioMPA": mpa_df,
            "EcospaceScenarioMPAPatch": patch_df,
        }
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                cfg = read_mpa_config(
                    "fake.eweaccdb",
                    n_patches=10,
                    fleet_ids=[1],
                )

        assert cfg.zones[0].patches == [4]  # PatchID 5 -> 0-based 4

    def test_fishery_table_all_false_means_open(self):
        """If fishery table exists but all Excluded=False, MPA is open."""
        from pypath.io.ewemdb import read_mpa_config

        mpa_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "MPAID": 1,
                    "Sequence": 1,
                    "MPAname": "Open",
                    "MPAmonth": 0,
                },
            ]
        )
        patch_df = pd.DataFrame(
            [
                {"ScenarioID": 1, "MPAID": 1, "PatchID": 1},
            ]
        )
        fish_df = pd.DataFrame(
            [
                {"ScenarioID": 1, "MPAID": 1, "FleetID": 1, "Excluded": False},
                {"ScenarioID": 1, "MPAID": 1, "FleetID": 2, "Excluded": False},
            ]
        )
        table_map = {
            "EcospaceScenarioMPA": mpa_df,
            "EcospaceScenarioMPAPatch": patch_df,
            "EcospaceScenarioMPAFishery": fish_df,
        }
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                cfg = read_mpa_config(
                    "fake.eweaccdb",
                    n_patches=5,
                    fleet_ids=[1, 2],
                )

        assert cfg.zones[0].excluded_fleets == []  # open, not no-take

    def test_missing_tables_returns_empty(self):
        from pypath.io.ewemdb import read_mpa_config

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=["SomeOtherTable"]
        ):
            cfg = read_mpa_config(
                "fake.eweaccdb",
                n_patches=5,
                fleet_ids=[1],
            )

        assert len(cfg.zones) == 0

    def test_db_exception_returns_empty(self):
        from pypath.io.ewemdb import read_mpa_config

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", side_effect=Exception("No driver")
        ):
            cfg = read_mpa_config(
                "fake.eweaccdb",
                n_patches=5,
                fleet_ids=[1],
            )

        assert len(cfg.zones) == 0
