"""I/O tests for Fleet Dynamics."""

from unittest.mock import patch

import numpy as np
import pandas as pd


class TestFleetDynamicsSchema:
    def test_scenario_fleet_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcosimScenarioFleet" in EWE_TABLES
        tbl = EWE_TABLES["EcosimScenarioFleet"]
        assert tbl["EcopathFleetID"] == "INTEGER"
        assert tbl["CapDepreciate"] == "DOUBLE"
        assert tbl["CapBaseGrowth"] == "DOUBLE"
        assert tbl["EffPower"] == "DOUBLE"

    def test_quota_table_exists(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcosimScenarioQuota" in EWE_TABLES
        tbl = EWE_TABLES["EcosimScenarioQuota"]
        assert tbl["GroupID"] == "INTEGER"
        assert tbl["FleetID"] == "INTEGER"
        assert tbl["TAC"] == "DOUBLE"


class TestReadFleetDynamics:
    def test_reads_fleet_costs(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        fl_df = pd.DataFrame(
            [
                {
                    "FleetID": 1,
                    "FleetName": "Trawl",
                    "FixedCost": 100.0,
                    "VariableCost": 10.0,
                    "SailingCost": 5.0,
                }
            ]
        )
        table_map = {"EcopathFleet": fl_df}
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                params = read_fleet_dynamics(
                    "fake.eweaccdb",
                    n_fleets=1,
                    n_links=3,
                    n_groups=3,
                    fleet_ids=[1],
                    fishing_links={"FishFrom": [0, 1, 2], "FishThrough": [0, 5, 5]},
                )

        assert params.fixed_cost[0] == 100.0
        assert params.variable_cost[0] == 10.0
        assert params.sailing_cost[0] == 5.0

    def test_reads_prices_from_ecopathcatch(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        catch_df = pd.DataFrame(
            [
                {
                    "GroupID": 1,
                    "FleetID": 1,
                    "Landing": 0.5,
                    "Discards": 0.0,
                    "DiscardMortality": 0.0,
                    "Price": 25.0,
                },
                {
                    "GroupID": 2,
                    "FleetID": 1,
                    "Landing": 0.3,
                    "Discards": 0.0,
                    "DiscardMortality": 0.0,
                    "Price": 40.0,
                },
            ]
        )
        table_map = {"EcopathCatch": catch_df}
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                params = read_fleet_dynamics(
                    "fake.eweaccdb",
                    n_fleets=1,
                    n_links=3,
                    n_groups=3,
                    fleet_ids=[1],
                    fishing_links={"FishFrom": [0, 1, 2], "FishThrough": [0, 5, 5]},
                )

        # Link 1 targets GroupID=1, Link 2 targets GroupID=2
        assert params.price[1] == 25.0
        assert params.price[2] == 40.0

    def test_reads_dynamics_params(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        sf_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "EcopathFleetID": 1,
                    "CapDepreciate": 0.1,
                    "CapBaseGrowth": 0.5,
                    "EffPower": 0.8,
                }
            ]
        )
        table_map = {"EcosimScenarioFleet": sf_df}
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                params = read_fleet_dynamics(
                    "fake.eweaccdb",
                    n_fleets=1,
                    n_links=3,
                    n_groups=3,
                    fleet_ids=[1],
                    fishing_links={"FishFrom": [0, 1], "FishThrough": [0, 5]},
                )

        assert params.cap_depreciate[0] == 0.1
        assert params.cap_base_growth[0] == 0.5
        assert params.eff_power[0] == 0.8

    def test_reads_quotas(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        q_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "GroupID": 1,
                    "FleetID": 1,
                    "QuotaShare": 1.0,
                    "TAC": 50.0,
                },
                {
                    "ScenarioID": 1,
                    "GroupID": 2,
                    "FleetID": 1,
                    "QuotaShare": 1.0,
                    "TAC": 30.0,
                },
            ]
        )
        table_map = {"EcosimScenarioQuota": q_df}
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                params = read_fleet_dynamics(
                    "fake.eweaccdb",
                    n_fleets=1,
                    n_links=3,
                    n_groups=3,
                    fleet_ids=[1],
                    fishing_links={"FishFrom": [0, 1, 2], "FishThrough": [0, 5, 5]},
                )

        assert params.tac is not None
        assert params.tac[0, 0] == 50.0  # group 1 -> idx 0
        assert params.tac[0, 1] == 30.0  # group 2 -> idx 1

    def test_missing_tables_returns_default(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=["SomeOtherTable"]
        ):
            params = read_fleet_dynamics(
                "fake.eweaccdb",
                n_fleets=1,
                n_links=3,
                n_groups=3,
                fleet_ids=[1],
                fishing_links={"FishFrom": [0, 1], "FishThrough": [0, 5]},
            )

        np.testing.assert_array_equal(params.fixed_cost, 0.0)
        np.testing.assert_array_equal(params.eff_power, 1.0)
        assert params.tac is None

    def test_db_exception_returns_default(self):
        from pypath.io.ewemdb import read_fleet_dynamics

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", side_effect=Exception("No driver")
        ):
            params = read_fleet_dynamics(
                "fake.eweaccdb",
                n_fleets=1,
                n_links=3,
                n_groups=3,
                fleet_ids=[1],
                fishing_links={"FishFrom": [0, 1], "FishThrough": [0, 5]},
            )

        assert params.fixed_cost.shape == (1,)
        assert params.tac is None
