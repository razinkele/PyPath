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

        scenario_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "ScenarioName": "IntTest",
                    "Description": "",
                    "Inrow": 2,
                    "Incol": 2,
                    "CellLength": 10.0,
                    "CellSize": 100.0,
                    "MinLon": 0.0,
                    "MinLat": 0.0,
                    "TotalTime": 2.0,
                    "TimeStep": 1.0,
                }
            ]
        )
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
                    "GroupID": 2,
                    "EcopathGroupID": 2,
                    "Mvel": 0.5,
                    "RelMoveBad": 0.5,
                    "RelVulBad": 0.5,
                    "IsAdvected": False,
                    "IsMigratory": False,
                    "BarrierAvoidanceWeight": 0.0,
                },
                {
                    "ScenarioID": 1,
                    "GroupID": 3,
                    "EcopathGroupID": 3,
                    "Mvel": 0.0,
                    "RelMoveBad": 0.5,
                    "RelVulBad": 0.5,
                    "IsAdvected": False,
                    "IsMigratory": False,
                    "BarrierAvoidanceWeight": 0.0,
                },
            ]
        )
        table_map = {
            "EcospaceScenario": scenario_df,
            "EcospaceScenarioGroup": group_df,
        }

        # n_groups=4: Producer + Consumer + Det + Fleet (all groups including fleets)
        # This must match scenario.params.NUM_GROUPS so ecospace arrays are sized correctly
        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                eco_result = read_ecospace("fake.eweaccdb", n_groups=4)

        # Build Ecopath/Ecosim model
        params = _make_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpath_result = rpath(params)
        scenario = rsim_scenario(rpath_result, params, years=range(1, 3))

        # Run spatial sim with read_ecospace result
        result = rsim_run_spatial(scenario, ecospace=eco_result.ecospace)

        # Verify valid results: shape = (n_months+1, n_groups+1, n_patches)
        assert result.out_Biomass_spatial.shape[1] == 5  # 4 groups + 1 (1-based)
        assert result.out_Biomass_spatial.shape[2] == 4  # 2x2 grid
        assert np.all(np.isfinite(result.out_Biomass_spatial))
        assert np.all(result.out_Biomass_spatial >= 0)
