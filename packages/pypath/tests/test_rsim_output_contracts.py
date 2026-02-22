"""
Contract tests for RsimOutput fields.

Validates all 18 documented fields of RsimOutput for:
- Presence (field exists on the output object)
- Shape (array dimensions match expected layout)
- Dtype (numeric arrays are float, not int or object)
- Value ranges (biomass non-negative, crash_year is int, etc.)
- Consistency (end_state matches final biomass, annual matches monthly)

These tests address GitHub Issue #5.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import RsimState, rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params


@pytest.fixture(scope="module")
def simple_output():
    """Run a 5-group model for 5 years and return (result, scenario).

    Model: Phyto (producer) -> Zoo -> Fish (fished) + Det + Fleet.
    """
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "Det", "Fleet"],
        types=[1, 0, 0, 2, 3],
    )
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9
    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 5.0
    params.model.loc[2, "EE"] = 0.5
    params.model.loc[3, "Biomass"] = 100.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model.loc[4, "BioAcc"] = np.nan
    params.model.loc[4, "Unassim"] = np.nan
    params.model["Det"] = 1.0
    params.model.loc[4, "Det"] = np.nan
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    params.model.loc[2, "Fleet"] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    scenario = rsim_scenario(model, params, years=range(1, 6))
    result = rsim_run(scenario, years=range(1, 6))
    return result, scenario


@pytest.fixture(scope="module")
def baltic_output():
    """Run a 7-group Baltic model for 5 years and return (result, scenario).

    Model: Phyto -> Zoo, Benthos -> Herring -> Cod + Det + Fishery.
    """
    params = create_rpath_params(
        groups=[
            "Phytoplankton",
            "Zooplankton",
            "Benthos",
            "Herring",
            "Cod",
            "Detritus",
            "Fishery",
        ],
        types=[1, 0, 0, 0, 0, 2, 3],
    )
    params.model.loc[0, "Biomass"] = 25.0
    params.model.loc[0, "PB"] = 150.0
    params.model.loc[0, "EE"] = 0.85
    params.model.loc[1, "Biomass"] = 12.0
    params.model.loc[1, "PB"] = 35.0
    params.model.loc[1, "QB"] = 100.0
    params.model.loc[1, "EE"] = 0.90
    params.model.loc[2, "Biomass"] = 30.0
    params.model.loc[2, "PB"] = 3.0
    params.model.loc[2, "QB"] = 10.0
    params.model.loc[2, "EE"] = 0.80
    params.model.loc[3, "Biomass"] = 8.0
    params.model.loc[3, "PB"] = 1.2
    params.model.loc[3, "QB"] = 4.0
    params.model.loc[3, "EE"] = 0.75
    params.model.loc[4, "Biomass"] = 3.0
    params.model.loc[4, "PB"] = 0.5
    params.model.loc[4, "QB"] = 2.5
    params.model.loc[4, "EE"] = 0.40
    params.model.loc[5, "Biomass"] = 50.0
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[5, "Unassim"] = 0.0
    params.model.loc[6, "BioAcc"] = np.nan
    params.model.loc[6, "Unassim"] = np.nan
    params.model["Detritus"] = 1.0
    params.model.loc[6, "Detritus"] = np.nan
    params.diet["Zooplankton"] = [0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
    params.diet["Benthos"] = [0.3, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]
    params.diet["Herring"] = [0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0]
    params.diet["Cod"] = [0.0, 0.2, 0.3, 0.4, 0.1, 0.0, 0.0]
    params.diet["Phytoplankton"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params.model.loc[3, "Fishery"] = 1.5
    params.model.loc[4, "Fishery"] = 0.3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    scenario = rsim_scenario(model, params, years=range(1, 6))
    result = rsim_run(scenario, years=range(1, 6))
    return result, scenario
