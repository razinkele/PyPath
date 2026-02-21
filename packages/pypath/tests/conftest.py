"""Shared pytest configuration for pypath-ewe core tests.

The pypath package should be installed via `pip install -e packages/pypath`.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_1d_grid


@pytest.fixture
def spatial_scenario():
    """Create a balanced 5-group Ecosim scenario for spatial tests.

    Returns (scenario, rpath_params) tuple.
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

    scenario = rsim_scenario(model, params, years=range(1, 11))
    return scenario, params


@pytest.fixture
def single_patch_ecospace(spatial_scenario):
    """1-patch EcospaceParams -- spatial should equal non-spatial."""
    scenario, _ = spatial_scenario
    ng = scenario.params.NUM_GROUPS + 1  # +1 for index-0 "Outside"
    grid = create_1d_grid(n_patches=1)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, 1)),
        habitat_capacity=np.ones((ng, 1)),
        dispersal_rate=np.zeros(ng),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
    )


@pytest.fixture
def simple_ecospace(spatial_scenario):
    """3-patch EcospaceParams with mild dispersal for dynamics tests."""
    scenario, _ = spatial_scenario
    ng = scenario.params.NUM_GROUPS + 1
    grid = create_1d_grid(n_patches=3, spacing=1.0)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, 3)),
        habitat_capacity=np.ones((ng, 3)),
        dispersal_rate=np.full(ng, 2.0),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
    )
