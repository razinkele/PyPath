"""
Integration tests for spatial Ecosim behaviors.

Verifies three core properties of the spatial simulation:
1. Mass conservation - total biomass is conserved (accounting for dynamics)
2. Movement redistribution - dispersal spreads biomass correctly
3. Zero-biomass patches - empty patches behave correctly

These tests address GitHub Issue #6.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial


@pytest.fixture
def base_scenario():
    """Balanced 3-consumer scenario for integration tests.

    Model: Phyto (producer) -> Zoo -> Fish (fished) + Det + Fleet.
    Returns (scenario, n_groups) where n_groups excludes index-0.
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
    n_groups = scenario.params.NUM_GROUPS
    return scenario, n_groups


def _make_ecospace(n_groups, n_patches, dispersal=2.0, advection=False, gravity=0.0):
    """Helper to build EcospaceParams with uniform habitat."""
    ng = n_groups + 1  # include index-0
    grid = create_1d_grid(n_patches=n_patches, spacing=1.0)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, n_patches)),
        habitat_capacity=np.ones((ng, n_patches)),
        dispersal_rate=np.full(ng, dispersal),
        advection_enabled=np.full(ng, advection, dtype=bool),
        gravity_strength=np.full(ng, gravity),
    )
