"""Each Ecospace patch must use its own share of the base biomass and flows (#51).

A patch that holds 1/N of the biomass must also hold 1/N of Bbase and QQbase,
so that N identical patches add up to the non-spatial Ecosim run.
"""

import numpy as np
import pytest

from pypath.core.ecosim import rsim_run
from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial


def _uniform_ecospace(n_groups_plus_1, n_patches, dispersal=0.0):
    return EcospaceParams(
        grid=create_1d_grid(n_patches=n_patches, spacing=1.0),
        habitat_preference=np.ones((n_groups_plus_1, n_patches)),
        habitat_capacity=np.ones((n_groups_plus_1, n_patches)),
        dispersal_rate=np.full(n_groups_plus_1, dispersal),
        advection_enabled=np.zeros(n_groups_plus_1, dtype=bool),
        gravity_strength=np.zeros(n_groups_plus_1),
    )


@pytest.mark.parametrize("n_patches", [2, 5])
@pytest.mark.parametrize("dispersal", [0.0, 2.0])
def test_uniform_patches_add_up_to_ecosim(spatial_scenario, n_patches, dispersal):
    scenario, _ = spatial_scenario
    ng = scenario.params.NUM_GROUPS + 1
    years = range(1, 4)

    expected = rsim_run(scenario, years=years).out_Biomass
    got = rsim_run_spatial(
        scenario,
        ecospace=_uniform_ecospace(ng, n_patches, dispersal),
        years=years,
    )

    np.testing.assert_allclose(got.out_Biomass, expected, rtol=1e-6, atol=1e-10)
    # Every patch holds the same share of every group.
    spatial = got.out_Biomass_spatial
    np.testing.assert_allclose(
        spatial, np.repeat(spatial[:, :, :1], n_patches, axis=2), rtol=1e-9
    )


def test_uniform_patches_with_fishing_change(spatial_scenario):
    """Same check with fishing effort doubled, so the run leaves equilibrium."""
    scenario, _ = spatial_scenario
    ng = scenario.params.NUM_GROUPS + 1
    scenario.fishing.ForcedEffort[:] = 2.0
    years = range(1, 4)

    expected = rsim_run(scenario, years=years).out_Biomass
    got = rsim_run_spatial(scenario, ecospace=_uniform_ecospace(ng, 4), years=years)

    assert not np.allclose(expected[-1], expected[0])
    np.testing.assert_allclose(got.out_Biomass, expected, rtol=1e-6, atol=1e-10)
