"""Habitat capacity in Ecospace follows EwE6 (#58).

PyPath multiplied each patch's B_BaseRef by the capacity and kept the base
flows, so a group in poor habitat (low capacity) had its base biomass cut
while its base consumption stayed: its per-capita consumption rose as its
habitat got worse, and it grew fastest where it should be limited.

EwE6 (cEcoSpace.vb) instead
- starts each cell at B * HabCap * nWaterCells / TotHabCap (capacity
  normalised to an area-weighted mean of 1), and
- rescales each link's search rate by the map-wide overlap of predator and
  prey habitat (MeanBP), so map-wide base consumption equals Ecopath's.

In PyPath's closed-form kernel that is: per patch, base biomass B * c and
base link flow Q0 * c_prey * c_pred / mean(c_prey * c_pred). Per-capita rates
then depend on the habitat of the groups a group interacts with, not on its
own capacity, and the map as a whole starts at its Ecopath balance.
"""

import numpy as np
import pytest

import pypath.spatial.integration as integ
from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial
from pypath.spatial.environmental import EnvironmentalDrivers

ZOO, FISH = 2, 3  # rsim rows in the spatial_scenario fixture


def _ecospace(n_rows, capacity):
    n_patches = capacity.shape[1]
    return EcospaceParams(
        grid=create_1d_grid(n_patches=n_patches, spacing=1.0),
        habitat_preference=np.ones((n_rows, n_patches)),
        habitat_capacity=capacity,
        dispersal_rate=np.zeros(n_rows),
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
    )


def _zoo_capacity(n_rows, values):
    cap = np.ones((n_rows, len(values)))
    cap[ZOO - 1] = values  # capacity rows are 0-based groups
    return cap


def _first_call(scenario, ecospace, drivers):
    """Arguments of the first deriv_vector_spatial call of a run."""
    captured = {}
    original = integ.deriv_vector_spatial

    def spy(state, params, forcing, fishing, eco, env, *args, **kwargs):
        if "args" not in captured:
            captured["args"] = (
                state.copy(),
                params,
                forcing,
                fishing,
                eco,
                env,
                kwargs,
            )
        return original(state, params, forcing, fishing, eco, env, *args, **kwargs)

    integ.deriv_vector_spatial = spy
    try:
        rsim_run_spatial(
            scenario,
            ecospace=ecospace,
            environmental_drivers=drivers,
            years=range(1, 3),
        )
    finally:
        integ.deriv_vector_spatial = original
    return captured["args"]


def _local_rates(args):
    state, params, forcing, fishing, eco, env, kwargs = args
    kwargs = {k: v for k, v in kwargs.items() if k != "rates_out"}
    return integ.deriv_vector_spatial(
        state, params, forcing, fishing, eco, env, **kwargs
    )


def test_run_starts_at_biomass_times_capacity(spatial_scenario):
    scenario, _ = spatial_scenario
    n_rows = scenario.params.NUM_GROUPS + 1
    eco = _ecospace(n_rows, _zoo_capacity(n_rows, [0.2, 1.8]))
    out = rsim_run_spatial(
        scenario,
        ecospace=eco,
        environmental_drivers=EnvironmentalDrivers(),
        years=range(1, 3),
    )
    b_zoo = scenario.start_state.Biomass[ZOO]
    np.testing.assert_allclose(
        out.out_Biomass_spatial[0, ZOO], b_zoo * 0.5 * np.array([0.2, 1.8]), rtol=1e-12
    )
    # map-wide start biomass is the Ecopath biomass
    np.testing.assert_allclose(
        out.out_Biomass[0], scenario.start_state.Biomass, rtol=1e-12
    )


def test_map_wide_rates_do_not_depend_on_the_habitat_pattern(spatial_scenario):
    """Summed over the patches, the start derivative is that of the uniform map
    (the model's own, whether or not it starts at its Ecopath balance): the
    habitat pattern redistributes the base flows, it does not create any."""
    scenario, _ = spatial_scenario
    n_rows = scenario.params.NUM_GROUPS + 1
    drivers = EnvironmentalDrivers()
    uniform = _local_rates(
        _first_call(scenario, _ecospace(n_rows, np.ones((n_rows, 2))), drivers)
    )
    varied = _local_rates(
        _first_call(
            scenario, _ecospace(n_rows, _zoo_capacity(n_rows, [0.2, 1.8])), drivers
        )
    )
    np.testing.assert_allclose(
        varied.sum(axis=1), uniform.sum(axis=1), rtol=1e-9, atol=1e-12
    )


def test_own_capacity_does_not_drive_per_capita_growth(spatial_scenario):
    """Zoo's habitat differs between the patches, its partners' does not: Zoo's
    per-capita rate is the same in both patches and the same as on a uniform
    map. Before, the poor patch made Zoo grow and the good one shrink."""
    scenario, _ = spatial_scenario
    n_rows = scenario.params.NUM_GROUPS + 1
    drivers = EnvironmentalDrivers()
    u_args = _first_call(scenario, _ecospace(n_rows, np.ones((n_rows, 2))), drivers)
    r_uniform = _local_rates(u_args)[ZOO] / u_args[0][ZOO]
    v_args = _first_call(
        scenario, _ecospace(n_rows, _zoo_capacity(n_rows, [0.2, 1.8])), drivers
    )
    varied = _local_rates(v_args)
    np.testing.assert_allclose(varied[ZOO] / v_args[0][ZOO], r_uniform, rtol=1e-9)
    # its predator eats more where Zoo is denser
    fish_uniform = _local_rates(u_args)[FISH]
    assert varied[FISH, 1] > fish_uniform[1] and varied[FISH, 0] < fish_uniform[0]


@pytest.mark.parametrize("level", [1.0, 0.5])
def test_uniform_capacity_changes_nothing(spatial_scenario, level):
    scenario, _ = spatial_scenario
    n_rows = scenario.params.NUM_GROUPS + 1
    eco_plain = _ecospace(n_rows, np.ones((n_rows, 3)))
    plain = rsim_run_spatial(scenario, ecospace=eco_plain, years=range(1, 3))
    eco_cap = _ecospace(n_rows, np.full((n_rows, 3), level))
    capped = rsim_run_spatial(
        scenario,
        ecospace=eco_cap,
        environmental_drivers=EnvironmentalDrivers(),
        years=range(1, 3),
    )
    np.testing.assert_allclose(capped.out_Biomass, plain.out_Biomass, rtol=1e-9)
