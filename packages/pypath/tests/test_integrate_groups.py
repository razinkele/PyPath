"""Integrating chosen fast-equilibrium groups (pypath.core.ecosim.integrate_groups)."""

import copy

import numpy as np
import pytest

from pypath.core.ecosim import integrate_groups, rsim_run, substep_check
from pypath.spatial import EcospaceParams, WaterTransport, rsim_run_spatial
from pypath.spatial.mesh import grid_from_mesh

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detritus fate of a detritus group to itself:UserWarning"
)

YEARS = range(1, 4)


def _fast(scenario):
    """Names of the scenario's fast-equilibrium groups."""
    flags = np.asarray(scenario.params.NoIntegrate)
    return [str(scenario.params.spname[i]) for i in np.flatnonzero(flags[1:] != 0) + 1]


def _perturbed(scenario):
    """Fishing effort doubled from the second year on."""
    out = copy.deepcopy(scenario)
    effort = np.asarray(out.fishing.ForcedEffort, dtype=float).copy()
    effort[12:] *= 2.0
    out.fishing.ForcedEffort = effort
    return out


def test_with_no_group_named_the_run_is_the_default(spatial_scenario):
    scenario, _ = spatial_scenario
    same = integrate_groups(scenario, [])
    assert same is not scenario and same.params is not scenario.params
    np.testing.assert_array_equal(
        rsim_run(same, years=YEARS).out_Biomass,
        rsim_run(scenario, years=YEARS).out_Biomass,
    )


def test_a_named_group_is_integrated_and_the_scenario_given_is_left_alone(
    spatial_scenario,
):
    scenario, _ = spatial_scenario
    fast = _fast(scenario)
    assert fast  # the fixture is built under the EwE rule
    before = np.asarray(scenario.params.NoIntegrate).copy()
    changed = integrate_groups(scenario, [fast[0]], substeps=16)
    assert _fast(changed) == fast[1:]
    assert changed.params.RK4_STEPS == 16
    np.testing.assert_array_equal(scenario.params.NoIntegrate, before)
    assert scenario.params.RK4_STEPS == 1


def test_groups_by_row_are_numbered_as_transport_numbers_them(spatial_scenario):
    scenario, _ = spatial_scenario
    name = _fast(scenario)[0]
    row = [str(n) for n in scenario.params.spname].index(name) - 1
    by_name = integrate_groups(scenario, [name])
    by_row = integrate_groups(scenario, [row])
    np.testing.assert_array_equal(by_name.params.NoIntegrate, by_row.params.NoIntegrate)
    assert WaterTransport(  # the same list names the carried groups
        zones=np.array([1, 2]),
        flow=np.zeros((1, 2, 2)),
        volume=np.ones((1, 2)),
        groups=[row],
    ).groups == [row]


def test_unknown_groups_and_bad_substeps_are_refused(spatial_scenario):
    scenario, _ = spatial_scenario
    with pytest.raises(KeyError, match="Cod"):
        integrate_groups(scenario, ["Cod"])
    with pytest.raises(KeyError, match="row 99"):
        integrate_groups(scenario, [99])
    for bad in (0, -4, 2.5):
        with pytest.raises(ValueError, match="substeps"):
            integrate_groups(scenario, [], substeps=bad)


def test_a_group_that_is_already_integrated_changes_nothing(spatial_scenario):
    scenario, _ = spatial_scenario
    names = [str(n) for n in scenario.params.spname]
    slow = next(n for n in names[1:] if n not in _fast(scenario))
    same = integrate_groups(scenario, [slow])
    np.testing.assert_array_equal(same.params.NoIntegrate, scenario.params.NoIntegrate)


def test_under_the_rpath_rule_every_group_is_integrated_already(spatial_scenario):
    scenario, _ = spatial_scenario
    rpath_rule = copy.deepcopy(scenario)
    rpath_rule.params.RK4_FAST_EQUILIBRIUM = False
    same = integrate_groups(rpath_rule, _fast(scenario), substeps=8)
    np.testing.assert_array_equal(same.params.NoIntegrate, scenario.params.NoIntegrate)
    assert same.params.RK4_STEPS == 8


def test_integrated_with_enough_substeps_the_run_settles_near_the_default(
    spatial_scenario,
):
    """The fixture is far from balance: it grows many-fold in three years.

    The integrated run no longer changes with more sub-steps, and what
    separates it from the default is the difference between an equilibrium
    and its integration: large in the first months of a violent transient,
    a percent by the end.
    """
    scenario, _ = spatial_scenario
    live = slice(1, scenario.params.NUM_LIVING + scenario.params.NUM_DEAD + 1)

    def run(substeps):
        changed = integrate_groups(
            _perturbed(scenario), _fast(scenario), substeps=substeps
        )
        return rsim_run(changed, years=YEARS).out_Biomass[:, live]

    default = rsim_run(_perturbed(scenario), years=YEARS).out_Biomass[:, live]
    coarse, fine = run(16), run(64)
    assert np.isfinite(fine).all()
    np.testing.assert_allclose(coarse, fine, rtol=1e-3)
    np.testing.assert_allclose(fine[-1], default[-1], rtol=0.02)
    assert np.abs(fine[1] / default[1] - 1).max() > 0.05  # the early months differ


def test_substep_check_tells_enough_from_too_few(spatial_scenario):
    scenario, _ = spatial_scenario
    few = integrate_groups(_perturbed(scenario), _fast(scenario), substeps=1)
    enough = integrate_groups(_perturbed(scenario), _fast(scenario), substeps=16)
    good = substep_check(enough, years=YEARS)
    assert good.substeps == 16 and good.finer == 32
    assert good.max_difference < 1e-3 and good.converged
    assert good.group in [str(n) for n in scenario.params.spname]
    bad = substep_check(few, years=YEARS)
    assert not bad.converged
    assert bad.max_difference > 1e-2 or not np.isfinite(bad.max_difference)
    # the scenario checked is left as it was
    assert few.params.RK4_STEPS == 1


def _zone_grid():
    xy = np.array([(ix * 2000.0, iy * 2000.0) for iy in range(2) for ix in range(7)])
    quads = np.array([[ix, ix + 1, 8 + ix, 7 + ix] for ix in range(6)])
    grid = grid_from_mesh(xy, quads, np.repeat([10, 20, 30], [1, 2, 3]))
    grid.cell_metadata["type"] = grid.patch_ids
    return grid


def _ecospace(scenario, grid, transport=None):
    n_rows = scenario.params.NUM_GROUPS + 1
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n_rows, grid.n_patches)),
        habitat_capacity=np.ones((n_rows, grid.n_patches)),
        dispersal_rate=np.zeros(n_rows),
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
        water_transport=transport,
    )


def _ring(groups):
    flow = np.zeros((36, 3, 3))
    for i in range(3):
        flow[:, i, (i + 1) % 3] = 2000.0
        flow[:, (i + 1) % 3, i] = 1500.0
    return WaterTransport(
        zones=np.array([10, 20, 30]),
        flow=flow,
        volume=np.tile([1e8, 2e8, 4e8], (36, 1)),
        groups=groups,
    )


def test_in_space_an_integrated_group_follows_ecosim_and_can_be_carried(
    spatial_scenario,
):
    scenario, _ = spatial_scenario
    name = _fast(scenario)[0]
    row = [str(n) for n in scenario.params.spname].index(name) - 1
    integrated = integrate_groups(scenario, [name], substeps=16)
    grid = _zone_grid()
    plain = rsim_run(integrated, years=YEARS).out_Biomass
    live = slice(1, scenario.params.NUM_LIVING + scenario.params.NUM_DEAD + 1)
    spatial = rsim_run_spatial(
        integrated, ecospace=_ecospace(integrated, grid), years=YEARS
    )
    np.testing.assert_allclose(spatial.out_Biomass[:, live], plain[:, live], rtol=1e-9)
    carried = rsim_run_spatial(
        integrated, ecospace=_ecospace(integrated, grid, _ring([row])), years=YEARS
    )
    np.testing.assert_allclose(
        carried.out_Biomass_spatial, spatial.out_Biomass_spatial, rtol=1e-9
    )


def test_transport_tells_how_to_integrate_a_group_that_is_still_fast(
    spatial_scenario,
):
    scenario, _ = spatial_scenario
    name = _fast(scenario)[0]
    row = [str(n) for n in scenario.params.spname].index(name) - 1
    with pytest.raises(ValueError) as error:
        rsim_run_spatial(
            scenario,
            ecospace=_ecospace(scenario, _zone_grid(), _ring([row])),
            years=range(1, 2),
        )
    message = str(error.value)
    assert "fast-equilibrium group" in message and name in message
    assert "integrate_groups" in message
