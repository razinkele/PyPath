"""Implicit (backward-Euler) diffusion in Ecospace runs (roadmap B3).

Diffusion moves rate x (L / d) x (s_i - s_j) across each edge, with s the
area-scaled density B x mean(A) / A (``dispersal.diffusion_flux``): in matrix
form dB/dt = -rate x Lap x D_s x B. One explicit RK4 step a month keeps that
stable only to about 4.2 per year per cell side on a rook grid; EwE's rates
on small cells are far beyond it. ``rsim_run_spatial`` therefore takes
diffusion out of the RK4 derivative and solves (I + h x rate x Lap x D_s)
B_new = B after every RK4 step, for the groups that disperse.
"""

import logging

import numpy as np
import pytest
import scipy.sparse

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_regular_grid, rsim_run_spatial
from pypath.spatial.dispersal import diffusion_flux, implicit_diffusion_step
from pypath.spatial.ecospace_params import EcospaceGrid


def _uneven_grid():
    """Three patches in a row with unequal areas, borders and distances."""
    adjacency = scipy.sparse.csr_matrix(
        ([1, 1, 1, 1], ([0, 1, 1, 2], [1, 0, 2, 1])), shape=(3, 3)
    )
    return EcospaceGrid(
        n_patches=3,
        patch_ids=np.arange(3),
        patch_areas=np.array([1.0, 2.0, 4.0]),
        # 0.03 and 0.05 degrees apart: 3.33 and 5.55 km (x 111)
        patch_centroids=np.array([[0.0, 0.0], [0.03, 0.0], [0.08, 0.0]]),
        adjacency_matrix=adjacency,
        edge_lengths={(0, 1): 1.5, (1, 2): 2.5},
    )


# --- the implicit step ---


def test_a_tiny_step_is_explicit_diffusion_on_uneven_patches():
    # (I + h G)^-1 B = B - h G B + O(h^2): the change per unit time is
    # diffusion_flux's, to within h x rate x |G|.
    grid = _uneven_grid()
    b = np.array([3.0, 1.0, 0.5])
    h, rate = 1e-7, 7.0
    got = implicit_diffusion_step(b, rate, grid, grid.adjacency_matrix, h)
    flux = diffusion_flux(b, rate, grid, grid.adjacency_matrix)
    np.testing.assert_allclose((got - b) / h, flux, rtol=1e-5)


def test_the_step_conserves_biomass_and_keeps_it_non_negative():
    grid = _uneven_grid()
    b = np.array([0.0, 5.0, 0.0])
    for rate in (0.1, 10.0, 1e6):
        got = implicit_diffusion_step(b, rate, grid, grid.adjacency_matrix, 1 / 12)
        assert got.sum() == pytest.approx(b.sum(), rel=1e-9)  # round-off at 1e6
        assert np.all(got >= 0)


def test_a_very_fast_step_levels_the_density():
    # Patches at the same density exchange nothing, whatever their areas.
    grid = _uneven_grid()
    b = np.array([7.0, 0.0, 0.0])
    got = implicit_diffusion_step(b, 1e9, grid, grid.adjacency_matrix, 1 / 12)
    np.testing.assert_allclose(got / grid.patch_areas, 1.0, rtol=1e-6)


def test_no_rate_or_no_time_leaves_biomass_alone():
    grid = _uneven_grid()
    b = np.array([3.0, 1.0, 0.5])
    np.testing.assert_array_equal(
        implicit_diffusion_step(b, 0.0, grid, grid.adjacency_matrix, 1 / 12), b
    )
    np.testing.assert_array_equal(
        implicit_diffusion_step(b, 5.0, grid, grid.adjacency_matrix, 0.0), b
    )


# --- in a run ---


def _scenario(rule):
    """Fish, Plankton (fast under the EwE rule), Detritus and a fleet."""
    params = create_rpath_params(
        ["Fish", "Plankton", "Detritus", "Fleet"], [0, 1, 2, 3]
    )
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [10.0, 1.0, 5.0, 0.8]
    params.model.loc[1, ["Biomass", "PB", "EE"]] = [5.0, 50.0, 0.6]
    params.model.loc[2, "Biomass"] = 1.0
    params.diet.iloc[1, 1] = 1.0
    params.model.loc[0, "Fleet"] = 0.5
    return rsim_scenario(rpath(params), params, years=range(1, 4), fast_group_rule=rule)


@pytest.fixture(scope="module", params=["ewe", "rpath"])
def scenario(request):
    return _scenario(request.param)


def _ecospace(rates, advection=False, gravity=0.0):
    grid = create_regular_grid(bounds=(0, 0, 5, 5), nx=5, ny=5)
    c = grid.patch_centroids
    habitat = 0.1 + 0.9 * (c[:, 0] - c[:, 0].min()) / np.ptp(c[:, 0])
    n = len(rates)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.tile(habitat, (n, 1)),
        habitat_capacity=np.tile(habitat, (n, 1)),
        dispersal_rate=np.asarray(rates, dtype=float),
        advection_enabled=np.full(n, advection),
        gravity_strength=np.full(n, gravity),
    )


def test_fast_movement_stays_finite_and_conserves_biomass(scenario):
    # 200 per year per cell side, as EwE's rate on 0.5 km cells: far beyond
    # what an explicit monthly RK4 step keeps stable.
    eco = _ecospace([200.0, 200.0, 200.0, 0.0])
    out = rsim_run_spatial(scenario, ecospace=eco)
    spatial = out.out_Biomass_spatial
    assert np.all(np.isfinite(spatial)) and np.all(spatial >= 0)
    np.testing.assert_allclose(spatial.sum(axis=2), out.out_Biomass, rtol=1e-9)


def test_explicit_movement_at_that_rate_blows_up(scenario):
    # Not finite, or (clipped at zero each step) grown past any plausible size
    eco = _ecospace([200.0, 200.0, 200.0, 0.0])
    out = rsim_run_spatial(scenario, ecospace=eco, implicit_movement=False)
    spatial = out.out_Biomass_spatial
    assert not (np.all(np.isfinite(spatial)) and np.nanmax(spatial) < 1e6)


def test_a_fast_groups_rate_still_changes_nothing():
    # Under the EwE rule a fast group's biomass is its local equilibrium each
    # step, so dispersal never reached it; the implicit step keeps that.
    sc = _scenario("ewe")
    base = rsim_run_spatial(sc, ecospace=_ecospace([2.0, 2.0, 2.0, 0.0]))
    alt = rsim_run_spatial(sc, ecospace=_ecospace([2.0, 0.0, 2.0, 0.0]))
    np.testing.assert_array_equal(base.out_Biomass_spatial, alt.out_Biomass_spatial)


def test_explicit_movement_beyond_its_stable_rate_is_named(scenario, caplog):
    eco = _ecospace([50.0, 0.0, 0.0, 0.0])
    with caplog.at_level(logging.WARNING, logger="pypath.spatial.integration"):
        rsim_run_spatial(scenario, ecospace=eco, implicit_movement=False)
    assert "Fish" in caplog.text and "implicit_movement" in caplog.text


def test_implicit_movement_gives_no_stability_warning(scenario, caplog):
    eco = _ecospace([50.0, 0.0, 0.0, 0.0])
    with caplog.at_level(logging.WARNING, logger="pypath.spatial.integration"):
        rsim_run_spatial(scenario, ecospace=eco)
    assert "implicit_movement" not in caplog.text


def test_slow_movement_matches_explicit_diffusion(scenario):
    # At 0.1 per year the split, implicit step and the explicit RK4 term
    # differ by under 0.3 % of each group's mean (measured: the gap falls
    # about 8 times from 0.1 to 0.01 per year, as a first-order split does);
    # a step 12 times too long (months for years) is off by several percent.
    eco = _ecospace([0.1, 0.1, 0.1, 0.0])
    implicit = rsim_run_spatial(scenario, ecospace=eco).out_Biomass_spatial
    explicit = rsim_run_spatial(
        scenario, ecospace=eco, implicit_movement=False
    ).out_Biomass_spatial
    for row in (1, 2, 3):
        gap = np.max(np.abs(implicit[:, row] - explicit[:, row]))
        assert gap / np.mean(explicit[:, row]) < 0.01, row
