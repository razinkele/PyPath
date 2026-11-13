"""EwE's capacity-weighted movement (open item B2).

EwE moves biomass across the face between two water cells at Mrate, slowed
by the habitat ratio when it moves into poorer habitat: the rate from cell x
to cell y is ``Mrate x min(1, HabCap_y / HabCap_x)`` (SetMovementParameters,
cEcoSpace.vb 3804-3815). Density in proportion to the habitat capacity is
therefore at rest, and movement leads towards it; plain diffusion levels
the density instead. A cell without habitat is left at the full rate and
never entered.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_regular_grid, rsim_run_spatial
from pypath.spatial.dispersal import diffusion_flux, implicit_diffusion_step
from pypath.spatial.ecospace_params import EcospaceGrid

ORACLE = Path(__file__).parent / "data" / "ewe_oracle" / "blacksea"
CAP = np.array([1.0, 0.5, 0.25])


def _row_of_cells(n=3, areas=None):
    """``n`` cells in a row; equal cells unless ``areas`` is given."""
    pairs = [(i, i + 1) for i in range(n - 1)]
    rows = [a for a, b in pairs] + [b for a, b in pairs]
    cols = [b for a, b in pairs] + [a for a, b in pairs]
    adjacency = scipy.sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    )
    return EcospaceGrid(
        n_patches=n,
        patch_ids=np.arange(n),
        patch_areas=np.ones(n) if areas is None else np.asarray(areas, float),
        patch_centroids=np.column_stack([0.1 * np.arange(n), np.zeros(n)]),
        adjacency_matrix=adjacency,
        edge_lengths={pair: 1.0 for pair in pairs},
    )


# --- the movement terms ---


def test_density_in_proportion_to_capacity_does_not_move():
    grid = _row_of_cells()
    b = 2.0 * CAP
    flux = diffusion_flux(b, 3.0, grid, grid.adjacency_matrix, capacity=CAP)
    np.testing.assert_allclose(flux, 0.0, atol=1e-12)
    got = implicit_diffusion_step(
        b, 3.0, grid, grid.adjacency_matrix, 1 / 12, capacity=CAP
    )
    np.testing.assert_allclose(got, b, rtol=1e-12)


def test_movement_leads_towards_the_capacity_pattern():
    grid = _row_of_cells(areas=[1.0, 2.0, 4.0])
    b = np.array([1.0, 1.0, 1.0])
    got = implicit_diffusion_step(
        b, 1e6, grid, grid.adjacency_matrix, 1.0, capacity=CAP
    )
    density = got / grid.patch_areas
    np.testing.assert_allclose(density / density[0], CAP / CAP[0], rtol=1e-5)
    assert got.sum() == pytest.approx(b.sum(), rel=1e-9)


def test_moving_into_poorer_habitat_is_slowed_by_the_habitat_ratio():
    # two cells, capacity 1 and 0.25: out of the good cell at 0.25 x the
    # plain rate, out of the poor cell at the plain rate
    grid = _row_of_cells(2)
    cap = np.array([1.0, 0.25])
    adj = grid.adjacency_matrix
    for b, factor in (([1.0, 0.0], 0.25), ([0.0, 1.0], 1.0)):
        plain = diffusion_flux(np.array(b), 2.0, grid, adj)
        got = diffusion_flux(np.array(b), 2.0, grid, adj, capacity=cap)
        np.testing.assert_allclose(got, factor * plain, rtol=1e-12)


def test_a_cell_without_habitat_is_left_and_never_entered():
    grid = _row_of_cells()
    cap = np.array([1.0, 0.0, 1.0])
    b = np.ones(3)
    flux = diffusion_flux(b, 2.0, grid, grid.adjacency_matrix, capacity=cap)
    assert flux[1] < 0 and flux[0] > 0 and flux[2] > 0
    got = implicit_diffusion_step(
        b, 1e9, grid, grid.adjacency_matrix, 1.0, capacity=cap
    )
    assert got[1] == pytest.approx(0.0, abs=1e-6)
    assert got.sum() == pytest.approx(3.0, rel=1e-9)


def test_no_or_uniform_capacity_is_plain_diffusion():
    grid = _row_of_cells(areas=[1.0, 2.0, 4.0])
    adj = grid.adjacency_matrix
    b = np.array([3.0, 1.0, 0.5])
    plain = diffusion_flux(b, 7.0, grid, adj)
    assert np.array_equal(diffusion_flux(b, 7.0, grid, adj, capacity=None), plain)
    np.testing.assert_allclose(
        diffusion_flux(b, 7.0, grid, adj, capacity=np.full(3, 0.4)), plain, rtol=1e-12
    )
    step = implicit_diffusion_step(b, 7.0, grid, adj, 0.1)
    np.testing.assert_allclose(
        implicit_diffusion_step(b, 7.0, grid, adj, 0.1, capacity=np.full(3, 0.4)),
        step,
        rtol=1e-12,
    )


def test_a_tiny_implicit_step_is_the_explicit_capacity_flux():
    grid = _row_of_cells(areas=[1.0, 2.0, 4.0])
    b = np.array([3.0, 1.0, 0.5])
    h, rate = 1e-7, 5.0
    got = implicit_diffusion_step(b, rate, grid, grid.adjacency_matrix, h, capacity=CAP)
    flux = diffusion_flux(b, rate, grid, grid.adjacency_matrix, capacity=CAP)
    np.testing.assert_allclose((got - b) / h, flux, rtol=1e-5)
    assert flux.sum() == pytest.approx(0.0, abs=1e-12)


# --- in a run ---


def _scenario():
    params = create_rpath_params(
        ["Fish", "Plankton", "Detritus", "Fleet"], [0, 1, 2, 3]
    )
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [10.0, 1.0, 5.0, 0.8]
    params.model.loc[1, ["Biomass", "PB", "EE"]] = [5.0, 50.0, 0.6]
    params.model.loc[2, "Biomass"] = 1.0
    params.diet.iloc[1, 1] = 1.0
    params.model.loc[0, "Fleet"] = 0.5
    return rsim_scenario(rpath(params), params, years=range(1, 3))


def test_fast_movement_holds_a_group_in_proportion_to_its_capacity():
    # moving fast, Fish settles where EwE's movement is at rest: density in
    # proportion to the habitat capacity (plain diffusion would level it)
    grid = create_regular_grid(bounds=(0, 0, 5, 1), nx=5, ny=1)
    habitat = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
    eco = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((4, 5)),
        habitat_capacity=np.tile(habitat, (4, 1)),
        dispersal_rate=np.array([1e4, 0.0, 0.0, 0.0]),
        advection_enabled=np.zeros(4, dtype=bool),
        gravity_strength=np.zeros(4),
    )
    spatial = rsim_run_spatial(_scenario(), ecospace=eco).out_Biomass_spatial
    fish = spatial[-1, 1] / spatial[-1, 1].sum()
    np.testing.assert_allclose(fish, habitat / habitat.sum(), rtol=1e-3)


# --- against EwE's Ecospace (Black Sea, oracle run 43288513485) ---


@pytest.mark.integration
def test_black_sea_cell_maps_follow_ewes():
    from pypath.io.ewemdb import (
        check_ewemdb_support,
        ecosim_scenario_from_ewemdb,
        read_ecospace,
    )

    db = (
        Path(__file__).resolve().parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"
    )
    if not db.exists() or not any(check_ewemdb_support().values()):
        pytest.skip("Black Sea database or driver missing")
    scenario = ecosim_scenario_from_ewemdb(str(db), scenario=1)
    names = [str(n).strip() for n in scenario.params.spname]
    res = read_ecospace(
        str(db),
        n_groups=scenario.params.NUM_LIVING + scenario.params.NUM_DEAD,
        scenario_id=1,
    )
    e = res.ecospace
    n_rows, n_p = scenario.params.NUM_GROUPS, e.grid.n_patches
    hab = np.ones((n_rows, n_p))
    hab[: e.habitat_capacity.shape[0]] = e.habitat_capacity
    rates = np.zeros(n_rows)
    rates[: len(e.dispersal_rate)] = e.dispersal_rate
    eco = EcospaceParams(
        grid=e.grid,
        habitat_preference=np.ones((n_rows, n_p)),
        habitat_capacity=hab,
        dispersal_rate=rates,
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
    )
    bs = rsim_run_spatial(scenario, ecospace=eco, years=range(1, 2)).out_Biomass_spatial
    density = bs / (e.grid.patch_areas / e.grid.patch_areas.sum())[None, None, :]
    meta = e.grid.cell_metadata
    cells = pd.DataFrame(
        {"Row": meta["row"] + 1, "Col": meta["col"] + 1, "p": range(n_p)}
    )
    maps = pd.read_csv(ORACLE / "ecospace_maps.csv")
    distances = {}
    for month in (1, 12):
        mm = cells.merge(maps[maps.TimeStep == month], on=["Row", "Col"])
        assert len(mm) == n_p
        for group in maps.columns[3:]:
            py = density[month, names.index(group), mm.p.to_numpy()]
            ewe = mm[group].to_numpy()
            distances[month, group] = (
                0.5 * np.abs(py / py.sum() - ewe / ewe.sum()).sum()
            )
    # total variation distance between each group's distribution over the
    # cells and EwE's, median over the groups. Plain diffusion gave 0.111 at
    # month 1 and 0.266 at month 12; EwE's movement gives 0.008 and 0.034.
    # (Phytoplankton, a fast group that is not moved, stays at 0.13-0.16.)
    for month, limit in ((1, 0.03), (12, 0.06)):
        median = np.median([d for (m, _), d in distances.items() if m == month])
        assert median < limit, (month, median)
