"""Water flows between zones carrying biomass (pypath.spatial.transport).

Flows here are made by the tests; no hydrodynamic model output is used.
"""

import copy

import numpy as np
import pytest

from pypath.io.shyfem import ShyfemBoxSeries
from pypath.spatial import (
    EcospaceGrid,
    EcospaceParams,
    WaterTransport,
    rsim_run_spatial,
)
from pypath.spatial.mesh import grid_from_mesh

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detritus fate of a detritus group to itself:UserWarning"
)

YEAR = 365.25 * 86400.0


def _ring(n_months=12, q=50.0, back=20.0, volumes=(1e8, 2e8, 4e8)):
    """Three zones 10, 20, 30 in a ring: q m3/s forward, ``back`` backward."""
    flow = np.zeros((n_months, 3, 3))
    for i in range(3):
        flow[:, i, (i + 1) % 3] = q
        flow[:, (i + 1) % 3, i] = back
    volume = np.tile(np.asarray(volumes, dtype=float), (n_months, 1))
    return dict(zones=np.array([10, 20, 30]), flow=flow, volume=volume)


def _zone_grid(sizes=(1, 2, 3), ids=(10, 20, 30), side=2000.0):
    """Squares in a row grouped into zones of ``sizes`` squares each."""
    n = sum(sizes)
    xy = np.array([(ix * side, iy * side) for iy in range(2) for ix in range(n + 1)])
    quads = np.array([[ix, ix + 1, n + 2 + ix, n + 1 + ix] for ix in range(n)])
    zones = np.repeat(ids, sizes)
    grid = grid_from_mesh(xy, quads, zones)
    grid.cell_metadata["type"] = grid.patch_ids
    cells = grid_from_mesh(xy, quads)
    cells.cell_metadata["type"] = zones
    return grid, cells


# --- the operator ---------------------------------------------------------


def test_the_rate_matrix_conserves_and_rests_at_the_volumes():
    transport = WaterTransport(groups=[0], **_ring())
    a = transport.rate_matrix(0, transport.volume[0])
    np.testing.assert_allclose(a.sum(axis=0), 0.0, atol=1e-9)  # nothing is lost
    np.testing.assert_allclose(a @ transport.volume[0], 0.0, atol=1e-6)  # at rest
    # a zone's water is replaced at (out flow) / volume
    assert -a[0, 0] == pytest.approx(70.0 / 1e8 * YEAR)
    assert a[1, 0] == pytest.approx(50.0 / 1e8 * YEAR)


def test_a_release_spreads_to_one_concentration_and_keeps_the_total():
    transport = WaterTransport(groups=[0], **_ring())
    totals = np.array([7.0, 0.0, 0.0])
    for _ in range(36):
        totals = transport.step_zones(totals, 0, transport.volume[0], 1 / 12)
    assert totals.sum() == pytest.approx(7.0, rel=1e-12)
    np.testing.assert_allclose(totals / transport.volume[0], 1e-8, rtol=1e-6)


def test_one_implicit_month_is_stable_for_flows_that_flush_a_zone_in_hours():
    fast = _ring(q=5e4, back=4e4)  # 1e8 m3 flushed in about twenty minutes
    transport = WaterTransport(groups=[0], **fast)
    totals = transport.step_zones(
        np.array([7.0, 0.0, 0.0]), 0, fast["volume"][0], 1 / 12
    )
    assert np.all(totals >= 0) and totals.sum() == pytest.approx(7.0)
    # one month is hundreds of flushing times: mixed to within half a percent
    np.testing.assert_allclose(totals / fast["volume"][0], 1e-8, rtol=5e-3)


def test_weights_other_than_volume_set_the_rest_state():
    transport = WaterTransport(groups=[0], **_ring())
    weights = np.array([3.0, 1.0, 2.0])
    a = transport.rate_matrix(0, weights)
    np.testing.assert_allclose(a @ weights, 0.0, atol=1e-9)
    np.testing.assert_allclose(a.sum(axis=0), 0.0, atol=1e-9)
    # a zone of weight 0 holds none of the group and takes none
    without = np.array([3.0, 0.0, 2.0])
    dead = transport.rate_matrix(0, without)
    assert np.all(dead[1] == 0) and np.all(dead[:, 1] == 0)
    np.testing.assert_allclose(dead @ without, 0.0, atol=1e-9)  # still at rest


def test_a_closed_boundary_keeps_the_rest_state_an_open_one_takes_biomass():
    """A river of 30 m3/s into zone 10 that leaves to the sea from zone 30."""
    setup = _ring()
    setup["flow"][:, 0, 1] += 30.0  # the river water runs 10 -> 20 -> 30
    setup["flow"][:, 1, 2] += 30.0
    boundaries = dict(
        boundary_id=np.array([-2, -1]),
        boundary_zone=np.array([0, 2]),
        boundary_in=np.tile([30.0, 100.0], (12, 1)),
        boundary_out=np.tile([0.0, 130.0], (12, 1)),
    )
    closed = WaterTransport(groups=[0], **setup, **boundaries)
    volume = closed.volume[0]
    np.testing.assert_allclose(closed.rate_matrix(0, volume) @ volume, 0.0, atol=1e-6)
    # also when the zones' water does not quite balance (storage, rounding)
    rough = WaterTransport(groups=[0], **setup, **boundaries)
    rough.flow[:, 1, 2] *= 1.02
    for weights in (volume, np.array([3.0, 1.0, 2.0])):
        np.testing.assert_allclose(
            rough.rate_matrix(0, weights) @ weights, 0.0, atol=1e-6
        )

    river = WaterTransport(groups=[0], open_boundaries=[-2], **setup, **boundaries)
    a = river.rate_matrix(0, volume)
    # the river brings nothing: zone 10 is diluted at river flow / volume
    assert (a @ volume)[0] == pytest.approx(-30.0 / 1e8 * YEAR * volume[0])
    assert (a @ volume)[1:] == pytest.approx([0.0, 0.0], abs=1e-6)
    with pytest.raises(ValueError, match="not boundary ids"):
        WaterTransport(groups=[0], open_boundaries=[-9], **setup, **boundaries)


def test_bad_tables_are_refused():
    setup = _ring()
    with pytest.raises(ValueError, match="flow must be"):
        WaterTransport(groups=[0], **{**setup, "flow": setup["flow"][:, :2]})
    with pytest.raises(ValueError, match="not negative"):
        WaterTransport(groups=[0], **{**setup, "flow": -setup["flow"]})
    with pytest.raises(ValueError, match="positive"):
        WaterTransport(groups=[0], **{**setup, "volume": setup["volume"] * 0})
    with pytest.raises(ValueError, match="rest must be"):
        WaterTransport(groups=[0], rest="depth", **setup)
    with pytest.raises(ValueError, match="twelve months"):
        WaterTransport(groups=[0], beyond="repeat_year", **_ring(n_months=5))


def test_months_beyond_the_flows():
    refuse = WaterTransport(groups=[0], **_ring(n_months=14))
    assert refuse.month_index(13) == 13
    with pytest.raises(ValueError, match="outside the 14 months"):
        refuse.month_index(14)
    hold = WaterTransport(groups=[0], beyond="hold", **_ring(n_months=14))
    assert hold.month_index(500) == 13
    again = WaterTransport(groups=[0], beyond="repeat_year", **_ring(n_months=14))
    # the last twelve months (2..13) repeat: month 14 is like month 2
    assert [again.month_index(m) for m in (13, 14, 15, 25, 26)] == [13, 2, 3, 13, 2]


# --- from a SHYFEM box series ---------------------------------------------


def _series(n_months=3):
    """Monthly series: boxes 1, 2; section 1 -> 2 and a river into box 1."""
    times = np.array(
        [f"2001-{m + 1:02d}-01T00:00:00" for m in range(n_months)],
        dtype="datetime64[s]",
    )
    values = np.zeros((n_months, 2, 2))
    values[:, :, 0] = [1e8, 3e8]  # volume
    flow = np.zeros((n_months, 3, 3))
    flow[:, 0] = [10.0, 40.0, 30.0]  # 1 -> 2: 40 forward, 30 back
    flow[:, 1] = [10.0, 10.0, 0.0]  # river (-2) -> 1
    flow[:, 2] = [-10.0, 5.0, 15.0]  # sea (-1) -> 2: 5 in, 15 out
    return ShyfemBoxSeries(
        times=times,
        boxes=np.array([1, 2]),
        variables=["volume", "salt"],
        values=values,
        section_from=np.array([1, -2, -1]),
        section_to=np.array([2, 1, 2]),
        section_flow=flow,
        monthly=True,
    )


def test_flows_volumes_and_boundaries_from_a_box_series():
    transport = WaterTransport.from_shyfem_box_series(_series(), groups=[1])
    assert transport.zones.tolist() == [1, 2]
    np.testing.assert_allclose(transport.flow[0], [[0.0, 40.0], [30.0, 0.0]])
    np.testing.assert_allclose(transport.volume[2], [1e8, 3e8])
    assert transport.boundary_id.tolist() == [-2, -1]
    assert transport.boundary_zone.tolist() == [0, 1]
    np.testing.assert_allclose(transport.boundary_in[0], [10.0, 5.0])
    np.testing.assert_allclose(transport.boundary_out[0], [0.0, 15.0])
    # every zone's water balances, so the volumes are a rest state
    a = transport.rate_matrix(0, transport.volume[0])
    np.testing.assert_allclose(a @ transport.volume[0], 0.0, atol=1e-6)


def test_a_box_series_must_cover_the_months_from_the_start_date():
    series = _series(4)
    late = WaterTransport.from_shyfem_box_series(series, [0], start="2001-03-01")
    assert late.n_months == 2
    with pytest.raises(ValueError, match="starts 2 months after"):
        WaterTransport.from_shyfem_box_series(series, [0], start="2000-11-01")
    gap = copy.deepcopy(series)
    keep = [0, 1, 3]
    gap.times, gap.values = gap.times[keep], gap.values[keep]
    gap.section_flow = gap.section_flow[keep]
    with pytest.raises(ValueError, match="gaps"):
        WaterTransport.from_shyfem_box_series(gap, [0])
    empty = copy.deepcopy(series)
    empty.section_from = empty.section_to = np.empty(0, dtype=int)
    with pytest.raises(ValueError, match="no section flows"):
        WaterTransport.from_shyfem_box_series(empty, [0])


# --- in a spatial run -----------------------------------------------------


def _ecospace(scenario, grid, transport=None, capacity=None, dispersal=0.0):
    n_rows = scenario.params.NUM_GROUPS + 1
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n_rows, grid.n_patches)),
        habitat_capacity=(
            np.ones((n_rows, grid.n_patches)) if capacity is None else capacity
        ),
        dispersal_rate=np.full(n_rows, dispersal),
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
        water_transport=transport,
    )


@pytest.fixture
def integrated_scenario(spatial_scenario):
    """The fixture's scenario with every group integrated (no fast groups)."""
    scenario, params = spatial_scenario
    scenario = copy.deepcopy(scenario)
    scenario.params.NoIntegrate = np.zeros_like(scenario.params.NoIntegrate)
    return scenario, params


def test_at_the_capacity_rest_state_transport_changes_nothing(integrated_scenario):
    """Uniform map, rest="capacity": the run is the run without transport."""
    scenario, _ = integrated_scenario
    grid, _ = _zone_grid()
    years = range(1, 3)
    plain = rsim_run_spatial(scenario, ecospace=_ecospace(scenario, grid), years=years)
    transport = WaterTransport(groups=[0, 1, 3], rest="capacity", **_ring(24))
    moved = rsim_run_spatial(
        scenario, ecospace=_ecospace(scenario, grid, transport), years=years
    )
    np.testing.assert_allclose(
        moved.out_Biomass_spatial, plain.out_Biomass_spatial, rtol=1e-10, atol=1e-13
    )


def test_water_carries_a_group_towards_the_volumes_and_leaves_the_others(
    integrated_scenario,
):
    scenario, _ = integrated_scenario
    grid, _ = _zone_grid()
    years = range(1, 2)
    plain = rsim_run_spatial(scenario, ecospace=_ecospace(scenario, grid), years=years)
    # detritus, in water exchanged a few hundred times a year
    transport = WaterTransport(
        groups=[3], rest="volume", **_ring(12, q=2000.0, back=1500.0)
    )
    moved = rsim_run_spatial(
        scenario, ecospace=_ecospace(scenario, grid, transport), years=years
    )
    det = 4  # state row of group 3
    share = moved.out_Biomass_spatial[1, det] / moved.out_Biomass_spatial[1, det].sum()
    start = plain.out_Biomass_spatial[0, det] / plain.out_Biomass_spatial[0, det].sum()
    np.testing.assert_allclose(start, grid.patch_areas / grid.patch_areas.sum())
    # after one month the detritus lies as the water volumes do: 1 : 2 : 4
    np.testing.assert_allclose(share, np.array([1, 2, 4]) / 7.0, rtol=0.05)
    # the first month's local step is the same; only the carried row moved
    np.testing.assert_allclose(
        moved.out_Biomass_spatial[1, det].sum(), plain.out_Biomass_spatial[1, det].sum()
    )
    np.testing.assert_allclose(
        moved.out_Biomass_spatial[1, 3], plain.out_Biomass_spatial[1, 3]
    )


def test_cells_inside_zones_carry_the_zones_totals(integrated_scenario):
    """Flows are between zones: a grid of the cells inside them moves the same totals."""
    scenario, _ = integrated_scenario
    grid, cells = _zone_grid()
    years = range(1, 3)

    def run(g):
        transport = WaterTransport(groups=[1, 3], rest="volume", **_ring(24))
        out = rsim_run_spatial(
            scenario, ecospace=_ecospace(scenario, g, transport), years=years
        )
        return out.out_Biomass_spatial

    by_zone, by_cell = run(grid), run(cells)
    zone_of = np.repeat([0, 1, 2], [1, 2, 3])
    summed = np.zeros_like(by_zone)
    for cell in range(cells.n_patches):
        summed[:, :, zone_of[cell]] += by_cell[:, :, cell]
    np.testing.assert_allclose(summed, by_zone, rtol=1e-8, atol=1e-12)


def test_groups_a_run_does_not_integrate_are_refused(spatial_scenario):
    scenario, _ = spatial_scenario  # the EwE rule: producers and detritus are fast
    grid, _ = _zone_grid()
    names = [str(n) for n in scenario.params.spname]
    fast = int(np.flatnonzero(np.asarray(scenario.params.NoIntegrate)[1:] > 0)[0])
    transport = WaterTransport(groups=[fast], **_ring(12))
    with pytest.raises(ValueError) as error:
        rsim_run_spatial(
            scenario, ecospace=_ecospace(scenario, grid, transport), years=range(1, 2)
        )
    assert "fast-equilibrium group" in str(error.value)
    assert names[fast + 1] in str(error.value)
    assert 'fast_group_rule="rpath"' in str(error.value)


def test_a_run_longer_than_the_flows_and_a_grid_outside_them_are_refused(
    integrated_scenario,
):
    scenario, _ = integrated_scenario
    grid, _ = _zone_grid()
    short = WaterTransport(groups=[3], **_ring(12))
    with pytest.raises(ValueError, match="24 months, the water flows 12"):
        rsim_run_spatial(
            scenario, ecospace=_ecospace(scenario, grid, short), years=range(1, 3)
        )
    repeat = WaterTransport(groups=[3], beyond="repeat_year", **_ring(12))
    out = rsim_run_spatial(
        scenario, ecospace=_ecospace(scenario, grid, repeat), years=range(1, 3)
    )
    assert np.isfinite(out.out_Biomass_spatial).all()
    other, _ = _zone_grid(ids=(10, 20, 99))
    with pytest.raises(ValueError, match="no zone 99"):
        rsim_run_spatial(
            scenario,
            ecospace=_ecospace(
                scenario, other, WaterTransport(groups=[3], **_ring(12))
            ),
            years=range(1, 2),
        )
    regular = EcospaceGrid.from_regular_grid((0, 0, 3, 1), nx=3, ny=1)  # ids 0, 1, 2
    with pytest.raises(ValueError, match="no zone 0"):
        rsim_run_spatial(
            scenario,
            ecospace=_ecospace(
                scenario, regular, WaterTransport(groups=[3], **_ring(12))
            ),
            years=range(1, 2),
        )
