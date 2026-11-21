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


def test_exchange_only_is_the_default_and_keeps_the_total_off_the_rest_state():
    """A river of 30 m3/s runs 10 -> 20 -> 30: only the two-way water counts."""
    setup = _ring()  # 50 forward, 20 back
    setup["flow"][:, 0, 1] += 30.0
    setup["flow"][:, 1, 2] += 30.0
    transport = WaterTransport(groups=[0], **setup)
    assert transport.throughflow is False
    volume = transport.volume[0]
    a = transport.rate_matrix(0, volume)
    np.testing.assert_allclose(a.sum(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(a @ volume, 0.0, atol=1e-6)
    assert a[1, 0] == pytest.approx(20.0 / 1e8 * YEAR)  # min(80, 20)
    totals = np.array([5.0, 1.0, 0.0])  # far from the rest state
    for month in range(12):
        totals = transport.step_zones(totals, month, volume, 1 / 12)
        assert totals.sum() == pytest.approx(6.0, rel=1e-12)


def test_the_rate_matrix_conserves_and_rests_at_the_volumes():
    transport = WaterTransport(groups=[0], throughflow=True, **_ring())
    a = transport.rate_matrix(0, transport.volume[0])
    np.testing.assert_allclose(a.sum(axis=0), 0.0, atol=1e-9)  # nothing is lost
    np.testing.assert_allclose(a @ transport.volume[0], 0.0, atol=1e-6)  # at rest
    # a zone's water is replaced at (out flow) / volume
    assert -a[0, 0] == pytest.approx(70.0 / 1e8 * YEAR)
    assert a[1, 0] == pytest.approx(50.0 / 1e8 * YEAR)


@pytest.mark.parametrize("throughflow", [False, True])
def test_a_release_spreads_to_one_concentration_and_keeps_the_total(throughflow):
    transport = WaterTransport(groups=[0], throughflow=throughflow, **_ring())
    totals = np.array([7.0, 0.0, 0.0])
    for _ in range(36):
        totals = transport.step_zones(totals, 0, transport.volume[0], 1 / 12)
    assert totals.sum() == pytest.approx(7.0, rel=1e-12)
    np.testing.assert_allclose(totals / transport.volume[0], 1e-8, rtol=1e-6)


def test_one_implicit_month_is_stable_for_flows_that_flush_a_zone_in_hours():
    fast = _ring(q=5e4, back=4e4)  # 1e8 m3 flushed in about twenty minutes
    transport = WaterTransport(groups=[0], throughflow=True, **fast)
    totals = transport.step_zones(
        np.array([7.0, 0.0, 0.0]), 0, fast["volume"][0], 1 / 12
    )
    assert np.all(totals >= 0) and totals.sum() == pytest.approx(7.0)
    # one month is hundreds of flushing times: mixed to within half a percent
    np.testing.assert_allclose(totals / fast["volume"][0], 1e-8, rtol=5e-3)


@pytest.mark.parametrize("throughflow", [False, True])
def test_weights_other_than_volume_set_the_rest_state(throughflow):
    transport = WaterTransport(groups=[0], throughflow=throughflow, **_ring())
    weights = np.array([3.0, 1.0, 2.0])
    a = transport.rate_matrix(0, weights)
    np.testing.assert_allclose(a @ weights, 0.0, atol=1e-9)
    np.testing.assert_allclose(a.sum(axis=0), 0.0, atol=1e-9)
    # a zone of weight 0 holds none of the group and takes none
    without = np.array([3.0, 0.0, 2.0])
    dead = transport.rate_matrix(0, without)
    assert np.all(dead[1] == 0) and np.all(dead[:, 1] == 0)
    np.testing.assert_allclose(dead @ without, 0.0, atol=1e-9)  # still at rest


def _estuary(**kwargs):
    """A river of 30 m3/s into zone 10; zone 30 exchanges with the sea."""
    setup = _ring()
    setup["flow"][:, 0, 1] += 30.0  # the river water runs 10 -> 20 -> 30
    setup["flow"][:, 1, 2] += 30.0
    return WaterTransport(
        groups=[0],
        boundary_id=np.array([-2, -1]),
        boundary_zone=np.array([0, 2]),
        boundary_in=np.tile([30.0, 100.0], (12, 1)),
        boundary_out=np.tile([0.0, 130.0], (12, 1)),
        **setup,
        **kwargs,
    )


def test_with_throughflow_a_river_dilutes_and_the_sea_boundary_does_not():
    estuary = _estuary(throughflow=True)
    assert estuary.open_boundaries == [-2]  # the boundary with a net inflow
    volume = estuary.volume[0]
    a = estuary.rate_matrix(0, volume)
    # at one concentration: the river zone is diluted at river flow / volume,
    # the others, the sea zone included, are at rest
    assert (a @ volume)[0] == pytest.approx(-30.0 / 1e8 * YEAR * volume[0])
    assert (a @ volume)[1:] == pytest.approx([0.0, 0.0], abs=1e-6)
    # a tracer is flushed out: less every month, never negative
    totals, last = volume.copy(), volume.sum()
    for month in range(12):
        totals = estuary.step_zones(totals, month, volume, 1 / 12)
        assert 0 < totals.sum() < last
        last = totals.sum()


def test_with_throughflow_the_rest_state_holds_when_no_boundary_dilutes():
    whole = _estuary(throughflow=True, open_boundaries=[])
    volume = whole.volume[0]
    np.testing.assert_allclose(whole.rate_matrix(0, volume) @ volume, 0.0, atol=1e-6)
    # also when the zones' water does not quite balance (storage, rounding)
    whole.flow[:, 1, 2] *= 1.02
    for weights in (volume, np.array([3.0, 1.0, 2.0])):
        np.testing.assert_allclose(
            whole.rate_matrix(0, weights) @ weights, 0.0, atol=1e-6
        )
    with pytest.raises(ValueError, match="not boundary ids"):
        _estuary(throughflow=True, open_boundaries=[-9])


def test_exchange_only_ignores_the_boundaries():
    closed = _estuary()
    volume = closed.volume[0]
    a = closed.rate_matrix(0, volume)
    np.testing.assert_allclose(a.sum(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(a @ volume, 0.0, atol=1e-6)


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
    transport = WaterTransport.from_shyfem_box_series(
        _series(), groups=[1], throughflow=True, open_boundaries=[]
    )
    assert WaterTransport.from_shyfem_box_series(
        _series(), groups=[1], throughflow=True
    ).open_boundaries == [-2]  # the river
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


def test_zones_the_grid_does_not_cover_are_outside_the_map(
    integrated_scenario,
):
    """Zones 10 and 20 on the map, zone 30 of the flows outside it."""
    scenario, _ = integrated_scenario
    grid, _ = _zone_grid(sizes=(1, 2), ids=(10, 20))
    years = range(1, 2)
    plain = rsim_run_spatial(scenario, ecospace=_ecospace(scenario, grid), years=years)
    for rest in ("volume", "capacity"):
        transport = WaterTransport(
            groups=[3], rest=rest, **_ring(12, q=2000.0, back=1500.0)
        )
        moved = rsim_run_spatial(
            scenario, ecospace=_ecospace(scenario, grid, transport), years=years
        )
        det = 4
        assert np.isfinite(moved.out_Biomass_spatial).all()
    # rest="capacity", a uniform map: at rest, so the total is the plain run's
    assert moved.out_Biomass_spatial[1, det].sum() == pytest.approx(
        plain.out_Biomass_spatial[1, det].sum(), rel=1e-9
    )
    # at the capacity rest state (density by area: 1 : 2) nothing has moved
    share = moved.out_Biomass_spatial[1, det] / moved.out_Biomass_spatial[1, det].sum()
    np.testing.assert_allclose(share, [1 / 3, 2 / 3], atol=1e-12)
    # by volume the two zones on the map end at 3 : 1, whatever zone 30 holds
    volume = WaterTransport(
        groups=[3], **_ring(12, q=2000.0, back=1500.0, volumes=(3e8, 1e8, 4e8))
    )
    out = rsim_run_spatial(
        scenario, ecospace=_ecospace(scenario, grid, volume), years=years
    ).out_Biomass_spatial
    np.testing.assert_allclose(out[1, det] / out[1, det].sum(), [0.75, 0.25], rtol=0.05)
    # exchange only: the map's total is kept on the way there
    assert out[1, det].sum() == pytest.approx(plain.out_Biomass_spatial[1, det].sum())


def test_stanza_and_ibm_groups_are_refused(integrated_scenario):
    scenario, _ = integrated_scenario
    grid, _ = _zone_grid()
    ecospace = _ecospace(scenario, grid)
    names = [str(n) for n in scenario.params.spname]
    transport = WaterTransport(groups=[1, 2], **_ring(12))
    assert transport.prepare(ecospace, 12) == [2, 3]
    with pytest.raises(ValueError, match="multi-stanza group"):
        transport.prepare(ecospace, 12, stanza_rows=[3], names=names)
    with pytest.raises(ValueError, match="an IBM group") as error:
        transport.prepare(ecospace, 12, ibm_rows=[2], names=names)
    assert names[2] in str(error.value)
    with pytest.raises(ValueError, match="outside 0.."):
        WaterTransport(groups=[99], **_ring(12)).prepare(ecospace, 12)


def test_biomass_that_is_not_finite_is_passed_through():
    """A run that has already failed is not turned into a solver error."""
    transport = WaterTransport(groups=[0], **_ring())
    broken = np.array([1.0, np.nan, 2.0])
    out = transport.step_zones(broken, 0, transport.volume[0], 1 / 12)
    assert np.isnan(out[1]) and out[0] == 1.0 and out[2] == 2.0


def test_a_river_with_some_backflow_still_dilutes():
    """River: 32 in and 2 out. A tracer from uniform density must not grow."""
    estuary = _estuary(throughflow=True)
    estuary.boundary_in[:, 0], estuary.boundary_out[:, 0] = 32.0, 2.0
    again = WaterTransport(
        groups=[0],
        throughflow=True,
        zones=estuary.zones,
        flow=estuary.flow,
        volume=estuary.volume,
        boundary_id=estuary.boundary_id,
        boundary_zone=estuary.boundary_zone,
        boundary_in=estuary.boundary_in,
        boundary_out=estuary.boundary_out,
    )
    assert again.open_boundaries == [-2]
    volume = again.volume[0]
    a = again.rate_matrix(0, volume)
    assert (a @ volume)[0] == pytest.approx(-30.0 / 1e8 * YEAR * volume[0])
    totals = np.array([1.0, 1.0, 1.0])  # not the rest state (volumes 1 : 2 : 4)
    start = totals.sum()
    for month in range(24):
        totals = again.step_zones(totals, month % 12, volume, 1 / 12)
        assert totals.sum() <= start * (1 + 1e-12)
    assert totals.sum() < 0.9 * start


def test_a_group_named_twice_is_carried_once(integrated_scenario):
    scenario, _ = integrated_scenario
    grid, _ = _zone_grid()
    transport = WaterTransport(groups=[3, 1, 3], **_ring(12))
    assert transport.groups == [3, 1]
    assert transport.prepare(_ecospace(scenario, grid), 12) == [4, 2]


def test_a_boundary_with_several_sections_is_judged_as_one():
    """Boundary -1: 100 in through one section, 120 out through another."""
    setup = _ring()
    sea = WaterTransport(
        groups=[0],
        throughflow=True,
        boundary_id=np.array([-1, -1, -2]),
        boundary_zone=np.array([2, 1, 0]),
        boundary_in=np.tile([100.0, 0.0, 20.0], (12, 1)),
        boundary_out=np.tile([0.0, 120.0, 0.0], (12, 1)),
        **setup,
    )
    assert sea.open_boundaries == [-2]  # the river only: -1 takes more than it brings
    explicit = WaterTransport(
        groups=[0],
        throughflow=True,
        open_boundaries=[-1],
        boundary_id=np.array([-1, -1, -2]),
        boundary_zone=np.array([2, 1, 0]),
        boundary_in=np.tile([100.0, 0.0, 20.0], (12, 1)),
        boundary_out=np.tile([0.0, 120.0, 0.0], (12, 1)),
        **setup,
    )
    assert explicit.open_boundaries == [-1]


def test_a_group_beyond_the_runs_groups_is_refused(integrated_scenario):
    """EcospaceParams may carry more rows than the scenario has groups."""
    scenario, _ = integrated_scenario
    grid, _ = _zone_grid()
    ecospace = _ecospace(scenario, grid)  # NUM_GROUPS + 1 rows
    n = scenario.params.NUM_GROUPS
    extra = WaterTransport(groups=[n], **_ring(12))
    with pytest.raises(ValueError, match=f"outside 0..{n - 1}"):
        extra.prepare(ecospace, 12, n_groups=n)
    ecospace.water_transport = extra
    with pytest.raises(ValueError, match=f"outside 0..{n - 1}"):
        rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 2))


def test_water_transport_is_the_last_field_of_ecospace_params():
    """Positional arguments of EcospaceParams keep their places."""
    import dataclasses

    names = [f.name for f in dataclasses.fields(EcospaceParams)]
    assert names[-1] == "water_transport"
    assert names.index("external_flux") + 1 == names.index("environmental_drivers")


def test_with_diffusion_too_the_capacity_rest_state_changes_nothing(
    integrated_scenario,
):
    """Dispersal above 0, a uniform map, exchange only, rest="capacity"."""
    scenario, _ = integrated_scenario
    grid, _ = _zone_grid()
    years = range(1, 3)
    plain = rsim_run_spatial(
        scenario, ecospace=_ecospace(scenario, grid, dispersal=5.0), years=years
    )
    transport = WaterTransport(groups=[0, 1, 3], rest="capacity", **_ring(24))
    moved = rsim_run_spatial(
        scenario,
        ecospace=_ecospace(scenario, grid, transport, dispersal=5.0),
        years=years,
    )
    np.testing.assert_allclose(
        moved.out_Biomass_spatial, plain.out_Biomass_spatial, rtol=1e-10, atol=1e-13
    )


def test_a_fast_group_can_be_integrated_and_carried_while_others_stay_fast(
    spatial_scenario,
):
    """Under the EwE rule: integrate one fast group and carry it."""
    scenario, _ = spatial_scenario
    grid, _ = _zone_grid()
    flags = np.asarray(scenario.params.NoIntegrate).copy()
    fast = int(np.flatnonzero(flags[1:] > 0)[0])  # a group row
    other = 1 if fast != 1 else 2

    integrated = copy.deepcopy(scenario)
    changed = flags.copy()
    changed[fast + 1] = 0
    integrated.params.NoIntegrate = changed
    carried = WaterTransport(groups=[fast], rest="capacity", **_ring(12))
    out = rsim_run_spatial(
        integrated, ecospace=_ecospace(integrated, grid, carried), years=range(1, 2)
    )
    assert np.isfinite(out.out_Biomass_spatial).all()

    # another group made fast is refused, the integrated one still accepted
    mixed = copy.deepcopy(integrated)
    more = changed.copy()
    more[other + 1] = 1
    mixed.params.NoIntegrate = more
    both = WaterTransport(groups=[fast, other], **_ring(12))
    with pytest.raises(ValueError, match="fast-equilibrium group") as error:
        rsim_run_spatial(
            mixed, ecospace=_ecospace(mixed, grid, both), years=range(1, 2)
        )
    assert str(mixed.params.spname[other + 1]) in str(error.value)


def test_exchange_evens_out_a_prey_that_its_grazer_leaves_uneven(
    integrated_scenario,
):
    """Where it must act: the grazer's habitat differs between the zones.

    Without transport the prey's density then differs from zone to zone.
    Exchange only with rest="capacity" (the prey's own capacity is uniform)
    mixes it: the spread across the zones shrinks.
    """
    scenario, _ = integrated_scenario
    scenario = copy.deepcopy(scenario)
    # the producer turns over 200 times a year: integrated, it needs sub-steps
    scenario.params.RK4_STEPS = 16
    grid, _ = _zone_grid()
    n_rows = scenario.params.NUM_GROUPS + 1
    capacity = np.ones((n_rows, grid.n_patches))
    capacity[1] = [0.4, 1.0, 1.6]  # the grazer (Zoo)
    years = range(1, 4)
    prey = 1  # state row of the producer, group row 0

    def density(transport):
        out = rsim_run_spatial(
            scenario,
            ecospace=_ecospace(scenario, grid, transport, capacity=capacity),
            years=years,
        )
        return out.out_Biomass_spatial[:, prey, :] / grid.patch_areas

    def spread(d):
        return d.max(axis=1) / d.min(axis=1) - 1.0

    alone = density(None)
    mixed = density(
        WaterTransport(groups=[0], rest="capacity", **_ring(36, q=2000.0, back=1500.0))
    )
    assert np.isfinite(mixed).all()
    assert spread(alone)[-1] > 0.02  # the grazer does leave the prey uneven
    assert spread(mixed)[-1] < 0.25 * spread(alone)[-1]
    assert np.all(spread(mixed)[1:] < spread(alone)[1:])
