"""A habitat capacity for every month (pypath.spatial.monthly_capacity)."""

import numpy as np
import pytest

from pypath.spatial import (
    EcospaceParams,
    EnvironmentalLayer,
    MonthlyCapacity,
    capacity_from_layers,
    create_linear_response,
    rsim_run_spatial,
)
from pypath.spatial.mesh import grid_from_mesh
from pypath.spatial.mpa import MPAConfig, MPAZone

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detritus fate of a detritus group to itself:UserWarning"
)

YEARS = range(1, 3)
MONTHS = 24


def _grid(sizes=(1, 2, 3)):
    """Squares in a row grouped into zones of ``sizes`` squares; and the squares."""
    n = sum(sizes)
    xy = np.array(
        [(ix * 2000.0, iy * 2000.0) for iy in range(2) for ix in range(n + 1)]
    )
    quads = np.array([[ix, ix + 1, n + 2 + ix, n + 1 + ix] for ix in range(n)])
    zones = np.repeat(np.arange(1, len(sizes) + 1), sizes)
    return grid_from_mesh(xy, quads, zones), grid_from_mesh(xy, quads), zones


def _ecospace(scenario, grid, capacity=None, monthly=None):
    n_rows = scenario.params.NUM_GROUPS + 1
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n_rows, grid.n_patches)),
        habitat_capacity=(
            np.ones((n_rows, grid.n_patches)) if capacity is None else capacity
        ),
        dispersal_rate=np.zeros(n_rows),
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
        monthly_capacity=monthly,
    )


def _base(scenario, grid, rows=(1, 2), pattern=(0.4, 1.0, 0.7)):
    """A base map in which the groups ``rows`` differ between the zones."""
    capacity = np.ones((scenario.params.NUM_GROUPS + 1, grid.n_patches))
    capacity[list(rows)] = np.asarray(pattern)[: grid.n_patches]
    return capacity


def _run(scenario, grid, capacity=None, monthly=None, **kwargs):
    return rsim_run_spatial(
        scenario,
        ecospace=_ecospace(scenario, grid, capacity, monthly),
        years=YEARS,
        **kwargs,
    ).out_Biomass_spatial


# --- the table ------------------------------------------------------------


def test_bad_tables_are_refused():
    values = np.ones((12, 2, 3))
    with pytest.raises(ValueError, match="named twice"):
        MonthlyCapacity(groups=[1, 1], values=values)
    with pytest.raises(ValueError, match=r"\[n_months, 2, n_patches\]"):
        MonthlyCapacity(groups=[1, 2], values=np.ones((12, 3, 3)))
    with pytest.raises(ValueError, match="not negative"):
        MonthlyCapacity(groups=[1, 2], values=-values)
    with pytest.raises(ValueError, match="finite"):
        MonthlyCapacity(groups=[1, 2], values=values * np.nan)
    with pytest.raises(ValueError, match="beyond must be"):
        MonthlyCapacity(groups=[1, 2], values=values, beyond="loop")
    with pytest.raises(ValueError, match="twelve months"):
        MonthlyCapacity(groups=[1, 2], values=values[:5], beyond="mean_year")


def test_months_outside_the_table():
    values = np.arange(24.0).reshape(24, 1, 1) * np.ones((24, 1, 3))
    refuse = MonthlyCapacity(groups=[1], values=values, first_month=12)
    assert refuse.month_index(12) == 0 and refuse.month_index(35) == 23
    for outside in (0, 11, 36):
        with pytest.raises(ValueError, match="outside the monthly capacity"):
            refuse.month_index(outside)
    repeat = MonthlyCapacity(
        groups=[1], values=values, first_month=12, beyond="repeat_year"
    )
    assert [repeat.month_index(m) for m in (0, 5, 36, 41)] == [0, 5, 12, 17]
    hold = MonthlyCapacity(groups=[1], values=values, first_month=12, beyond="hold")
    assert hold.month_index(0) == 0 and hold.month_index(99) == 23
    mean = MonthlyCapacity(
        groups=[1], values=values, first_month=12, beyond="mean_year"
    )
    assert mean.n_months == 24 and mean.values.shape[0] == 36
    assert mean.month_index(0) == 24 and mean.month_index(40) == 24 + 4
    np.testing.assert_allclose(mean.values[24, 0], 6.0)  # mean of months 0 and 12
    # the month's capacity over the base map's peak, not below EwE's minimum
    np.testing.assert_allclose(hold.efficiency(13, np.array([2.0])), 0.5)
    assert hold.efficiency(12, np.array([2.0])).min() == 1e-6


# --- in a run -------------------------------------------------------------


def test_without_a_monthly_capacity_the_run_is_as_before(spatial_scenario):
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    base = _base(scenario, grid)
    np.testing.assert_array_equal(
        _run(scenario, grid, base, None), _run(scenario, grid, base)
    )


def test_a_monthly_capacity_equal_to_the_base_map_is_the_static_run(spatial_scenario):
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    base = _base(scenario, grid)
    monthly = MonthlyCapacity(
        groups=[1, 2], values=np.tile(base[[1, 2]], (MONTHS, 1, 1))
    )
    np.testing.assert_allclose(
        _run(scenario, grid, base, monthly),
        _run(scenario, grid, base),
        rtol=1e-12,
        atol=1e-14,
    )
    # the scale of the two cancels: both doubled is the same run
    doubled = MonthlyCapacity(
        groups=[1, 2], values=2.0 * np.tile(base[[1, 2]], (MONTHS, 1, 1))
    )
    twice = base.copy()
    twice[[1, 2]] *= 2.0
    np.testing.assert_allclose(
        _run(scenario, grid, twice, doubled),
        _run(scenario, grid, base),
        rtol=1e-12,
        atol=1e-14,
    )


def test_on_a_uniform_base_map_a_monthly_capacity_of_one_changes_nothing(
    spatial_scenario,
):
    """A uniform base sets no calibration; the monthly capacity brings one in."""
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    monthly = MonthlyCapacity(groups=[1, 2], values=np.ones((MONTHS, 2, 3)))
    np.testing.assert_allclose(
        _run(scenario, grid, None, monthly),
        _run(scenario, grid),
        rtol=1e-9,
        atol=1e-12,
    )


def test_the_mpa_capacity_bonus_is_a_monthly_capacity(spatial_scenario):
    """An oracle the engine already has: a zone whose capacity rises by half."""
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    n = scenario.params.NUM_GROUPS
    base = np.ones((n + 1, 3))
    base[:n] = [0.4, 1.0, 0.7]  # every group, so the bonus acts on every row
    # no fleet is excluded: the zone only raises the capacity
    reserve = MPAZone(
        1, "reserve", patches=[0], start_month=6, excluded_fleets=[], capacity_bonus=1.5
    )
    mpa = MPAConfig(zones=[reserve])
    with_mpa = rsim_run_spatial(
        scenario, ecospace=_ecospace(scenario, grid, base), years=YEARS, mpa=mpa
    ).out_Biomass_spatial
    values = np.tile(base[:n], (MONTHS, 1, 1))
    # the MPA reads the step's number (1 is the first), the capacity the
    # month (0 is the first): step 6 is month 5
    values[5:, :, 0] *= 1.5
    monthly = MonthlyCapacity(groups=list(range(n)), values=values)
    np.testing.assert_allclose(
        _run(scenario, grid, base, monthly), with_mpa, rtol=1e-12, atol=1e-14
    )
    assert not np.allclose(with_mpa, _run(scenario, grid, base), rtol=1e-6)


def test_a_capacity_uniform_in_space_keeps_the_patches_alike_and_changes_the_level(
    spatial_scenario,
):
    """Half the capacity everywhere in the second year: a level effect only."""
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    values = np.ones((MONTHS, 1, 3))
    values[12:] = 0.5
    out = _run(scenario, grid, None, MonthlyCapacity(groups=[1], values=values))
    plain = _run(scenario, grid)
    density = out / grid.patch_areas
    np.testing.assert_allclose(
        density, np.broadcast_to(density[:, :, :1], density.shape), rtol=1e-9
    )
    np.testing.assert_allclose(out[:13], plain[:13], rtol=1e-9, atol=1e-12)
    # the level matters here, where a static map's does not: the grazer and
    # its prey both answer to the halved capacity
    grazer, prey = 2, 1
    assert abs(out[-1, grazer].sum() / plain[-1, grazer].sum() - 1) > 0.02
    assert abs(out[-1, prey].sum() / plain[-1, prey].sum() - 1) > 0.02


def test_a_month_at_zero_capacity_stays_finite(spatial_scenario):
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    values = np.ones((MONTHS, 2, 3))
    values[6:9] = 0.0  # nothing suitable anywhere for three months
    values[12:, :, 0] = 0.0  # and one zone unsuitable from the second year
    out = _run(scenario, grid, _base(scenario, grid), MonthlyCapacity([1, 2], values))
    assert np.isfinite(out).all() and out.min() >= 0


def test_zones_and_the_cells_inside_them_run_alike(spatial_scenario):
    scenario, _ = spatial_scenario
    zones_grid, cells_grid, zone_of = _grid()
    position = zone_of - 1
    pattern = np.array([0.4, 1.0, 0.7])
    rng = np.random.default_rng(3)
    series = 0.3 + 0.7 * rng.random((MONTHS, 2, 3))

    def run(grid, of):
        n_rows = scenario.params.NUM_GROUPS + 1
        base = np.ones((n_rows, grid.n_patches))
        base[[1, 2]] = pattern[of]
        return _run(scenario, grid, base, MonthlyCapacity([1, 2], series[:, :, of]))

    by_zone = run(zones_grid, np.arange(3))
    by_cell = run(cells_grid, position)
    summed = np.zeros_like(by_zone)
    for cell in range(cells_grid.n_patches):
        summed[:, :, position[cell]] += by_cell[:, :, cell]
    np.testing.assert_allclose(summed, by_zone, rtol=1e-9, atol=1e-12)


def test_refusals(spatial_scenario):
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    ones = np.ones((MONTHS, 1, 3))
    with pytest.raises(ValueError, match="5 patches, the grid 3"):
        _run(scenario, grid, None, MonthlyCapacity([1], np.ones((MONTHS, 1, 5))))
    with pytest.raises(
        ValueError, match=r"months 0\.\.23, the monthly capacity covers 0\.\.11"
    ):
        _run(scenario, grid, None, MonthlyCapacity([1], ones[:12]))
    assert np.isfinite(
        _run(
            scenario, grid, None, MonthlyCapacity([1], ones[:12], beyond="repeat_year")
        )
    ).all()
    n = scenario.params.NUM_GROUPS
    with pytest.raises(ValueError, match=f"outside 0..{n - 1}"):
        _run(scenario, grid, None, MonthlyCapacity([n], ones))
    names = [str(x) for x in scenario.params.spname]
    monthly = MonthlyCapacity([1], ones)
    ecospace = _ecospace(scenario, grid)
    assert monthly.prepare(ecospace, MONTHS, n_groups=n) == [2]
    with pytest.raises(ValueError, match="multi-stanza group") as error:
        monthly.prepare(ecospace, MONTHS, n_groups=n, stanza_rows=[2], names=names)
    assert names[2] in str(error.value)
    with pytest.raises(ValueError, match="an IBM group"):
        monthly.prepare(ecospace, MONTHS, n_groups=n, ibm_rows=[2])


def test_an_mpa_bonus_combines_with_a_varying_base_map_and_not_a_uniform_one(
    spatial_scenario,
):
    """On a uniform map the bonus acts in another form: the pair is refused."""
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    reserve = MPAZone(
        1, "reserve", patches=[0], start_month=6, excluded_fleets=[], capacity_bonus=1.5
    )
    mpa = MPAConfig(zones=[reserve])

    def run(base, monthly, zones=mpa):
        return rsim_run_spatial(
            scenario,
            ecospace=_ecospace(scenario, grid, base, monthly),
            years=YEARS,
            mpa=zones,
        ).out_Biomass_spatial

    base = _base(scenario, grid)
    same = MonthlyCapacity([1], np.tile(base[[1]], (MONTHS, 1, 1)))
    np.testing.assert_allclose(run(base, same), run(base, None), rtol=1e-12, atol=1e-14)
    ones = MonthlyCapacity([1], np.ones((MONTHS, 1, 3)))
    with pytest.raises(
        ValueError, match="cannot be combined with an MPA capacity bonus"
    ):
        run(None, ones)
    # a zone without a bonus is no obstacle
    closed = MPAConfig(zones=[MPAZone(2, "closed", patches=[0], excluded_fleets=[])])
    np.testing.assert_allclose(
        run(None, ones, closed), run(None, None, closed), rtol=1e-9, atol=1e-12
    )


# --- from layers ----------------------------------------------------------


def _salt_layer(months=24, first=0):
    """Salinity by zone: fresh, brackish, salt; the middle zone salter in summer."""
    values = np.tile([0.0, 2.0, 6.0], (months, 1)).astype(float)
    summer = (np.arange(first, first + months) % 12 >= 5) & (
        np.arange(first, first + months) % 12 <= 7
    )
    values[summer, 1] = 5.0
    return EnvironmentalLayer(
        name="salt",
        units="psu",
        values=values,
        times=np.arange(first, first + months) / 12.0,
        hold=True,
    )


def test_capacity_by_month_from_layers_and_responses():
    layers = {"salt": _salt_layer()}
    marine = create_linear_response(0.0, 6.0)
    names = ["Phyto", "Zoo", "Fish", "Det", "Fleet"]
    monthly = MonthlyCapacity.from_layers(
        layers, {"Fish": {"salt": marine}}, group_names=names
    )
    assert monthly.groups == [2] and monthly.first_month == 0
    assert monthly.values.shape == (24, 1, 3)
    np.testing.assert_allclose(monthly.values[0, 0], [0.0, 2 / 6, 1.0])
    np.testing.assert_allclose(monthly.values[6, 0], [0.0, 5 / 6, 1.0])  # July
    # the base map on the same scale: the mean over the months
    base = capacity_from_layers(layers, {"Fish": {"salt": marine}}, group_names=names)
    np.testing.assert_allclose(base[2], monthly.values[:, 0].mean(axis=0))
    late = MonthlyCapacity.from_layers(
        {"salt": _salt_layer(12, first=36)}, {2: {"salt": marine}}
    )
    assert late.first_month == 36 and late.n_months == 12


def test_layers_that_cannot_give_a_monthly_capacity():
    marine = create_linear_response(0.0, 6.0)
    still = EnvironmentalLayer(
        name="depth", units="m", values=np.array([1.0, 2.0, 3.0])
    )
    with pytest.raises(ValueError, match="changes in time"):
        MonthlyCapacity.from_layers({"depth": still}, {2: {"depth": marine}})
    with pytest.raises(KeyError, match="oxygen"):
        MonthlyCapacity.from_layers({"salt": _salt_layer()}, {2: {"oxygen": marine}})
    with pytest.raises(KeyError, match="Cod"):
        MonthlyCapacity.from_layers(
            {"salt": _salt_layer()}, {"Cod": {"salt": marine}}, group_names=["Fish"]
        )
    gapped = _salt_layer()
    gapped.values, gapped.times = gapped.values[::2], gapped.times[::2]
    gapped.n_timesteps = 12
    with pytest.raises(ValueError, match="gaps"):
        MonthlyCapacity.from_layers({"salt": gapped}, {2: {"salt": marine}})


def test_a_summer_intrusion_moves_a_marine_group_into_the_middle_zone(
    spatial_scenario,
):
    """Where it must act: the middle zone is suitable in summer only."""
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    layers = {"salt": _salt_layer()}
    marine = create_linear_response(0.0, 6.0)
    responses = {2: {"salt": marine}}  # the fish
    base = capacity_from_layers(
        layers, responses, n_groups=scenario.params.NUM_GROUPS + 1
    )
    monthly = MonthlyCapacity.from_layers(layers, responses)
    static = _run(scenario, grid, base)
    moving = _run(scenario, grid, base, monthly)
    assert np.isfinite(moving).all()
    fish, middle = 3, 1
    # by the end of the first summer the fish in the middle zone have done
    # better than with the year's mean map, and by late winter worse
    assert moving[8, fish, middle] > static[8, fish, middle]
    assert moving[17, fish, middle] < moving[8, fish, middle] * (
        static[17, fish, middle] / static[8, fish, middle]
    )


# --- movement that follows the month's map ----------------------------------


def _switching(scenario, grid, movement, rows=(2,), change=12):
    """Base map for ``change`` months, then its mirror image.

    ``rows`` are rows of the EcospaceParams arrays: 2 is the fish, whose
    row in the results is 3.
    """
    base = _base(scenario, grid, rows=rows)
    values = np.repeat(base[list(rows)][np.newaxis], MONTHS, axis=0)
    values[change:] = values[change:, :, ::-1]
    monthly = MonthlyCapacity(groups=list(rows), values=values, movement=movement)
    return base, monthly


def _moving(scenario, grid, base, monthly, rate=5.0, rows=(2,), **kwargs):
    ecospace = _ecospace(scenario, grid, base, monthly)
    for row in rows:
        ecospace.dispersal_rate[row] = rate
    for name, value in kwargs.pop("ecospace_fields", {}).items():
        setattr(ecospace, name, value)
    return rsim_run_spatial(
        scenario, ecospace=ecospace, years=YEARS, **kwargs
    ).out_Biomass_spatial


def test_movement_keeps_the_base_map_unless_asked(spatial_scenario):
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    assert MonthlyCapacity(groups=[2], values=np.ones((12, 1, 3))).movement is False
    base, off = _switching(scenario, grid, movement=False)
    default = MonthlyCapacity(groups=off.groups, values=off.values)
    np.testing.assert_array_equal(
        _moving(scenario, grid, base, off), _moving(scenario, grid, base, default)
    )


@pytest.mark.parametrize("implicit", [True, False])
def test_movement_follows_the_months_map_when_asked(spatial_scenario, implicit):
    """Also the stale-factorisation check: month 12 must not reuse month 0."""
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    fish, rate = 3, 5.0 if implicit else 0.5
    base, off = _switching(scenario, grid, movement=False)
    _, on = _switching(scenario, grid, movement=True)
    kept = _moving(scenario, grid, base, off, rate, implicit_movement=implicit)
    follows = _moving(scenario, grid, base, on, rate, implicit_movement=implicit)
    assert np.isfinite(follows).all()
    # while the month's map is the base map the two runs are one run
    np.testing.assert_array_equal(follows[:13], kept[:13])
    # afterwards the first zone is the poor one: movement that follows the
    # month's map takes fish out of it, movement on the base map brings
    # them back in
    density = follows[:, fish] / grid.patch_areas
    kept_density = kept[:, fish] / grid.patch_areas
    assert density[-1, 2] / density[-1, 0] < (0.8 if implicit else 0.9) * (
        kept_density[-1, 2] / kept_density[-1, 0]
    )
    if implicit:
        # the figures the how-to quotes (the base map asks for 1.75, its
        # mirror image for 0.57)
        assert kept_density[-1, 2] / kept_density[-1, 0] == pytest.approx(
            1.42, abs=0.02
        )
        assert density[-1, 2] / density[-1, 0] == pytest.approx(0.66, abs=0.02)


def test_movement_alone_settles_on_the_months_map(spatial_scenario):
    """Diffusion's rest state is the month's capacity times the area."""
    from pypath.spatial.dispersal import implicit_diffusion_step, movement_capacity

    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    base, on = _switching(scenario, grid, movement=True, change=1)
    ecospace = _ecospace(scenario, grid, base, on)
    assert movement_capacity(ecospace, 2) is not None
    rows = on.movement_rows(month=0, base_peak=np.array([1.0]))
    np.testing.assert_allclose(rows[2], base[2])
    later = on.movement_rows(month=5, base_peak=np.array([1.0]))
    np.testing.assert_allclose(later[2], base[2][::-1])
    assert movement_capacity(ecospace, 2, later) is later[2]
    assert movement_capacity(ecospace, 1, later) is movement_capacity(ecospace, 1)
    biomass = np.array([3.0, 1.0, 2.0])
    for _ in range(400):
        biomass = implicit_diffusion_step(
            biomass,
            5.0,
            grid,
            grid.adjacency_matrix,
            1 / 12,
            capacity=movement_capacity(ecospace, 2, later),
        )
    assert biomass.sum() == pytest.approx(6.0, rel=1e-10)
    rest = later[2] * grid.patch_areas
    np.testing.assert_allclose(biomass / biomass.sum(), rest / rest.sum(), rtol=1e-6)
    # a month whose map is the same everywhere is plain diffusion
    flat = MonthlyCapacity(groups=[2], values=np.full((12, 1, 3), 0.5), movement=True)
    assert (
        movement_capacity(ecospace, 2, flat.movement_rows(0, np.array([1.0]))) is None
    )
    assert flat.movement_rows(0, np.array([1.0]))[2].tolist() == [0.5, 0.5, 0.5]
    assert MonthlyCapacity(groups=[2], values=flat.values).movement_rows(0, [1.0]) == {}
    # a patch at no capacity stays at EwE's minimum, so nothing divides by 0
    empty = MonthlyCapacity(
        groups=[2], values=np.tile([0.0, 1.0, 0.5], (12, 1, 1)), movement=True
    )
    assert empty.movement_rows(0, np.array([1.0]))[2].min() == 1e-6
    out = implicit_diffusion_step(
        np.array([3.0, 1.0, 2.0]),
        5.0,
        grid,
        grid.adjacency_matrix,
        1 / 12,
        capacity=empty.movement_rows(0, np.array([1.0]))[2],
    )
    assert np.isfinite(out).all() and out.sum() == pytest.approx(6.0, rel=1e-10)


def test_a_group_without_a_monthly_capacity_moves_as_before(spatial_scenario):
    """Zoo moves on its base map whatever the fish's switch says."""
    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    on = MonthlyCapacity(groups=[2], values=np.ones((MONTHS, 1, 3)), movement=True)
    off = MonthlyCapacity(groups=[2], values=np.ones((MONTHS, 1, 3)))
    base = _base(scenario, grid, rows=(1,))  # only Zoo's map varies
    np.testing.assert_array_equal(
        _moving(scenario, grid, base, on, rows=(1, 2)),
        _moving(scenario, grid, base, off, rows=(1, 2)),
    )


def test_the_rest_state_of_water_transport_follows_with_movement(spatial_scenario):
    from pypath.spatial.transport import WaterTransport

    scenario, _ = spatial_scenario
    grid, _, _ = _grid()
    flow = np.zeros((MONTHS, 3, 3))
    for a, b in ((0, 1), (1, 2)):
        flow[:, a, b] = flow[:, b, a] = 40.0

    def transport():
        return WaterTransport(
            zones=[1, 2, 3],
            flow=flow,
            volume=np.full((MONTHS, 3), 1e8),
            groups=[2],
        )

    base, off = _switching(scenario, grid, movement=False)
    _, on = _switching(scenario, grid, movement=True)
    assert transport().rest == "capacity"
    runs = {
        name: _moving(
            scenario,
            grid,
            base,
            monthly,
            rate=0.0,
            ecospace_fields={"water_transport": transport()},
        )
        for name, monthly in (("kept", off), ("follows", on))
    }
    fish = 3
    np.testing.assert_array_equal(runs["follows"][:13], runs["kept"][:13])
    ratio = {
        name: (run[-1, fish, 2] / grid.patch_areas[2])
        / (run[-1, fish, 0] / grid.patch_areas[0])
        for name, run in runs.items()
    }
    assert ratio["follows"] < 0.8 * ratio["kept"]
    # with the water's volume as rest state the capacity plays no part
    by_volume = {}
    for name, monthly in (("kept", off), ("follows", on)):
        carrier = transport()
        carrier.rest = "volume"
        by_volume[name] = _moving(
            scenario,
            grid,
            base,
            monthly,
            rate=0.0,
            ecospace_fields={"water_transport": carrier},
        )
    np.testing.assert_array_equal(by_volume["follows"], by_volume["kept"])
