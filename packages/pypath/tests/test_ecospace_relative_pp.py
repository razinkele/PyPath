"""EwE's relative primary production map in a spatial run
(EcospaceParams.relative_pp).

In EwE's Ecospace a cell's total nutrient is the map-wide total times the
cell's relative primary production, divided by the map's mean
(cSpaceSolver.vb 570, 860; cEcoSpace.vb 3540), and the producers'
production follows the free nutrient. No EwE run with such a map has been
compared with PyPath's: the tests check the rule, not parity.
"""

import copy
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from pypath.examples import example_ecosim_scenario
from pypath.io.ewemdb import _build_fallback_grid
from pypath.spatial import EcospaceParams, rsim_run_spatial
from pypath.spatial import integration as integ
from pypath.spatial.mesh import grid_from_mesh

pytestmark = pytest.mark.filterwarnings("ignore")

YEARS = range(1, 6)
PATTERN = np.array([1.0, 0.6, 0.2])
PELAGIC = Path(__file__).resolve().parents[3] / "Data" / "curonian_pelagic.sqlite"


@pytest.fixture(scope="module")
def lagoon():
    """The Curonian Lagoon scenario (EwE rule, PBmaxs 2) and its producer's row."""
    scenario = example_ecosim_scenario("Curonian_Lagoon", years=YEARS)
    names = [str(n).strip() for n in scenario.params.spname]
    return scenario, names.index("Fitoplankton")


def _ecospace(scenario, grid, producer=None, capacity=None, relative_pp=None, rate=0.0):
    n = scenario.params.NUM_GROUPS
    cap = np.ones((n, grid.n_patches))
    if capacity is not None:
        cap[producer - 1] = capacity
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n, grid.n_patches)),
        habitat_capacity=cap,
        dispersal_rate=np.full(n, rate),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
        relative_pp=relative_pp,
    )


def _run(scenario, ecospace, **kwargs):
    return rsim_run_spatial(
        scenario, ecospace=ecospace, years=YEARS, **kwargs
    ).out_Biomass_spatial


def _three_cells():
    return _build_fallback_grid(1, 3, 10.0)


# --- the map ------------------------------------------------------------------


def test_a_map_must_fit_the_grid_and_be_a_map(lagoon):
    scenario, _ = lagoon
    grid = _three_cells()
    for bad, message in (
        (np.ones(4), r"relative_pp shape \(4,\) != \(3,\)"),
        (np.array([1.0, -0.1, 1.0]), "finite and not negative"),
        (np.array([1.0, np.nan, 1.0]), "finite and not negative"),
        (np.zeros(3), "zero in every patch"),
    ):
        with pytest.raises(ValueError, match=message):
            _ecospace(scenario, grid, relative_pp=bad)
    assert _ecospace(scenario, grid).relative_pp is None
    kept = _ecospace(scenario, grid, relative_pp=[2, 1, 0]).relative_pp
    assert kept.dtype == float and kept.tolist() == [2.0, 1.0, 0.0]


def test_no_map_and_a_map_that_is_the_same_everywhere_are_one_run(lagoon):
    scenario, producer = lagoon
    grid = _three_cells()
    for capacity in (None, PATTERN):  # the plain and the calibrated form
        plain = _run(scenario, _ecospace(scenario, grid, producer, capacity))
        flat = _run(
            scenario,
            _ecospace(scenario, grid, producer, capacity, relative_pp=np.full(3, 7.0)),
        )
        np.testing.assert_array_equal(flat, plain)


def test_only_the_pattern_of_the_map_counts(lagoon):
    scenario, _ = lagoon
    grid = _three_cells()
    once = _run(scenario, _ecospace(scenario, grid, relative_pp=PATTERN))
    tenfold = _run(scenario, _ecospace(scenario, grid, relative_pp=10 * PATTERN))
    np.testing.assert_allclose(tenfold, once, rtol=1e-10)


# --- what it does -------------------------------------------------------------


def test_a_producer_follows_the_map_and_keeps_most_of_its_total(lagoon):
    """With nearly all nutrient free, a cell at s times the mean produces
    2s / (1 + s) as much; at PBmaxs 2 the producer settles at
    (3s - 1) / (1 + s) of its Ecopath level. The same pattern as a habitat
    capacity caps production instead: (2c - 1) of the level at capacity 1."""
    scenario, producer = lagoon
    assert scenario.params.NutBaseFreeProp == pytest.approx(0.9999)
    grid = _three_cells()

    def producer_total(**kwargs):
        cells = _run(scenario, _ecospace(scenario, grid, producer, **kwargs))[
            :, producer
        ]
        return cells[-1].sum() / cells[0].sum(), cells[-1] / cells[-1].sum()

    s = PATTERN / PATTERN.mean()
    rule = np.maximum((3 * s - 1) / (s + 1), 0.0).mean()
    assert rule == pytest.approx(0.833, abs=1e-3)
    as_map, share = producer_total(relative_pp=PATTERN)
    # a little above the rule: the grazers thin out where production is low
    assert rule - 0.01 < as_map < rule + 0.06
    assert share[0] > share[1] > share[2] > 0.05  # it lives in every cell
    as_capacity, share = producer_total(capacity=PATTERN)
    assert as_capacity == pytest.approx(0.667, abs=0.02)
    assert share[2] < 1e-6  # gone where the capacity is below 0.5
    assert as_map > as_capacity + 0.15


def test_the_map_is_weighed_by_the_patches_areas(lagoon):
    """Zones of one, two and three squares, and the squares themselves, run
    alike: the map's mean is over the area, not over the patches."""
    scenario, producer = lagoon
    sizes = (1, 2, 3)
    n = sum(sizes)
    xy = np.array(
        [(ix * 2000.0, iy * 2000.0) for iy in range(2) for ix in range(n + 1)]
    )
    quads = np.array([[ix, ix + 1, n + 2 + ix, n + 1 + ix] for ix in range(n)])
    zone_of = np.repeat(np.arange(len(sizes)), sizes)
    zones = grid_from_mesh(xy, quads, zone_of + 1)
    cells = grid_from_mesh(xy, quads)
    by_zone = _run(scenario, _ecospace(scenario, zones, relative_pp=PATTERN))
    by_cell = _run(scenario, _ecospace(scenario, cells, relative_pp=PATTERN[zone_of]))
    summed = np.zeros_like(by_zone)
    for cell in range(n):
        summed[:, :, zone_of[cell]] += by_cell[:, :, cell]
    np.testing.assert_allclose(summed, by_zone, rtol=1e-9, atol=1e-12)
    # and it is not the plain mean over patches: that map would differ
    plain_mean = _run(
        scenario, _ecospace(scenario, zones, relative_pp=PATTERN * (1.0, 1.0, 2.0))
    )
    assert not np.allclose(plain_mean[-1, producer], by_zone[-1, producer], rtol=1e-3)


# --- where it cannot act --------------------------------------------------------


def test_the_map_needs_the_ewe_rules_nutrient_limitation():
    scenario = example_ecosim_scenario(
        "Curonian_Lagoon", years=YEARS, fast_group_rule="rpath"
    )
    assert scenario.params.NutBaseFreeProp is None
    ecospace = _ecospace(scenario, _three_cells(), relative_pp=PATTERN)
    with pytest.raises(ValueError, match="only the EwE rule has"):
        rsim_run_spatial(scenario, ecospace=ecospace, years=YEARS)
    # a map that is no map is let through
    flat = _ecospace(scenario, _three_cells(), relative_pp=np.ones(3))
    out = rsim_run_spatial(scenario, ecospace=flat, years=YEARS)
    assert np.isfinite(out.out_Biomass_spatial).all()


# --- the ways a run evaluates the patches ---------------------------------------


def test_the_batched_kernel_is_left_out_and_the_workers_carry_the_map(
    lagoon, monkeypatch
):
    scenario, _ = lagoon
    grid = _three_cells()
    calls = []
    real = integ.deriv_batched

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(integ, "deriv_batched", counting)
    serial = _run(
        copy.deepcopy(scenario), _ecospace(scenario, grid, relative_pp=PATTERN)
    )
    assert calls == []  # one nutrient forcing for the whole map would not do
    pooled = _run(
        copy.deepcopy(scenario),
        _ecospace(scenario, grid, relative_pp=PATTERN),
        n_jobs=2,
    )
    np.testing.assert_array_equal(pooled, serial)


# --- from an EwE database ---------------------------------------------------------


def _with_a_map(tmp_path, value_of):
    """A copy of the Curonian pelagic database whose RelPPMap follows the
    column: ``value_of(row, col)`` for each water cell, as EwE stores it."""
    from pypath.io.ewemdb import read_ecospace

    db = tmp_path / "pelagic.sqlite"
    shutil.copy(PELAGIC, db)
    meta = read_ecospace(str(db), n_groups=19).ecospace.grid.cell_metadata
    by_row = {}
    for r, c in zip(meta["row"], meta["col"]):
        by_row.setdefault(int(r), []).append(f"{value_of(int(r), int(c)):.4f}")
    text = ";".join(",".join(by_row.get(r, [])) for r in range(94)) + ";"
    with sqlite3.connect(db) as con:
        con.execute(
            "UPDATE EcospaceScenario SET RelPPMap = ? WHERE ScenarioID = 1", (text,)
        )
    return db, meta


def test_a_stored_map_is_kept_only_when_asked_and_named_otherwise(tmp_path, caplog):
    from pypath.io.ewemdb import read_ecospace

    if not PELAGIC.exists():
        pytest.skip("the Curonian pelagic database is not here")
    # as stored, the map is 1 everywhere: nothing to keep either way
    assert (
        read_ecospace(str(PELAGIC), n_groups=19, relative_pp=True).ecospace.relative_pp
        is None
    )
    db, meta = _with_a_map(tmp_path, lambda r, c: 0.5 + c / 50.0)
    with caplog.at_level("WARNING"):
        default = read_ecospace(str(db), n_groups=19).ecospace
    assert default.relative_pp is None
    assert "relative primary production map that varies" in caplog.text
    kept = read_ecospace(str(db), n_groups=19, relative_pp=True).ecospace
    np.testing.assert_allclose(
        kept.relative_pp, 0.5 + meta["col"].to_numpy() / 50.0, atol=1e-4
    )
    # everything else is as without it
    np.testing.assert_array_equal(kept.habitat_capacity, default.habitat_capacity)
    np.testing.assert_array_equal(kept.dispersal_rate, default.dispersal_rate)
