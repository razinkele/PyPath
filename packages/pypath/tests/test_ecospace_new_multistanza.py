"""EwE's NewMultiStanza distribution of stanza groups over Ecospace cells.

EwE 6.7.144's Ecospace (ModelType 2, its default for a new scenario) does not
integrate a stanza group cell by cell. Each life stage keeps a population
index W per cell, which gains recruits and loses at the cell's loss rate Z:

    dW/dt = Rflow - Z W + movement            (cSpaceSolver.vb 665-720)

Rflow is the cell's adult density over the adults' start biomass, times
(1 - exp(-Z Tstanza)), for the first stage, and the previous stage's density
for the others. Each month the stage's map-wide total from the age structure
is spread over the cells by W (cEcoSpace.vb 7406-7445), and the age
structure's update averages the cells' rates weighted by W (SpaceSplitUpdate,
7451-7470).
"""

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial
from pypath.spatial.stanza_space import StanzaSpaceIndex

NAN = np.nan
# one species, two stages: months 0-11 (Tstanza 1 year) and 12-59 (4 years)
STANZAS = SimpleNamespace(
    n_split=1,
    n_stanzas=np.array([0, 2]),
    ecopath_code=np.array([[NAN, NAN, NAN], [NAN, 1, 2]]),
    age1=np.array([[NAN, NAN, NAN], [NAN, 0, 12]]),
    age2=np.array([[NAN, NAN, NAN], [NAN, 11, 59]]),
)
AREA = np.array([0.25, 0.75])  # patch area shares
# rows: Outside, juveniles, adults, a group without stanzas (amounts per patch)
B = np.array([[0.0, 0.0], [0.1, 0.6], [0.5, 1.5], [0.2, 0.2]])


def _index(biomass=B, z=(2.0, 0.5)):
    idx = StanzaSpaceIndex.from_stanzas(STANZAS, AREA)
    loss = np.zeros_like(biomass)
    loss[1], loss[2] = z[0] * biomass[1], z[1] * biomass[2]
    idx.start(biomass, loss)
    return idx


def test_the_start_index_is_ewes_nstanza_base():
    # RecSplit / Z x the cell's capacity relative to the map (cEcoSpace.vb
    # 3187-3210, 2569): St = 1, then exp(-Z Tstanza) for the adults
    idx = _index()
    rel = (B / AREA) / B.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(idx.index[1], (1 - np.exp(-2.0)) / 2.0 * rel[1])
    np.testing.assert_allclose(idx.index[2], np.exp(-2.0) / 0.5 * rel[2])


def test_the_first_stage_recruits_from_the_adults_and_the_next_from_it():
    idx = _index()
    w1, w2 = idx.index[1].copy(), idx.index[2].copy()
    z1, z2 = np.array([1.0, 3.0]), np.array([0.4, 0.6])
    loss = np.zeros_like(B)
    loss[1], loss[2] = z1 * B[1], z2 * B[2]
    dt = 1 / 12
    idx.advance(B, loss, dt)
    dens = B / AREA
    rflow1 = dens[2] / B[2].sum() * (1 - np.exp(-z1 * 1.0))
    np.testing.assert_allclose(idx.index[1], (w1 + dt * rflow1) / (1 + dt * z1))
    np.testing.assert_allclose(idx.index[2], (w2 + dt * dens[1]) / (1 + dt * z2))


def test_an_empty_cell_loses_at_ewes_rate_of_100():
    biomass = B.copy()
    idx = _index(biomass)
    w1 = idx.index[1].copy()
    biomass[1, 0] = 0.0
    loss = np.zeros_like(biomass)
    loss[1], loss[2] = 2.0 * biomass[1], 0.5 * biomass[2]
    dt = 1 / 12
    idx.advance(biomass, loss, dt)
    rflow = biomass[2, 0] / AREA[0] / B[2].sum() * (1 - np.exp(-100.0))
    assert idx.index[1][0] == pytest.approx((w1[0] + dt * rflow) / (1 + dt * 100.0))


def test_the_index_moves_as_amounts():
    idx = _index()
    w1 = idx.index[1].copy()
    loss = np.zeros_like(B)
    loss[1], loss[2] = 2.0 * B[1], 0.5 * B[2]
    dt = 1 / 12
    rows = []

    def swap(row, amounts):
        rows.append(row)
        return amounts[::-1]

    idx.advance(B, loss, dt, move=swap)
    local = (w1 + dt * (B[2] / AREA) / B[2].sum() * (1 - np.exp(-2.0))) / (1 + dt * 2)
    np.testing.assert_allclose(idx.index[1], (local * AREA)[::-1] / AREA)
    assert rows == [1, 2]


def test_a_stage_is_spread_by_its_index():
    idx = _index()
    idx.index = {1: np.array([1.0, 1.0]), 2: np.array([3.0, 1.0])}
    total = np.array([0.0, 0.7, 2.0, 0.4])
    out = idx.distribute(total, B)
    np.testing.assert_allclose(out[1], 0.7 * AREA)  # the index is a density
    np.testing.assert_allclose(out[2], [1.0, 1.0])
    np.testing.assert_allclose(out[3], B[3])  # not a stanza group
    np.testing.assert_allclose(out[1:3].sum(axis=1), total[1:3])


def test_the_age_structure_averages_the_cells_by_the_index():
    idx = _index()
    w = {1: np.array([2.0, 1.0]), 2: np.array([0.5, 1.5])}
    idx.index = {k: v.copy() for k, v in w.items()}
    food = np.array([[0.0, 0.0], [0.3, 0.9], [0.4, 0.2], [0.1, 0.3]])
    loss = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.1], [0.2, 0.1]])
    b, fg, lo = idx.weighted_sums(B, food, loss)
    for s in (1, 2):
        # TotLoss / TotBiom and TotEatenBy / TotPred, with each cell's pred
        # its share of the map's (cEcoSpace.vb 7464-7470). EwE's cells all
        # count alike (ThabArea counts them); a patch counts by its area, so
        # these sum the patch amounts weighted by W
        assert lo[s] / b[s] == pytest.approx(
            np.sum(loss[s] * w[s]) / np.sum(B[s] * w[s])
        )
        pred = 0.37
        pred_patch = pred * B[s] / B[s].sum()
        assert fg[s] / pred == pytest.approx(
            np.sum(food[s] * w[s]) / np.sum(pred_patch * w[s])
        )
    np.testing.assert_allclose([b[3], fg[3], lo[3]], [0.4, 0.4, 0.3])


def test_a_uniform_index_gives_the_plain_sums():
    idx = _index()
    idx.index = {1: np.ones(2), 2: np.full(2, 4.0)}
    # patches of unequal area, with different rates per unit biomass
    biomass = np.array([[0.0, 0.0], [0.25, 0.75], [0.5, 1.0], [0.2, 0.2]])
    food = np.array([[0.0, 0.0], [0.1, 0.9], [0.4, 0.1], [0.1, 0.3]])
    loss = np.array([[0.0, 0.0], [0.25, 0.0], [0.05, 0.6], [0.2, 0.1]])
    for got, want in zip(
        idx.weighted_sums(biomass, food, loss),
        (biomass.sum(axis=1), food.sum(axis=1), loss.sum(axis=1)),
    ):
        np.testing.assert_allclose(got, want)


def test_a_species_left_out_keeps_no_index():
    idx = StanzaSpaceIndex.from_stanzas(STANZAS, AREA, leave_out={2})
    assert idx.species == []
    idx.start(B, 0.5 * B)
    assert not idx.index


# --- in rsim_run_spatial ---

REF = Path(__file__).parent / "data" / "rpath_reference" / "ecopath"
needs_reco = pytest.mark.skipif(
    not (REF / "stanza_indiv.csv").exists(), reason="Rpath reference data missing"
)


@pytest.fixture(scope="module")
def reco():
    """Rpath's REco model: 4 multi-stanza species."""
    model_df = pd.read_csv(REF / "model_params.csv")
    params = create_rpath_params(model_df["Group"].tolist(), model_df["Type"].tolist())
    params.model = model_df
    params.diet = pd.read_csv(REF / "diet_matrix.csv")
    params.stanzas.stgroups = pd.read_csv(REF / "stanza_groups.csv")
    params.stanzas.stindiv = pd.read_csv(REF / "stanza_indiv.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


def _scenario(reco, rule="ewe"):
    model, params = reco
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenario = rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)
    scenario.fishing.ForcedEffort[:] = 2.0
    # EwE's rule takes the stanza rates from the RK4 step's stages; Rpath's
    # from a derivative at the end of the month
    assert scenario.params.STANZA_STAGE_RATES == (rule == "ewe")
    return scenario


def _stanza_rows(reco):
    model, params = reco
    names = list(model.Group)
    return [names.index(g) + 1 for g in params.stanzas.stindiv["Group"]]


RULES = pytest.mark.parametrize("rule", ["ewe", "rpath"])


def _ecospace(ng, capacity, areas, new_multi_stanza=True):
    n_patches = capacity.shape[1]
    grid = create_1d_grid(n_patches=n_patches, spacing=1.0)
    grid.patch_areas = np.asarray(areas, dtype=float)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, n_patches)),
        habitat_capacity=capacity,
        dispersal_rate=np.full(ng, 2.0),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
        new_multi_stanza=new_multi_stanza,
    )


def test_ewes_default_is_the_new_multistanza_model():
    eco = _ecospace(3, np.ones((3, 2)), [1.0, 1.0])
    assert EcospaceParams(
        grid=eco.grid,
        habitat_preference=eco.habitat_preference,
        habitat_capacity=eco.habitat_capacity,
        dispersal_rate=eco.dispersal_rate,
        advection_enabled=eco.advection_enabled,
        gravity_strength=eco.gravity_strength,
    ).new_multi_stanza


@needs_reco
@RULES
def test_a_uniform_map_still_equals_ecosim(reco, rule):
    scenario = _scenario(reco, rule)
    ng = scenario.params.NUM_GROUPS
    expected = rsim_run(_scenario(reco, rule), years=range(1, 3))
    got = rsim_run_spatial(
        scenario,
        ecospace=_ecospace(ng, np.ones((ng, 3)), [1.0, 2.0, 4.0]),
        years=range(1, 3),
    )
    np.testing.assert_allclose(
        got.out_Biomass, expected.out_Biomass, rtol=1e-6, atol=1e-10
    )
    np.testing.assert_allclose(
        got.stanza_biomass, expected.stanza_biomass, rtol=1e-6, atol=1e-10
    )


@needs_reco
@RULES
def test_the_switch_changes_a_run_with_a_habitat_pattern(reco, rule):
    ng = _scenario(reco, rule).params.NUM_GROUPS
    cap = np.ones((ng, 3))
    cap[:, 0], cap[:, 2] = 0.2, 0.6
    runs = [
        rsim_run_spatial(
            _scenario(reco, rule),
            ecospace=_ecospace(ng, cap, [1.0, 2.0, 4.0], new_multi_stanza=on),
            years=range(1, 3),
        )
        for on in (True, False)
    ]
    rows = _stanza_rows(reco)
    for out in runs:
        assert np.all(np.isfinite(out.out_Biomass_spatial))
        # each stage's patches hold the age structure's total
        np.testing.assert_allclose(
            out.out_Biomass_spatial[1:, rows].sum(axis=2),
            out.stanza_biomass[1:, rows],
            rtol=1e-10,
        )
    assert not np.allclose(runs[0].out_Biomass_spatial, runs[1].out_Biomass_spatial)


@needs_reco
def test_a_species_moved_by_advection_keeps_its_transport(reco):
    # The index carries diffusion only: a species with habitat-directed
    # movement (or an external flux) keeps the patch-share rule, so its
    # transport is not spread back to the index every month
    model, params = reco
    scenario = _scenario(reco)
    ng = scenario.params.NUM_GROUPS
    names = list(model.Group)
    rows = [names.index(g) + 1 for g in ("JuvRoundfish1", "AduRoundfish1")]
    eco = _ecospace(ng, np.ones((ng, 3)), [1.0, 1.0, 1.0])
    for r in rows:
        eco.habitat_preference[r - 1] = [0.1, 1.0, 0.1]
        eco.advection_enabled[r - 1] = True
        eco.gravity_strength[r - 1] = 5.0
    out = rsim_run_spatial(scenario, ecospace=eco, years=range(1, 3))
    for r in rows:
        b = out.out_Biomass_spatial[-1, r]
        assert b[1] / b[[0, 2]].mean() > 1.5, names[r - 1]


def test_the_reader_takes_ewes_model_type():
    from pypath.io.ewemdb import _uses_new_multi_stanza

    # cDBDataSource.vb 7569-7579: 0 the old stanza model, 1 the IBM, else new
    assert [_uses_new_multi_stanza(m) for m in (0, 1, 2, None)] == [
        False,
        False,
        True,
        True,
    ]


# --- against EwE 6.7.144 (Black Sea, oracle run 43288513485) ---

ORACLE = Path(__file__).parent / "data" / "ewe_oracle" / "blacksea"
DB = Path(__file__).resolve().parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"
ANCHOVY = ("Anchovy 0-0+", "Anchovy 1-1+")


def _oracle(name):
    df = pd.read_csv(ORACLE / name)
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


@pytest.fixture(scope="module")
def black_sea():
    from pypath.io.ewemdb import (
        check_ewemdb_support,
        ecosim_scenario_from_ewemdb,
        read_ecospace,
    )

    if not DB.exists() or not any(check_ewemdb_support().values()):
        pytest.skip("Black Sea database or driver missing")
    scenario = ecosim_scenario_from_ewemdb(str(DB), scenario=1)
    p = scenario.params
    res = read_ecospace(str(DB), n_groups=p.NUM_LIVING + p.NUM_DEAD, scenario_id=1)
    e = res.ecospace
    n_rows, n_p = p.NUM_GROUPS, e.grid.n_patches
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
        new_multi_stanza=e.new_multi_stanza,
    )
    out = rsim_run_spatial(scenario, ecospace=eco, years=range(1, 11))
    names = [str(n).strip() for n in p.spname]
    return res, eco, out, names


@pytest.mark.integration
def test_the_black_sea_scenario_uses_the_new_model(black_sea):
    res = black_sea[0]
    assert int(res.scenario_meta["ModelType"]) == 2
    assert res.ecospace.new_multi_stanza


@pytest.mark.integration
def test_black_sea_anchovy_area_means_follow_ewes(black_sea):
    _, _, out, names = black_sea
    steps = _oracle("ecospace_steps.csv")
    dev = {}
    for g in ANCHOVY:
        ewe = steps[g].to_numpy(dtype=float)
        dev[g] = np.abs(out.out_Biomass[1 : len(ewe) + 1, names.index(g)] / ewe - 1)
    # spread by each patch's own share: juveniles 62 % max / 23 % median
    assert dev["Anchovy 0-0+"].max() < 0.45
    assert np.median(dev["Anchovy 0-0+"]) < 0.12
    assert dev["Anchovy 1-1+"].max() < 0.13


@pytest.mark.integration
def test_black_sea_anchovy_maps_follow_ewes(black_sea):
    _, eco, out, names = black_sea
    grid = eco.grid
    density = (
        out.out_Biomass_spatial
        / (grid.patch_areas / grid.patch_areas.sum())[None, None, :]
    )
    cells = pd.DataFrame(
        {
            "Row": grid.cell_metadata["row"] + 1,
            "Col": grid.cell_metadata["col"] + 1,
            "p": range(grid.n_patches),
        }
    )
    maps = _oracle("ecospace_maps.csv")
    mm = cells.merge(maps[maps.TimeStep == 12], on=["Row", "Col"])
    for g in ANCHOVY:
        py = density[12, names.index(g), mm.p.to_numpy()]
        ewe = mm[g].to_numpy()
        tvd = 0.5 * np.abs(py / py.sum() - ewe / ewe.sum()).sum()
        assert tvd < 0.02, g  # each patch's own share: 0.138 and 0.044
