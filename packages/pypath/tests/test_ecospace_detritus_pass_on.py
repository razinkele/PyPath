"""EwE's Ecospace passes no detritus between detritus pools.

EwE 6.7.144's Ecosim sends DetPassedProp x B x DF from each detritus pool to
the others (cEcoSimModel.vb 3192-3197). Its Ecospace solver adds that flow in
a loop over group indices 1 to nDet (cSpaceSolver.vb 1519-1526), which are
living groups, whose DetPassedProp is 0: no pool receives anything from
another. A donor still loses its share inside DetritusOut, taken from
Ecosim. ``EcospaceParams.detritus_pass_on`` (default False, as EwE) switches
the Ecosim flow back on.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.ecosim_deriv import _rpath_detritus_derivs
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

# one living group (1) and two pools, A (2) and B (3); link 0 is unused
NUM_LIVING, NUM_DEAD = 1, 2
BB = np.array([0.0, 1.0, 10.0, 5.0])
QQ = np.zeros((4, 4))
QQ[2, 1] = 0.3  # the living group eats pool A
M0 = np.array([0.0, 0.5, 0.1, 0.0])
UNASSIM = np.array([0.0, 0.2, 0.0, 0.0])
DET_FROM = np.array([0, 1, 2, 2, 3])
DET_TO = np.array([0, 2, 3, 2, 3])
DET_FRAC = np.array([0.0, 1.0, 0.5, 0.5, 1.0])
DET_OUT = np.array([0.0, 0.0, 0.02, 0.03])  # fixed export rates


def _flows(pass_on, ewe):
    return _rpath_detritus_derivs(
        np.zeros(4),
        QQ,
        BB,
        M0,
        UNASSIM,
        DET_FROM,
        DET_TO,
        DET_FRAC,
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        NUM_LIVING,
        NUM_DEAD,
        det_passed_prop=np.array([0.0, 0.0, 0.2, 0.0]) if ewe else None,
        det_ba=np.zeros(4),
        det_out=DET_OUT.copy() if ewe else None,
        pass_on=pass_on,
    )


def test_under_ewes_rule_a_pool_receives_nothing_from_another():
    gain_on, loss_on = _flows(True, ewe=True)
    gain_off, loss_off = _flows(False, ewe=True)
    assert gain_on[3] == pytest.approx(0.2 * 10.0 * 0.5)  # DetPassedProp x B x DF
    assert gain_off[3] == 0.0
    # the living group's flow still arrives, and the donor loses as before
    assert gain_off[2] == pytest.approx(gain_on[2]) == pytest.approx(0.5 + 0.2 * 0.3)
    np.testing.assert_allclose(loss_off, loss_on)


def test_under_rpaths_rule_the_flow_to_another_pool_is_dropped():
    gain_on, loss_on = _flows(True, ewe=False)
    gain_off, loss_off = _flows(False, ewe=False)
    assert gain_on[3] - gain_off[3] == pytest.approx(0.5 * 0.1 * 10.0)
    # a pool's flow to itself and its losses are unchanged
    assert gain_off[2] == pytest.approx(gain_on[2])
    np.testing.assert_allclose(loss_off, loss_on)


# --- in rsim_run_spatial (Rpath's REco, with its Discards passed to Detritus) ---

REF = Path(__file__).parent / "data" / "rpath_reference" / "ecopath"
needs_reco = pytest.mark.skipif(
    not (REF / "stanza_indiv.csv").exists(), reason="Rpath reference data missing"
)


@pytest.fixture(scope="module")
def reco():
    model_df = pd.read_csv(REF / "model_params.csv")
    # REco sends Discards out of the system; send them to Detritus instead
    model_df.loc[model_df.Group == "Discards", "Detritus"] = 1.0
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
    return scenario


def _uniform(ng, **kw):
    grid = create_1d_grid(n_patches=3, spacing=1.0)
    grid.patch_areas = np.array([1.0, 2.0, 4.0])
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, 3)),
        habitat_capacity=np.ones((ng, 3)),
        dispersal_rate=np.full(ng, 2.0),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
        **kw,
    )


def test_ewes_default_passes_nothing_between_pools():
    eco = _uniform(3)
    assert eco.detritus_pass_on is False


@needs_reco
@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_with_the_pass_on_a_uniform_map_equals_ecosim(reco, rule):
    scenario = _scenario(reco, rule)
    ng = scenario.params.NUM_GROUPS
    expected = rsim_run(_scenario(reco, rule), years=range(1, 3))
    got = rsim_run_spatial(
        scenario, ecospace=_uniform(ng, detritus_pass_on=True), years=range(1, 3)
    )
    np.testing.assert_allclose(
        got.out_Biomass, expected.out_Biomass, rtol=1e-6, atol=1e-10
    )


@needs_reco
def test_without_it_the_receiving_pool_misses_the_pass_on(reco):
    model, _ = reco
    names = list(model.Group)
    det, disc = names.index("Detritus") + 1, names.index("Discards") + 1
    scenario = _scenario(reco)
    p = scenario.params
    ng = p.NUM_GROUPS
    expected = rsim_run(_scenario(reco), years=range(1, 3)).out_Biomass
    got = rsim_run_spatial(scenario, ecospace=_uniform(ng), years=range(1, 3))
    got = got.out_Biomass
    frac = sum(
        float(f)
        for a, b, f in zip(p.DetFrom, p.DetTo, p.DetFrac)
        if int(a) == disc and int(b) == det
    )
    pass_on = float(p.DetPassedProp[disc]) * float(p.B_BaseRef[disc]) * frac
    assert pass_on > 0
    # one month without the flow it would have received
    assert expected[1, det] - got[1, det] == pytest.approx(pass_on / 12, rel=0.05)
    assert got[1, disc] == pytest.approx(expected[1, disc], rel=1e-6)


# --- against EwE 6.7.144 (Black Sea, oracle run 43288513485) ---

ORACLE = Path(__file__).parent / "data" / "ewe_oracle" / "blacksea"
DB = Path(__file__).resolve().parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"


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
    e = read_ecospace(
        str(DB), n_groups=p.NUM_LIVING + p.NUM_DEAD, scenario_id=1
    ).ecospace
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
    )
    out = rsim_run_spatial(scenario, ecospace=eco, years=range(1, 11))
    steps = pd.read_csv(ORACLE / "ecospace_steps.csv")
    steps.columns = [c.strip().strip('"') for c in steps.columns]
    names = [str(n).strip() for n in p.spname]
    dev = {}
    for g in steps.columns[3:]:
        ewe = steps[g].to_numpy(dtype=float)
        dev[g] = np.abs(out.out_Biomass[1 : len(ewe) + 1, names.index(g)] / ewe - 1)
    return dev


@pytest.mark.integration
def test_black_sea_detritus_and_aurelia_follow_ewes(black_sea):
    # with the pass-on: POM 22 % max / 16 % median, Aurelia 27 % / 22 %
    for g in ("POM", "Detritus", "Aurelia"):
        assert black_sea[g].max() < 0.10, g


@pytest.mark.integration
def test_black_sea_area_means_follow_ewes(black_sea):
    allg = np.concatenate(list(black_sea.values()))
    assert np.median(allg) < 0.03  # 3.4 % with the pass-on
    assert allg.max() < 0.30  # 35 %
