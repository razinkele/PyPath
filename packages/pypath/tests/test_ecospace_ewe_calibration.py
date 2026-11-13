"""EwE's Ecospace base calibration and nutrient rule (open item B5).

EwE 6.7.144's Ecospace differs from its Ecosim in two ways that PyPath now
follows, measured against the oracle's Black Sea run:

- the base calibration ``AdjustSpaceParsNew`` (``pypath.spatial.calibration``)
  sets the search and vulnerability rates for groups spread by habitat
  capacity, and each cell's feeding efficiency is its habitat capacity
  (EatEff);
- Ecospace never applies Ecosim's nutrient forcing: a cell's free nutrient is
  NutTot x its relative productivity less its biomass (cSpaceSolver.vb 860).

Not followed: Ecospace's start runs a second derivative at the start biomass
(SetBiomassesEcospace), which only changes the first months of groups with
handling time; leaving it out keeps a uniform map equal to Ecosim.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import _build_link_matrix, rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_regular_grid, rsim_run_spatial
from pypath.spatial.calibration import ewe_space_calibration, habitat_ratio

ORACLE = Path(__file__).parent / "data" / "ewe_oracle" / "blacksea"
DB = Path(__file__).resolve().parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"


def _scenario(years=range(1, 3)):
    params = create_rpath_params(
        ["Fish", "Plankton", "Detritus", "Fleet"], [0, 1, 2, 3]
    )
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [10.0, 1.0, 5.0, 0.8]
    params.model.loc[1, ["Biomass", "PB", "EE"]] = [5.0, 50.0, 0.6]
    params.model.loc[2, "Biomass"] = 1.0
    params.diet.iloc[1, 1] = 1.0
    params.model.loc[0, "Fleet"] = 0.5
    return rsim_scenario(rpath(params), params, years=years)


def _ecospace(habitat, rates=(0.0, 0.0, 0.0, 0.0)):
    grid = create_regular_grid(bounds=(0, 0, len(habitat), 1), nx=len(habitat), ny=1)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((4, len(habitat))),
        habitat_capacity=np.tile(np.asarray(habitat, float), (4, 1)),
        dispersal_rate=np.asarray(rates, float),
        advection_enabled=np.zeros(4, dtype=bool),
        gravity_strength=np.zeros(4),
    )


# --- the calibration ---


def _arrays(scenario):
    p = scenario.params
    return (
        np.asarray(p.B_BaseRef, float),
        _build_link_matrix(p, p.QQ),
        _build_link_matrix(p, p.VV),
    )


def test_a_uniform_habitat_gives_back_ecosim():
    b0, q0, x0 = _arrays(_scenario())
    cal = ewe_space_calibration(b0, q0, x0, np.ones((len(b0), 6)))
    np.testing.assert_allclose(cal.bratio, 1.0)
    np.testing.assert_allclose(cal.bbase, b0)
    links = q0 > 0
    np.testing.assert_allclose(cal.vv[links], x0[links], rtol=1e-6)
    np.testing.assert_allclose(cal.qq[links], q0[links], rtol=1e-6)


def test_producers_gain_through_the_habitat_ratio():
    # PbSpace = pbbiomass / BRatio on the base B0 x BRatio: Q0 x BRatio
    b0, q0, x0 = _arrays(_scenario())
    cap = np.ones((len(b0), 4))
    cap[1:] = [1.0, 0.5, 0.2, 0.1]
    cal = ewe_space_calibration(b0, q0, x0, cap)
    sun = np.nonzero(q0[0] > 0)[0]
    assert len(sun)
    np.testing.assert_allclose(cal.qq[0, sun], q0[0, sun] * cal.bratio[sun])
    np.testing.assert_allclose(cal.bratio[1:], 1.0 / np.mean([1.0, 0.5, 0.2, 0.1]))


def test_habitat_ratio_is_one_over_the_mean_capacity():
    cap = np.array([[1.0, 1.0, 1.0], [1.0, 0.5, 0.0]])
    np.testing.assert_allclose(habitat_ratio(cap), [1.0, 2.0])
    np.testing.assert_allclose(
        habitat_ratio(cap, np.array([0.5, 0.25, 0.25])), [1.0, 1 / 0.625]
    )


# --- in a run ---


def test_ecospace_does_not_apply_ecosims_nutrient_forcing():
    # EwE's Ecosim scales the free nutrient by the nutrient forcing; its
    # Ecospace does not (cSpaceSolver.vb 860)
    runs = []
    for value in (1.0, 0.6):
        scenario = _scenario()
        scenario.params.NutBaseFreeProp = 0.9
        scenario.forcing.NutForcing = np.full(24, value)
        spatial = rsim_run_spatial(scenario, ecospace=_ecospace([1.0, 0.5, 0.2]))
        runs.append(spatial.out_Biomass_spatial)
    np.testing.assert_allclose(runs[1], runs[0], rtol=1e-12)


def test_ecosim_still_applies_the_nutrient_forcing():
    ends = []
    for value in (1.0, 0.6):
        scenario = _scenario()
        scenario.params.NutBaseFreeProp = 0.9
        scenario.forcing.NutForcing = np.full(24, value)
        ends.append(rsim_run(scenario, method="RK4", years=range(1, 3)).out_Biomass[-1])
    assert not np.allclose(ends[0], ends[1])


def test_a_uniform_run_keeps_the_ecosim_kernel():
    # without a habitat pattern the calibration is Ecosim's: patches at the
    # same density stay identical
    spatial = rsim_run_spatial(_scenario(), ecospace=_ecospace([1.0, 1.0, 1.0]))
    out = spatial.out_Biomass_spatial[-1]
    np.testing.assert_allclose(out[:, 0], out[:, 1], rtol=1e-12)


# --- against EwE 6.7.144 (Black Sea, oracle runs 43288513485 / 43500051427) ---


def _black_sea():
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
    return scenario, eco, hab


def _oracle(name):
    df = pd.read_csv(ORACLE / name)
    df.columns = [c.strip().strip('"') for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip().str.strip('"')
    return df


@pytest.mark.integration
def test_the_calibration_is_ewes():
    scenario, eco, hab = _black_sea()
    p = scenario.params
    names = [str(n).strip() for n in p.spname]
    b0, q0, x0 = _arrays(scenario)
    cap = np.ones((len(b0), hab.shape[1]))
    cap[1:] = hab
    cal = ewe_space_calibration(b0, q0, x0, cap)
    groups = _oracle("ecospace_calibration_groups.csv").set_index("Group")
    for g in groups.index:
        assert cal.bratio[names.index(g)] == pytest.approx(
            groups.loc[g, "BRatio"], rel=5e-3
        )
    links = _oracle("ecospace_calibration_links.csv")
    stanza = {"Anchovy 0-0+", "Anchovy 1-1+"}
    for _, r in links.iterrows():
        i, j = names.index(r.Prey), names.index(r.Pred)
        assert q0[i, j] == pytest.approx(r.Qlink, rel=1e-5)
        assert cal.vspace[i, j] == pytest.approx(r.Vspace, rel=5e-3)
        if r.Pred not in stanza:  # EwE's stanza index is read after the run
            cmco = groups.loc[r.Pred, "CmCo"]
            assert cal.search[i, j] == pytest.approx(
                r.Aspace * (cmco - 1) / cmco, rel=0.03
            )


@pytest.mark.integration
def test_black_sea_area_means_and_maps_follow_ewes():
    scenario, eco, _ = _black_sea()
    names = [str(n).strip() for n in scenario.params.spname]
    out = rsim_run_spatial(scenario, ecospace=eco, years=range(1, 11))
    steps = _oracle("ecospace_steps.csv")
    deviation = []
    for g in steps.columns[3:]:
        ewe = steps[g].to_numpy(dtype=float)
        py = out.out_Biomass[1 : len(ewe) + 1, names.index(g)]
        deviation.append(np.abs(py / ewe - 1))
    deviation = np.concatenate(deviation)
    # plain diffusion and the old capacity rule gave 21 % / 416 %, B2 alone
    # 13.6 % / 225 %; with the calibration and the nutrient rule
    # 3.9 % / 62 % (juvenile anchovy)
    assert np.median(deviation) < 0.06
    assert deviation.max() < 0.8

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
    maps = pd.read_csv(ORACLE / "ecospace_maps.csv")
    mm = cells.merge(maps[maps.TimeStep == 12], on=["Row", "Col"])
    tvd = []
    for g in maps.columns[3:]:
        py = density[12, names.index(g), mm.p.to_numpy()]
        ewe = mm[g].to_numpy()
        tvd.append(0.5 * np.abs(py / py.sum() - ewe / ewe.sum()).sum())
    assert np.median(tvd) < 0.02  # 0.034 with B2 alone, 0.008 now
