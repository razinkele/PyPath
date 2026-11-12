"""EwE6 migration of detritus groups (deviation A1-s2, detritus part).

EwE's Ecopath leaves detritus migration out: a pool's input is its import
and the flows to it (InputToDet = DtImp + det, cEcoPathModel.vb 628), and its
export skips migration (UpdateExportCatch, vb 912-916). Ecosim moves it
(Derivt and SimDetritusMT, cEcoSimModel.vb 3117-3124, 3206; SetupSimVariables,
vb 4535):

- Immigration joins the pool's constant inflow, DetInFlow = DtImp + Immig;
- the pool loses Eatenof + (Emig + DetritusOut) x B, with Emig = EmigRate
  where only a rate is entered, else Emigration / B0;
- DetritusOut, set at the first Derivt, is (ToDetritus - BA + Immig + DtImp
  - EatenOf) / B - Emig, at least 1e-20.

So the base holds, a constant inflow damps the pool's response, and
emigration beyond the pool's surplus drains it.
"""

import copy
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest

import pypath.core.ecosim as EC
import pypath.core.ecosim_deriv as D

PHYTO, ZOO, FISH, DET = 1, 2, 3, 4  # rsim indices (0 = Outside)


def _params(det_b=10.0, **migration):
    """Phyto -> Zoo -> Fish, nothing eats Det. Det's inflow at the base:
    Phyto 600 + Zoo 20 + Fish 5 other mortality, 80 + 4 egestion = 709.
    ``migration``: column -> {group: value}."""
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(["Phyto", "Zoo", "Fish", "Det"], [1, 0, 0, 2])
    m = params.model
    m["Biomass"] = [100.0, 20.0, 5.0, det_b]
    m["PB"] = [10.0, 2.0, 1.0, np.nan]
    m["QB"] = [np.nan, 20.0, 4.0, np.nan]
    m["EE"] = np.nan
    m["BioAcc"] = 0.0
    m["Unassim"] = [0.0, 0.2, 0.2, 0.0]
    m["Det"] = [1.0, 1.0, 1.0, 0.0]
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    for col, values in migration.items():
        m[col] = 0.0
        for group, value in values.items():
            m.loc[m["Group"] == group, col] = value
    return params


def _balance(params):
    from pypath.core.ecopath import rpath

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params)


def _scenario(params, rule="ewe", years=2):
    from pypath.core.ecosim import rsim_scenario

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = rsim_scenario(
            _balance(params), params, years=range(1, years + 1), fast_group_rule=rule
        )
    # isolates migration; EwE's handling time (D5) has its own tests
    sc.params.EWE_HANDLING_TIME = False
    return sc


def _first_call(monkeypatch, sc):
    """The derivative's rates and DetritusOut at rsim_run's first call."""
    cap = {}
    real = D.deriv_vector

    def spy(state, params, forcing, fishing, *a, **k):
        if "P" not in cap:
            cap.update(
                P=copy.deepcopy(params),
                Fo=copy.deepcopy(forcing),
                Fi=copy.deepcopy(fishing),
                s=np.array(state, float),
            )
        return real(state, params, forcing, fishing, *a, **k)

    for module in (D, EC):
        monkeypatch.setattr(module, "deriv_vector", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EC.rsim_run(sc, method="RK4")
    monkeypatch.undo()
    rates = {}
    P = cap["P"]
    D.deriv_vector(cap["s"].copy(), P, cap["Fo"], cap["Fi"], rates=rates)
    det_out = P.get("_det_out")
    return rates, (None if det_out is None else float(det_out[DET])), cap["s"]


MIGRATION = {"Immigration": {"Det": 50.0}, "Emigration": {"Det": 20.0}}


def test_the_reader_keeps_detritus_migration(caplog):
    import pandas as pd

    from pypath.io.ewemdb import _resolve_migration

    groups = pd.DataFrame(
        {
            "GroupName": ["Fish", "Detritus"],
            "Immigration": [1.0, 50.0],
            "Emigration": [0.0, 20.0],
            "EmigRate": [0.1, 2.0],
        }
    )
    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        out = _resolve_migration(groups, [0, 2])
    assert caplog.text == ""
    np.testing.assert_array_equal(out["Immigration"], [1.0, 50.0])
    np.testing.assert_array_equal(out["Emigration"], [0.0, 20.0])
    np.testing.assert_array_equal(out["EmigRate"], [0.1, 2.0])


def test_ecopath_leaves_detritus_migration_out_but_reports_it():
    plain = _balance(_params())
    migrating = _balance(_params(EmigRate={"Det": 5.0}, **MIGRATION))
    np.testing.assert_allclose(migrating.EE, plain.EE, rtol=1e-12)
    assert migrating.Immigration[DET - 1] == 50.0
    assert migrating.Emigration[DET - 1] == 20.0
    # EmigRate counts only where Emigration is 0 (EwE's Emig rule)
    assert migrating.EmigRate[DET - 1] == 0.0
    assert _balance(_params(EmigRate={"Det": 5.0})).EmigRate[DET - 1] == 5.0


def test_immigration_joins_the_inflow_and_emigration_leaves_detritus_out(
    monkeypatch,
):
    """DetGain gains the 50 immigrating; DetritusOut = (709 + 50) / 10 - 2
    (Emig = 20 / 10), 3 above the 70.9 without migration, and the pool's
    loss, (Emig + DetritusOut) x B, still matches its inflow."""
    r0, out0, _ = _first_call(monkeypatch, _scenario(_params()))
    r1, out1, s1 = _first_call(monkeypatch, _scenario(_params(**MIGRATION)))
    assert r1["DetGain"][DET] - r0["DetGain"][DET] == pytest.approx(50.0, rel=1e-9)
    assert out1 - out0 == pytest.approx(50.0 / 10.0 - 2.0, rel=1e-9)
    assert r1["DetLoss"][DET] == pytest.approx((2.0 + out1) * s1[DET], rel=1e-12)
    assert r1["DetLoss"][DET] == pytest.approx(r1["DetGain"][DET], rel=1e-9)


def test_the_base_holds_with_detritus_migration():
    out = EC.rsim_run(_scenario(_params(**MIGRATION)), method="RK4")
    np.testing.assert_allclose(out.out_Biomass[-1], out.out_Biomass[0], rtol=1e-9)


def test_a_constant_inflow_damps_the_pools_response():
    """Phyto's production halves in year 2. Det sits near its equilibrium
    (inflow / outflow rate): with 50 immigrating it keeps (50 + T) / 75.9 x
    10 for a flow T from the food web, against T / 70.9 x 10 without."""

    def run(params):
        sc = _scenario(params)
        sc.forcing.ForcedSearch[12:, PHYTO] = 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return EC.rsim_run(sc, method="RK4")

    plain, migrating = run(_params()), run(_params(**MIGRATION))
    drop_plain = plain.out_Biomass[-1, DET] / plain.out_Biomass[0, DET]
    drop_migrating = migrating.out_Biomass[-1, DET] / migrating.out_Biomass[0, DET]
    assert drop_plain < 0.9  # the pool responds
    t = drop_plain * 709.0  # the food web's flow at the end, near equilibrium
    expected = (50.0 + t) / (709.0 + 50.0)
    assert drop_migrating == pytest.approx(expected, rel=2e-2)
    assert drop_migrating > drop_plain + 0.01


def test_emigration_beyond_the_surplus_drains_the_pool(monkeypatch):
    """Emigration 1000 > the 709 inflow: DetritusOut stops at 1e-20 and the
    pool loses Emig x B = 1000 a year."""
    rates, det_out, s = _first_call(
        monkeypatch, _scenario(_params(Emigration={"Det": 1000.0}))
    )
    assert det_out == pytest.approx(1e-20)
    assert rates["DetLoss"][DET] == pytest.approx(100.0 * s[DET], rel=1e-9)
    assert rates["DetGain"][DET] < rates["DetLoss"][DET]


def test_a_detritus_emigration_rate_counts_where_emigration_is_0(monkeypatch):
    """Emig = EmigRate where only a rate is entered (vb 4535): EmigRate 2
    gives Emigration 20's DetritusOut, and Emigration wins over a rate."""
    _, by_flow, _ = _first_call(
        monkeypatch, _scenario(_params(Emigration={"Det": 20.0}))
    )
    _, by_rate, _ = _first_call(monkeypatch, _scenario(_params(EmigRate={"Det": 2.0})))
    _, both, _ = _first_call(
        monkeypatch,
        _scenario(_params(Emigration={"Det": 20.0}, EmigRate={"Det": 7.0})),
    )
    assert by_flow == pytest.approx(70.9 - 2.0, rel=1e-6)
    assert by_rate == pytest.approx(by_flow, rel=1e-12)
    assert both == pytest.approx(by_flow, rel=1e-12)


def test_the_rpath_rule_reports_and_ignores_detritus_migration(monkeypatch, caplog):
    r0, _, _ = _first_call(monkeypatch, _scenario(_params(), rule="rpath"))
    with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim"):
        sc = _scenario(_params(**MIGRATION), rule="rpath")
    assert "detritus" in caplog.text.lower() and "migration" in caplog.text.lower()
    r1, _, _ = _first_call(monkeypatch, sc)
    assert r1["DetGain"][DET] == pytest.approx(r0["DetGain"][DET], rel=1e-12)


def test_the_fast_group_test_counts_detritus_immigration():
    """Det B 40: 709 / 40 = 17.7 a year is integrated; with 400 immigrating
    EwE's first loss is 1109, 27.7 a year > 24, so Det is solved at its
    equilibrium."""
    assert _scenario(_params(det_b=40.0)).params.NoIntegrate[DET] == 0
    sc = _scenario(_params(det_b=40.0, Immigration={"Det": 400.0}))
    assert sc.params.NoIntegrate[DET] == 1


def test_ecospace_uniform_patches_follow_rsim_run():
    """Each patch takes the Ecosim terms on its densities, so uniform patches
    follow rsim_run."""
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    def scenario():
        sc = _scenario(_params(**MIGRATION))
        sc.forcing.ForcedSearch[12:, PHYTO] = 0.5
        return sc

    n = scenario().params.NUM_GROUPS + 1
    eco = EcospaceParams(
        grid=create_1d_grid(n_patches=2, spacing=1.0),
        habitat_preference=np.ones((n, 2)),
        habitat_capacity=np.ones((n, 2)),
        dispersal_rate=np.zeros(n),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = EC.rsim_run(scenario(), method="RK4")
        got = rsim_run_spatial(scenario(), ecospace=eco, years=range(1, 3))
    np.testing.assert_allclose(
        got.out_Biomass, expected.out_Biomass, rtol=1e-6, atol=1e-10
    )


# --- EwE 6.7.144 oracle: LT2022 with ewe-oracle patches/lt2022_detritus_migration.sql
# Detritus (B 1.8, DtImports 1000) gets Immigration 2000 and Emigration 500;
# run 43276892568 (scenario 16, dataset 35). Ecopath is unchanged (detritus EE
# 0.124853089, export 5366.09; EwE only reports NetMigration -1500). Without
# detritus migration PyPath was 2.1 % off at the median and 44 % at worst,
# the detritivores at month 144 (Worms 5.78 against 8.60).
DB = Path(__file__).parents[3] / "Data" / "LT2022_detritus_migration.eweaccdb"
EWE_BIOMASS = {
    "Detritus": {1: 1.80000448, 12: 1.79992509, 60: 1.54495883, 144: 1.07003784},
    "Worms": {1: 14.2701902, 12: 14.2735929, 60: 12.0070944, 144: 8.60052395},
    "Meiobenthos": {1: 20.0005283, 12: 19.9970093, 60: 14.9973049, 144: 9.4139185},
    "Saduria entomon": {
        1: 2.00010157,
        12: 2.01355863,
        60: 1.56991744,
        144: 0.504553616,
    },
}


def _db():
    if not DB.exists():
        pytest.skip(f"{DB.name} not found")
    return str(DB)


@pytest.mark.integration
def test_lt2022_reader_and_ecopath_follow_the_ewe_oracle(caplog):
    from pypath.core.ecopath import rpath
    from pypath.io.ewemdb import read_ewemdb

    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        params = read_ewemdb(_db())
    assert "migrat" not in caplog.text.lower()
    m = params.model.set_index(params.model["Group"].astype(str).str.strip())
    assert m.loc["Detritus", "Immigration"] == pytest.approx(2000.0)
    assert m.loc["Detritus", "Emigration"] == pytest.approx(500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    names = [str(g).strip() for g in model.Group]
    assert model.EE[names.index("Detritus")] == pytest.approx(0.124853089, rel=1e-5)


@pytest.mark.integration
def test_lt2022_ecosim_follows_the_ewe_oracle():
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(_db(), scenario=16, timeseries_dataset=35)
        out = EC.rsim_run(sc, method="RK4")
    names = [str(n).strip() for n in sc.params.spname]
    for group, months in EWE_BIOMASS.items():
        j = names.index(group)
        for month, ewe in months.items():
            assert out.out_Biomass[month, j] == pytest.approx(ewe, rel=2e-3), (
                group,
                month,
            )
