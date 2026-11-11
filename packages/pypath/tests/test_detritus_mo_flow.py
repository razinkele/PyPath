"""Other mortality reaches detritus unscaled by feeding time, as in EwE6.

With MoPred > 0 a group loses mo x (1 - MoPred + MoPred x Ftime) x B to
other mortality (cEcoSimModel.Derivt, vb 3081), but EwE6 sends the unscaled
mo x B (times any other-mortality forcing) to detritus (SimDetritusMT,
vb 3160-3166): its detritus budget is not closed when Ftime != 1. PyPath
sent the scaled flow. LT2022 has MoPred = 1 for every group, so after the
year-2 production step, when feeding times rise, PyPath's detritus ran high
(one-month error 5e-4 from EwE's own state) and the detritivores followed:
Meiobenthos and Worms were 5 % off the EwE oracle in scenario 15.
The 'rpath' rule keeps the closed budget.
"""

from pathlib import Path

import numpy as np
import pytest

import pypath.core.ecosim as EC
import pypath.core.ecosim_deriv as D
from pypath.core.ecosim import rsim_run, rsim_scenario

from .test_forcing_channels import _simple_model

PHYTO, ZOO, DET = 1, 2, 3
LT2022 = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"


def _scenario(rule):
    model, params = _simple_model(detritus_fate=True)  # all dead matter to Det
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)
    sc.params.MoPred[ZOO] = 1.0
    return sc


def _first_call(monkeypatch, run):
    import pypath.spatial.integration as SI

    cap = {}
    real = D.deriv_vector

    def spy(state, params, forcing, fishing, *a, **k):
        if "P" not in cap and "ForcedPrey" in forcing:
            cap.update(P=params, Fo=forcing, Fi=fishing, s=np.array(state, float))
        return real(state, params, forcing, fishing, *a, **k)

    for module in (D, EC, SI):
        monkeypatch.setattr(module, "deriv_vector", spy)
    run()
    monkeypatch.undo()
    return cap


def _rates(cap, ftime_zoo, forced_mort_zoo=1.0):
    fo = {
        k: (np.array(v, float, copy=True) if hasattr(v, "shape") else v)
        for k, v in cap["Fo"].items()
    }
    fo["Ftime"][ZOO] = ftime_zoo
    fo["ForcedMort"][ZOO] = forced_mort_zoo
    r = {}
    D.deriv_vector(cap["s"].copy(), cap["P"], fo, cap["Fi"], rates=r)
    return r


@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_other_mortality_to_detritus(monkeypatch, rule):
    """Zoo forages 1.5 x and dies of other causes 1.5 x (MoPred = 1); EwE's
    detritus still gets mo x B from it, Rpath's closed budget mo x 1.5 x B.
    Other-mortality forcing (x 2) reaches detritus under both rules."""
    sc = _scenario(rule)
    cap = _first_call(monkeypatch, lambda: rsim_run(sc, method="RK4"))
    mo, b = float(sc.params.MzeroMort[ZOO]), float(cap["s"][ZOO])
    assert mo > 0
    base = _rates(cap, 1.0)
    fed = _rates(cap, 1.5)
    forced = _rates(cap, 1.5, forced_mort_zoo=2.0)
    # The group's own loss follows Ftime under both rules.
    assert fed["MzeroLoss"][ZOO] == pytest.approx(1.5 * mo * b, rel=1e-12)
    # Detritus gain, without the egestion that changes with the intake.
    unassim = float(sc.params.UnassimRespFrac[ZOO])

    def det_from_mort(r, r0):
        return (r["DetGain"][DET] - unassim * r["FoodGain"][ZOO]) - (
            r0["DetGain"][DET] - unassim * r0["FoodGain"][ZOO]
        )

    scale = 1.0 if rule == "ewe" else 1.5
    assert det_from_mort(fed, base) == pytest.approx((scale - 1.0) * mo * b, abs=1e-9)
    assert det_from_mort(forced, base) == pytest.approx(
        (2.0 * scale - 1.0) * mo * b, rel=1e-9
    )


@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_ecospace_detritus_follows_the_same_rule(monkeypatch, rule):
    """One Ecospace patch reproduces rsim_run with MoPred = 1 and a feeding
    time that moves (FtimeAdj on Zoo, production forcing on Phyto)."""
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    sc = _scenario(rule)
    sc.params.FtimeAdj[ZOO] = 0.5
    sc.forcing.ForcedSearch[12:, PHYTO] = 0.8
    n = sc.params.NUM_GROUPS + 1
    eco = EcospaceParams(
        grid=create_1d_grid(n_patches=1, spacing=1.0),
        habitat_preference=np.ones((n, 1)),
        habitat_capacity=np.ones((n, 1)),
        dispersal_rate=np.zeros(n),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
    )
    flat = _first_call(monkeypatch, lambda: rsim_run(sc, method="RK4"))
    spatial = _first_call(
        monkeypatch, lambda: rsim_run_spatial(sc, ecospace=eco, years=range(1, 3))
    )
    key = "DETRITUS_MO_UNSCALED"
    assert spatial["P"].get(key) == flat["P"].get(key) == (rule == "ewe")


# EwE 6.7.144 oracle, LT2022 scenario 15 + dataset 35 (run 42895374068):
# biomass at the end of months 24, 36 and 72. Single precision.
EWE_ORACLE_S15_BENTHOS = {
    "Meiobenthos": (13.3075209, 39.1734962, 12.937706),
    "Worms": (10.4885883, 22.9381561, 11.7568111),
    "Detritus": (1.46585596, 2.60063839, 1.57182407),
}


@pytest.mark.integration
def test_lt2022_scenario_15_benthos_follow_the_ewe_oracle():
    if not LT2022.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import EwEDatabaseError, ecosim_scenario_from_ewemdb

    try:
        sc = ecosim_scenario_from_ewemdb(
            str(LT2022), scenario=15, timeseries_dataset=35
        )
    except EwEDatabaseError as e:
        pytest.skip(f"No EwE database driver: {e}")
    out = rsim_run(sc, method="RK4", years=range(1, 7))
    names = [str(n).strip() for n in sc.params.spname]
    for group, values in EWE_ORACLE_S15_BENTHOS.items():
        i = names.index(group)
        for month, ewe in zip((24, 36, 72), values):
            assert out.out_Biomass[month, i] == pytest.approx(ewe, rel=2e-3), (
                group,
                month,
            )
