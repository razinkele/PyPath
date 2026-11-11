"""EwE6's predator effect on feeding time (EcosimScenarioGroup.RiskTime).

EwE6 splits a group's optimal consumption into a part that ignores
predation risk and a part that scales with it (cEcoSimModel.vb 3309-3314,
1765-1766):

    Qmain = (1 - RiskTime) x Cbase
    Qrisk = RiskTime x Cbase x (StartEatenOf / B0 + mo)
    RiskRate = EatenOfAvg / BBAvg + mo + 1e-10
    Qopt = Qmain + Qrisk / RiskRate

and the monthly feeding-time update uses Qopt / CBlast in place of
Cbase / CBlast. EatenOfAvg is the predation of the step's first Derivt (the
start-of-month state), BBAvg the end-of-month biomass after RK4 and before
the stanza update. LT2022 scenarios 10-12 set RiskTime 0.5 or 1 on 16
groups. PyPath ignored it.
"""

import warnings

import numpy as np
import pytest

ZOO, FISH = 2, 3  # spname indices (0 = Outside)


def test_risk_qopt_is_ewes_expression():
    from pypath.core.ecosim import _risk_qopt

    cbase, rt, mo = 50.0, 0.5, 0.25
    eaten_of0, b0 = 8.0, 20.0
    eaten_of, b_end = 12.0, 18.0
    risk0 = eaten_of0 / b0 + mo + 1e-10
    q_main = (1 - rt) * cbase
    q_risk = rt * cbase * risk0
    expected = q_main + q_risk / (eaten_of / b_end + mo + 1e-10)
    got = _risk_qopt(cbase, rt, risk0, eaten_of, b_end, mo)
    assert got == pytest.approx(expected, rel=1e-12)
    # at the base risk Qopt is Cbase; without RiskTime always Cbase
    assert _risk_qopt(cbase, rt, risk0, eaten_of0, b0, mo) == pytest.approx(cbase)
    assert _risk_qopt(cbase, 0.0, risk0, eaten_of, b_end, mo) == cbase


def _scenario():
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(["Phyto", "Zoo", "Fish", "Det"], [1, 0, 0, 2])
    m = params.model
    m["Biomass"] = [100.0, 20.0, 5.0, 10.0]
    m["PB"] = [10.0, 2.0, 1.0, np.nan]
    m["QB"] = [np.nan, 20.0, 4.0, np.nan]
    m["EE"] = [0.8, 0.6, 0.5, np.nan]
    m["BioAcc"] = 0.0
    m["Unassim"] = [0.0, 0.2, 0.2, 0.0]
    m["Det"] = [1.0, 1.0, 1.0, 0.0]
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenario = rsim_scenario(rpath(params), params, years=range(1, 4))
    scenario.params.FtimeAdj[ZOO] = 0.5
    # the predator doubles: Zoo's predation risk rises
    scenario.forcing.ForcedBio[:, FISH] = 10.0
    return scenario


def _run(risk_time):
    from pypath.core.ecosim import rsim_run

    sc = _scenario()
    if risk_time is not None:
        sc.params.RiskTime = np.zeros(len(sc.params.FtimeAdj))
        sc.params.RiskTime[ZOO] = risk_time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(sc, method="RK4")


def test_zero_risk_time_changes_nothing():
    a, b = _run(None), _run(0.0)
    np.testing.assert_array_equal(a.out_Biomass, b.out_Biomass)


def test_risk_time_lowers_feeding_time_when_predation_rises():
    """More predators raise RiskRate, so Qopt falls below Cbase: the prey
    spends less time feeding and eats less than without RiskTime."""
    base, risky = _run(0.0), _run(1.0)
    assert risky.end_state.Ftime[ZOO] < base.end_state.Ftime[ZOO]
    assert not np.allclose(risky.out_Biomass[:, ZOO], base.out_Biomass[:, ZOO])


@pytest.mark.integration
def test_lt2022_scenario_10_reads_risk_time(caplog):
    import logging
    from pathlib import Path

    db = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"
    if not db.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        sc = ecosim_scenario_from_ewemdb(str(db), scenario=10, years=range(1, 3))
    names = [str(n).strip() for n in sc.params.spname]
    rt = dict(zip(names, sc.params.RiskTime))
    assert rt["Herring"] == 0.5
    assert rt["Cod"] == 1.0
    assert rt["Zooplankton"] == 0.0
    assert "RiskTime" not in caplog.text
