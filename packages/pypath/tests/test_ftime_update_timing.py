"""Which month's consumption drives the monthly feeding-time update.

EwE6 accumulates EatenByAvg and PredAvg right after the first Derivt of the
month's rk4 step, i.e. at the start-of-month state, and updates
Ftime <- FTLL + (1 - FTLL) * Ftime * (1 - adj + adj * Qopt / CBlast) at the end
of the step with CBlast = EatenByAvg / PredAvg (cEcoSimModel.vb rk4, ~1671-1686
and ~1761-1773). PyPath used the consumption of the end-of-month state.
"""

from pathlib import Path

import numpy as np
import pytest

import pypath.core.ecosim_deriv as D
from pypath.core.ecosim import rsim_run, rsim_scenario

from .test_fast_group_rule import FISH, PHYTO, _stiff_model

DB_PATH = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"


def _ftime_by_month(monkeypatch, scenario):
    """Ftime the derivative sees in the first stage of every month (RK4)."""
    seen = []
    real = D.deriv_vector

    def spy(state, params, forcing, fishing, *a, **k):
        seen.append(np.array(forcing["Ftime"], float))
        return real(state, params, forcing, fishing, *a, **k)

    monkeypatch.setattr(D, "deriv_vector", spy)
    rsim_run(scenario, method="RK4")
    per_month = len(seen) // (scenario.forcing.ForcedBio.shape[0])
    return [seen[m * per_month] for m in range(len(seen) // per_month)]


def _perturbed_stiff_scenario(rule):
    model, params = _stiff_model()
    sc = rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)
    sc.params.FtimeAdj[FISH] = 0.5
    sc.forcing.ForcedSearch[:, PHYTO] = 0.5  # food chain shrinks during month 1
    return sc


def test_ewe_rule_updates_ftime_from_start_of_month_consumption(monkeypatch):
    # Month 1 starts at the Ecopath balance, so its start-of-month consumption
    # is the base: the update leaves Fish at Ftime 1 for month 2.
    ft = _ftime_by_month(monkeypatch, _perturbed_stiff_scenario("ewe"))
    assert ft[0][FISH] == pytest.approx(1.0)
    assert ft[1][FISH] == pytest.approx(1.0, abs=1e-9)
    assert abs(ft[2][FISH] - 1.0) > 1e-4  # month 2's own consumption moves it


def test_rpath_rule_keeps_its_ftime_update(monkeypatch):
    ft = _ftime_by_month(monkeypatch, _perturbed_stiff_scenario("rpath"))
    assert abs(ft[1][FISH] - 1.0) > 1e-4


# EwE 6.7.144 oracle (headless run, LT2022 scenario 16 + dataset 35):
# FeedingTime(t) is recorded after the end-of-month update, so it is the Ftime
# used in month t + 1.
EWE_ORACLE_FTIME = {
    "Mobile crustaceans": (1.0207062, 1.0548308, 1.0945579),
    "Herring": (1.0219719, 1.0703355, 1.1430068),
    "Sprat": (1.0232178, 1.0726314, 1.1461012),
}


@pytest.mark.integration
def test_lt2022_feeding_time_follows_the_ewe_oracle(monkeypatch):
    if not DB_PATH.exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    sc = ecosim_scenario_from_ewemdb(str(DB_PATH), scenario=16, timeseries_dataset=35)
    names = [str(n).strip() for n in sc.params.spname]
    ft = _ftime_by_month(monkeypatch, sc)
    for group, values in EWE_ORACLE_FTIME.items():
        i = names.index(group)
        for t, expected in zip((14, 15, 16), values):
            # ft[t] is the Ftime used in month t + 1 (0-based list). The
            # end-of-month update is off by ~3 % at t = 14; what remains here
            # (<= 0.21 %, Herring at t = 16) is deviation D1 building up.
            assert ft[t][i] == pytest.approx(expected, rel=5e-3), (group, t)


def test_ewe_rule_ftime_update_uses_the_derivatives_consumption(monkeypatch):
    """EwE's CBlast is the Derivt consumption (prey Ftime, mediation included).

    _compute_Q_matrix leaves prey Ftime out, so a predator whose prey forages
    more than at base looked exactly as fed as at base.
    """
    from pypath.core.ecosim import _ftime_update_rpath

    from .test_fast_group_rule import ZOO

    model, params = _stiff_model()
    sc = rsim_scenario(model, params, years=range(1, 3))
    sc.params.FtimeAdj[FISH] = 0.5
    sc.params.FtimeAdj[ZOO] = 0.0
    sc.start_state.Ftime[ZOO] = 1.3  # Fish's prey is more exposed from month 1
    first = {}
    seen = []
    real = D.deriv_vector

    def spy(state, params_, forcing, fishing, *a, **k):
        out = real(state, params_, forcing, fishing, *a, **k)
        if "rates" not in first and k.get("rates") is not None:
            first.update(rates=k["rates"], state=np.array(state, float))
        seen.append(np.array(forcing["Ftime"], float))
        return out

    monkeypatch.setattr(D, "deriv_vector", spy)
    rsim_run(sc, method="RK4")
    fg = first["rates"]["FoodGain"][FISH]
    expected = _ftime_update_rpath(
        1.0,
        0.5,
        sc.params.FtimeQBOpt[FISH],
        fg,
        first["state"][FISH],
        lower=float(sc.params.ForagingTimeLowerLimit),
        cap=float(sc.params.FtimeMax[FISH]),
    )
    per_month = len(seen) // sc.forcing.ForcedBio.shape[0]
    assert abs(expected - 1.0) > 1e-3  # the prey's Ftime does move the update
    assert seen[per_month][FISH] == pytest.approx(expected, rel=1e-9)
