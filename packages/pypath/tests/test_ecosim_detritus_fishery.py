"""Fishing on a detritus pool.

Rpath counts every group's catch in its loss (``FishingLoss`` in
``LossPropToB``, ``ecosim.cpp``), detritus pools included, so a fished pool
loses its catch. EwE 6.7.144's detritus Derivt has no fishing term
(``loss = Eatenof + (Emig + DetritusOut) x B``, ``cEcoSimModel.vb:3121``),
so its Ecosim ignores the catch. The 'rpath' rule follows Rpath; the 'ewe'
rule follows EwE and warns when the scenario is built.
"""

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import _forced_fishing_rate, rsim_run, rsim_scenario
from pypath.core.ecosim_continuous import ContinuousRHS
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.params import create_rpath_params
from tests.continuous_support import REF, needs_reco, run_dicts, scenario

pytestmark = needs_reco

DETRITUS, DISCARDS = 21, 22  # REco's pools (1-based rsim indices)
SWITCHES = (
    "RK4_FAST_EQUILIBRIUM",
    "STANZA_STAGE_RATES",
    "FTIME_START_OF_MONTH",
    "RK4_STEPS",
)


@lru_cache(maxsize=None)
def _reco(landings=()):
    """Rpath's REco with ``landings``, ``((group, fleet, value), ...)``,
    added to the model table; returns ``(model, params)``, shared, so they
    must not be mutated (``scenario`` builds a fresh scenario)."""
    df = pd.read_csv(REF / "model_params.csv")
    for group, fleet, value in landings:
        df.loc[df["Group"] == group, fleet] = value
    params = create_rpath_params(df["Group"].tolist(), df["Type"].tolist())
    params.model = df
    params.diet = pd.read_csv(REF / "diet_matrix.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


LANDING = (("Detritus", "Trawlers", 5.0),)
TWO_POOLS = (("Detritus", "Trawlers", 5.0), ("Discards", "Midwater", 0.01))
# Trawlers discard 1 of Detritus; their discards go to the Discards pool
DISCARDED = (("Detritus", "Trawlers.disc", 1.0),)


def _deriv(landings, rule):
    """Month 1's derivative and rates at the start state, effort 1."""
    state, pd_, fd, fi = run_dicts(scenario(_reco(landings), rule=rule, effort=1.0))
    rates = {}
    deriv = deriv_vector(state.copy(), pd_, fd, fi, rates=rates)
    return deriv, rates


def test_a_fished_pool_loses_its_catch_under_the_rpath_rule():
    """Rpath's ``rsim.deriv(sc, 1, 1)`` on REco with Trawlers landing 5 from
    Detritus gives Detritus ``DerivT = -5`` and ``FishingLoss = 5``."""
    plain, _ = _deriv((), "rpath")
    fished, rates = _deriv(LANDING, "rpath")
    assert rates["FishingLoss"][DETRITUS] == pytest.approx(5.0, rel=1e-12)
    assert fished[DETRITUS] == pytest.approx(-5.0, abs=1e-9)
    assert fished[DETRITUS] - plain[DETRITUS] == pytest.approx(-5.0, rel=1e-12)
    assert rates["DetLoss"][DETRITUS] - rates["DetGain"][DETRITUS] == pytest.approx(
        -fished[DETRITUS], rel=1e-12
    )


def test_the_ewe_rule_ignores_the_catch_as_ewe_does():
    plain, _ = _deriv((), "ewe")
    fished, rates = _deriv(LANDING, "ewe")
    assert rates["FishingLoss"][DETRITUS] == pytest.approx(5.0, rel=1e-12)
    assert fished[DETRITUS] == plain[DETRITUS]


def _ewe_fishery_warnings(landings, rule):
    model, params = _reco(landings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rsim_scenario(model, params, years=range(1, 3), fast_group_rule=rule)
    return [w for w in caught if "ignores fishing" in str(w.message)]


def test_the_ewe_rule_warns_once_about_a_detritus_fishery():
    hits = _ewe_fishery_warnings(TWO_POOLS, "ewe")
    assert len(hits) == 1
    assert issubclass(hits[0].category, UserWarning)
    message = str(hits[0].message)
    assert "Detritus" in message and "Discards" in message


def test_no_detritus_fishery_warning_under_the_rpath_rule_or_without_one():
    assert _ewe_fishery_warnings(TWO_POOLS, "rpath") == []
    assert _ewe_fishery_warnings((), "ewe") == []


def _rhs(scen):
    """``(rhs, y)``: month 1's continuous RHS of ``scen`` (with year 1's
    forced F, as ``rsim_run`` sets it) and a state away from base (groups
    alternately 1.3 and 0.7 times B0)."""
    state, pd_, fd, fi = run_dicts(scen)
    rate, replace = _forced_fishing_rate(scen.fishing, 0, state, pd_["NUM_GROUPS"])
    if rate is not None:
        fd["ForcedF"], fd["ForcedFReplace"] = rate, replace
    n_bio = pd_["NUM_LIVING"] + pd_["NUM_DEAD"]
    switches = {k: getattr(scen.params, k, None) for k in SWITCHES}
    b0 = np.array(pd_["Bbase"], dtype=float)
    rhs = ContinuousRHS(pd_, fi, n_bio, b0, switches=switches)
    n1 = pd_["NUM_GROUPS"] + 1
    rhs.set_month(fd, state, np.zeros(n1, dtype=bool), np.zeros(n1))
    y0 = b0[1 : n_bio + 1]
    return rhs, y0 * np.where(np.arange(len(y0)) % 2 == 0, 1.3, 0.7)


def _forced_f_on_detritus():
    scen = scenario(_reco(()), rule="rpath", effort=1.0)
    scen.fishing.ForcedFRate[:, DETRITUS] = 0.1
    return scen


CASES = {
    "landed": lambda: scenario(_reco(LANDING), rule="rpath", effort=1.0),
    "two pools": lambda: scenario(_reco(TWO_POOLS), rule="rpath", effort=1.0),
    "discarded": lambda: scenario(_reco(DISCARDED), rule="rpath", effort=1.0),
    "forced F": _forced_f_on_detritus,
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_system_budget_closes_with_a_detritus_fishery(case):
    rhs, y = _rhs(CASES[case]())
    r = rhs.rates(y)
    assert r["FishingLoss"][DETRITUS] > 0.0
    scale = float(np.abs(r["Production"]).sum() + r["FoodGain"].sum())
    gap = abs(float(rhs(0.0, y)[: rhs.n_bio].sum()) - rhs.exchange(r, y))
    assert gap <= 1e-12 * scale, gap / scale


def test_the_month_budget_closes_with_a_detritus_fishery():
    scen = scenario(_reco(LANDING), rule="rpath", effort=1.0)
    out = rsim_run(scen, method="lsoda", years=range(1, 2))
    for m in out.solver_info["months"]:
        assert abs(m["budget_residual"]) < 1e-5, (m["month"], m["budget_residual"])
        assert m["pool_budget_residual"] < 1e-5, (m["month"], m["pool_budget_residual"])
