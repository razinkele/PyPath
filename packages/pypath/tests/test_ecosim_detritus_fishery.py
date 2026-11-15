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


def _base_then_more_effort(scen):
    """Month 1's derivative and rates at the start state (which sets EwE's
    DetritusOut), then at a state away from base (groups alternately 1.3 and
    0.7 times B0) with every gear's effort doubled."""
    state, pd_, fd, fi = run_dicts(scen)
    base_rates, rates = {}, {}
    base = deriv_vector(state.copy(), pd_, fd, fi, rates=base_rates)
    y = state.copy()
    n_bio = pd_["NUM_LIVING"] + pd_["NUM_DEAD"]
    y[1 : n_bio + 1] *= np.where(np.arange(n_bio) % 2 == 0, 1.3, 0.7)
    more = dict(fd, ForcedEffort=np.asarray(fd["ForcedEffort"], dtype=float) * 2.0)
    away = deriv_vector(y, pd_, more, fi, rates=rates)
    return (base, base_rates), (away, rates)


def test_the_ewe_rule_ignores_the_catch_as_ewe_does():
    """Compared with the same scenario without the pool fishery. At the
    start state DetritusOut absorbs any change in a pool's loss, and as both
    scale with B it absorbs a fixed F x B at any state, so the second check
    doubles the effort: a catch applied under the EwE rule would move the
    pool's derivative by F x B there."""
    fished = _base_then_more_effort(scenario(_reco(LANDING), rule="ewe", effort=1.0))
    plain = _base_then_more_effort(
        _without_pool_fishery(scenario(_reco(LANDING), rule="ewe", effort=1.0))
    )
    assert fished[0][1]["FishingLoss"][DETRITUS] == pytest.approx(5.0, rel=1e-12)
    assert fished[1][1]["FishingLoss"][DETRITUS] > 0.0
    for (d_fished, _), (d_plain, _) in zip(fished, plain):
        np.testing.assert_array_equal(d_fished, d_plain)


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


def _forced_f_on_detritus(rule="rpath"):
    scen = scenario(_reco(()), rule=rule, effort=1.0)
    scen.fishing.ForcedFRate[:, DETRITUS] = 0.1
    return scen


CASES = {
    "landed": lambda: scenario(_reco(LANDING), rule="rpath", effort=1.0),
    "two pools": lambda: scenario(_reco(TWO_POOLS), rule="rpath", effort=1.0),
    "discarded": lambda: scenario(_reco(DISCARDED), rule="rpath", effort=1.0),
    "forced F": _forced_f_on_detritus,
    # the EwE rule ignores the catch, so nothing of it may leave the budget
    "ewe landed": lambda: scenario(_reco(LANDING), rule="ewe", effort=1.0),
    "ewe two pools": lambda: scenario(_reco(TWO_POOLS), rule="ewe", effort=1.0),
    "ewe forced F": lambda: _forced_f_on_detritus("ewe"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_system_budget_closes_with_a_detritus_fishery(case):
    rhs, y = _rhs(CASES[case]())
    r = rhs.rates(y)
    assert r["FishingLoss"][DETRITUS] > 0.0
    scale = float(np.abs(r["Production"]).sum() + r["FoodGain"].sum())
    gap = abs(float(rhs(0.0, y)[: rhs.n_bio].sum()) - rhs.exchange(r, y))
    assert gap <= 1e-12 * scale, gap / scale


MONTH_CASES = {
    "rpath landed": lambda: scenario(_reco(LANDING), rule="rpath", effort=1.0),
    "ewe landed": lambda: scenario(_reco(LANDING), rule="ewe", effort=1.0),
    "ewe forced F": lambda: _forced_f_on_detritus("ewe"),
}


@pytest.mark.parametrize("case", list(MONTH_CASES))
def test_the_month_budget_closes_with_a_detritus_fishery(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the EwE rule warns about forced F
        out = rsim_run(MONTH_CASES[case](), method="lsoda", years=range(1, 2))
    for m in out.solver_info["months"]:
        assert abs(m["budget_residual"]) < 1e-5, (m["month"], m["budget_residual"])
        assert m["pool_budget_residual"] < 1e-5, (m["month"], m["pool_budget_residual"])


# --- the EwE rule ignores the fishery entirely: no discards from a pool -----


def _without_pool_fishery(scen):
    """``scen`` with the catch rate of every fishing link from a pool set to
    0 (the same Ecopath model and Ecosim parameters otherwise)."""
    p = scen.params
    nl, n_bio = p.NUM_LIVING, p.NUM_LIVING + p.NUM_DEAD
    from_pool = (np.asarray(p.FishFrom) > nl) & (np.asarray(p.FishFrom) <= n_bio)
    from_pool[0] = False
    assert from_pool.any()
    p.FishQ = np.where(from_pool, 0.0, np.asarray(p.FishQ, dtype=float))
    return scen


def _two_derivs(scen):
    """Month 1's derivative and rates at the start state (which sets EwE's
    DetritusOut), then at a state away from base (groups alternately 1.3 and
    0.7 times B0)."""
    state, pd_, fd, fi = run_dicts(scen)
    base_rates, rates = {}, {}
    base = deriv_vector(state.copy(), pd_, fd, fi, rates=base_rates)
    y = state.copy()
    n_bio = pd_["NUM_LIVING"] + pd_["NUM_DEAD"]
    y[1 : n_bio + 1] *= np.where(np.arange(n_bio) % 2 == 0, 1.3, 0.7)
    away = deriv_vector(y, pd_, fd, fi, rates=rates)
    return (base, base_rates), (away, rates)


def _discarded(rule):
    return scenario(_reco(DISCARDED), rule=rule, effort=1.0)


def test_the_ewe_rule_sends_no_discards_from_a_fished_pool():
    """EwE6 sends discards to detritus from living groups only
    (SimDetritusMT, cEcoSimModel.vb:3154-3176): under the EwE rule the
    Discards pool gets nothing from Trawlers discarding 1 of Detritus. Under
    the 'rpath' rule it gets the discard, as Rpath's FishingGain[FishTo]."""
    for rule, extra in (("ewe", 0.0), ("rpath", 1.0)):
        (_, fished), _ = _two_derivs(_discarded(rule))
        (_, plain), _ = _two_derivs(_without_pool_fishery(_discarded(rule)))
        assert fished["FishingLoss"][DETRITUS] == pytest.approx(1.0, rel=1e-12)
        gained = fished["Discards"][DISCARDS] - plain["Discards"][DISCARDS]
        assert gained == pytest.approx(extra, rel=1e-12, abs=0.0), rule


def test_the_ewe_rule_leaves_every_derivative_unchanged_by_a_pool_fishery():
    fished = _two_derivs(_discarded("ewe"))
    plain = _two_derivs(_without_pool_fishery(_discarded("ewe")))
    for (d_fished, _), (d_plain, _) in zip(fished, plain):
        np.testing.assert_array_equal(d_fished, d_plain)


def test_the_ewe_rule_budget_closes_with_a_discarding_pool_fishery():
    rhs, y = _rhs(_discarded("ewe"))
    r = rhs.rates(y)
    assert r["FishingLoss"][DETRITUS] > 0.0
    scale = float(np.abs(r["Production"]).sum() + r["FoodGain"].sum())
    gap = abs(float(rhs(0.0, y)[: rhs.n_bio].sum()) - rhs.exchange(r, y))
    assert gap <= 1e-12 * scale, gap / scale
    out = rsim_run(_discarded("ewe"), method="lsoda", years=range(1, 2))
    for m in out.solver_info["months"]:
        assert abs(m["budget_residual"]) < 1e-5, (m["month"], m["budget_residual"])
        assert m["pool_budget_residual"] < 1e-5, (m["month"], m["pool_budget_residual"])


# --- forced F / forced catch on a pool under the EwE rule ------------------


def _forced_warnings(rule, frate=0.0, fcatch=0.0, group=DETRITUS):
    scen = scenario(_reco(()), rule=rule, effort=1.0)
    scen.fishing.ForcedFRate[:, group] = frate
    scen.fishing.ForcedCatch[:, group] = fcatch
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rsim_run(scen, method="RK4", years=range(1, 2))
    return [w for w in caught if "forced" in str(w.message).lower()]


def test_forced_fishing_on_a_pool_warns_once_under_the_ewe_rule():
    for kwargs in ({"frate": 0.1}, {"fcatch": 1.0}):
        hits = _forced_warnings("ewe", **kwargs)
        assert len(hits) == 1, kwargs
        assert issubclass(hits[0].category, UserWarning)
        assert "Detritus" in str(hits[0].message)


def test_no_forced_fishing_warning_under_the_rpath_rule_or_on_living_groups():
    assert _forced_warnings("rpath", frate=0.1) == []
    assert _forced_warnings("ewe", frate=0.1, group=5) == []
    assert _forced_warnings("ewe") == []


# --- EwE's fast-group test sees the derivative's pool: no pool fishery -----

FAST_CASES = {
    # 40 landed from Discards (B 1.2, inflow 0.6): the catch must not count
    # as the pool's loss
    "landed from a pool": (("Discards", "Midwater", 40.0),),
    # Trawlers discard 40 of Detritus into Discards: not an inflow, as EwE
    # sends discards from living groups only
    "discarded from a pool": (("Detritus", "Trawlers.disc", 40.0),),
}


@pytest.mark.parametrize("case", list(FAST_CASES))
def test_the_ewe_fast_group_test_ignores_a_pool_fishery(case):
    """Under the EwE rule the fishery takes nothing from the pool and sends
    no discards from it, so the fast-group test (EwE's first Derivt loss
    rate, threshold 24 /yr) must see the pool as without it: Discards stays
    integrated (0.5 /yr), as in plain REco."""
    fished = scenario(_reco(FAST_CASES[case]), rule="ewe", effort=1.0)
    plain = scenario(_reco(()), rule="ewe", effort=1.0)
    assert fished.params.NoIntegrate[DISCARDS] == 0
    np.testing.assert_array_equal(fished.params.NoIntegrate, plain.params.NoIntegrate)
