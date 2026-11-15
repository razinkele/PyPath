"""Feeding time as a state of the continuous pathway."""

import copy

import numpy as np
import pytest

import pypath.core.ecosim as ecosim_mod
import pypath.core.ecosim_continuous as cont_mod
from pypath.core.ecosim import _ftime_update_rpath, _risk_qopt, _run_setup, rsim_run
from pypath.core.ecosim_continuous import (
    ContinuousRHS,
    FeedingTime,
    feeding_time,
    ftime_target,
    hden_index,
)
from tests.continuous_support import (
    needs_reco,
    reco_plain,
    reco_stanzas,
    run_dicts,
    scenario,
)

pytestmark = needs_reco

SWITCHES = (
    "RK4_FAST_EQUILIBRIUM",
    "STANZA_STAGE_RATES",
    "FTIME_START_OF_MONTH",
    "RK4_STEPS",
)


def _ftime_scenario(reco, adj=0.5, cap_group=None, effort=2.0):
    scen = scenario(reco, rule="ewe", effort=effort)
    p = scen.params
    p.FtimeAdj[1 : p.NUM_LIVING + 1] = adj
    p.RiskTime[13] = 0.3  # Foragefish1
    if cap_group is not None:
        p.FtimeMax = np.full(p.NUM_GROUPS + 1, 2.0)
        p.FtimeMax[cap_group] = 1.0
    return scen


def _build(scen, forced=None):
    """``(rhs, pd, fd, state, feeding, y0)`` for month 1 of ``scen``."""
    state, pd, fd, fi = run_dicts(scen)
    n_bio = pd["NUM_LIVING"] + pd["NUM_DEAD"]
    feeding = feeding_time(scen.params, _run_setup(scen, range(1, 2)).risk0)
    switches = {k: getattr(scen.params, k, None) for k in SWITCHES}
    b0 = np.array(pd["Bbase"], dtype=float)
    rhs = ContinuousRHS(pd, fi, n_bio, b0, switches=switches, feeding=feeding)
    n1 = pd["NUM_GROUPS"] + 1
    mask = np.zeros(n1, dtype=bool)
    vals = np.zeros(n1)
    for g, v in (forced or {}).items():
        mask[g], vals[g] = True, v
    rhs.set_month(fd, state, mask, vals)
    y0 = np.concatenate([b0[1 : n_bio + 1], fd["Ftime"][feeding.groups]])
    return rhs, pd, fd, state, feeding, y0


def _wiggled(y0, n_bio):
    y = y0.copy()
    y[:n_bio] *= np.where(np.arange(n_bio) % 2 == 0, 1.3, 0.7)
    return y


def _capped_group(reco):
    """The group whose raw target at Ft = 1 most exceeds 1 (> 1 asserted).

    Capping that group's FtimeMax at 1 makes the cap bind at the start.
    (The brief's Ft = 2 against FtimeMax 2 cannot bind here: consumption is
    proportional to Ft, so no REco group's raw target exceeds 2 from Ft = 2.)
    """
    rhs, pd, fd, state, feeding, y0 = _build(_ftime_scenario(reco))
    n_bio = rhs.n_bio
    y = _wiggled(y0, n_bio)
    r = rhs.rates(y)
    full = state.copy()
    full[1 : n_bio + 1] = y[:n_bio]
    g = feeding.groups
    qopt = feeding.qbopt[g].copy()
    qopt[g == 13] = _risk_qopt(
        feeding.qbopt[13],
        feeding.risk_time[13],
        feeding.risk0[13],
        r["FoodLoss"][13],
        full[13],
        feeding.mo[13],
    )
    raw = ftime_target(
        y[n_bio:],
        feeding.adj[g],
        qopt,
        r["FoodGain"][g],
        hden_index(full, rhs._pd, fd)[g],
        feeding.lower,
        np.full(len(g), 1e9),
    )
    best = int(g[np.argmax(raw)])
    assert raw.max() > 1.0
    return best


@pytest.mark.parametrize("reco", [reco_plain, reco_stanzas])
def test_one_euler_month_is_the_monthly_rule(reco):
    capped = _capped_group(reco()) if reco is reco_plain else None
    scen = _ftime_scenario(reco(), cap_group=capped)
    rhs, pd, fd, state, feeding, y0 = _build(scen)
    n_bio = rhs.n_bio
    y = _wiggled(y0, n_bio)
    zero = 13 if capped != 13 else 12
    y[zero - 1] = 0.0  # a group at B = 0 keeps its feeding time
    dy = rhs(0.0, y)
    r = rhs.rates(y)
    full = state.copy()
    full[1 : n_bio + 1] = y[:n_bio]
    index = hden_index(full, rhs._pd, fd)
    assert np.array_equal(r["Ftime"][feeding.groups], np.clip(y[n_bio:], 0, 2))
    for k, g in enumerate(feeding.groups):
        qopt = _risk_qopt(
            feeding.qbopt[g],
            feeding.risk_time[g],
            feeding.risk0[g],
            r["FoodLoss"][g],
            full[g],
            feeding.mo[g],
        )
        want = _ftime_update_rpath(
            y[n_bio + k],
            feeding.adj[g],
            qopt,
            r["FoodGain"][g],
            index[g],
            lower=feeding.lower,
            cap=feeding.cap[g],
        )
        got = y[n_bio + k] + dy[n_bio + k] / 12
        assert got == pytest.approx(want, rel=1e-14), g
    assert dy[n_bio + list(feeding.groups).index(zero)] == 0.0
    if capped is not None:
        k = list(feeding.groups).index(capped)
        assert y[n_bio + k] == 1.0 == feeding.cap[capped]
        assert dy[n_bio + k] == 0.0  # target clipped to the cap, Ft at the cap


def test_ftime_target_keeps_ft_where_a_rate_is_not_positive():
    ft = np.array([1.0, 1.0, 1.0, 1.0])
    out = ftime_target(
        ft,
        np.full(4, 0.5),
        np.array([1.0, 1.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0, 1.0]),
        0.1,
        np.full(4, 2.0),
    )
    assert np.array_equal(out[:3], ft[:3])
    assert 0.1 <= out[3] <= 2.0


def test_feeding_time_selects_living_groups_with_adjustment():
    scen = scenario(reco_plain())
    p = scen.params
    assert feeding_time(p, None) is None
    p.FtimeAdj[3] = 0.5
    p.FtimeAdj[p.NUM_LIVING + 1] = 0.5  # detritus: not a feeding group
    f = feeding_time(p, None)
    assert isinstance(f, FeedingTime)
    assert f.groups.tolist() == [3]
    assert np.array_equal(f.cap, np.full(p.NUM_GROUPS + 1, 2.0))
    assert not f.risk_time.any() and not f.risk0.any()
    with pytest.raises(AttributeError):
        f.lower = 0.5


def test_feeding_time_stays_within_its_bounds():
    scen = _ftime_scenario(reco_plain(), adj=1.0, effort=3.0)
    out = rsim_run(scen, method="lsoda", years=range(1, 2))
    info = out.solver_info
    assert info["ftime_groups"]
    lower = scen.params.ForagingTimeLowerLimit
    cap = scen.params.FtimeMax[info["ftime_groups"]].max()
    tol = 1e-6
    for m in info["months"]:
        lo, hi = m["ftime_range"]
        assert lower - tol <= lo <= hi <= cap + tol
    assert np.all(np.isfinite(out.end_state.Ftime))
    assert np.all(np.isfinite(out.out_Biomass))


def test_feeding_time_is_integrated(monkeypatch):
    scen = _ftime_scenario(reco_plain())
    calls = {"rule": 0, "atol": [], "y_end": []}

    def spy_rule(*args, **kwargs):
        calls["rule"] += 1
        return _ftime_update_rpath(*args, **kwargs)

    real_integrate = cont_mod.integrate_month

    def spy_integrate(rhs, y0, dt, method, rtol, atol, **kwargs):
        calls["atol"].append((np.array(atol), rtol))
        res = real_integrate(rhs, y0, dt, method, rtol, atol, **kwargs)
        calls["y_end"].append(res.y_end)
        return res

    monkeypatch.setattr(ecosim_mod, "_ftime_update_rpath", spy_rule)
    monkeypatch.setattr(cont_mod, "integrate_month", spy_integrate)
    out = rsim_run(scen, method="lsoda", years=range(1, 2))
    assert calls["rule"] == 0
    groups = np.array(out.solver_info["ftime_groups"])
    n_bio = scen.params.NUM_LIVING + scen.params.NUM_DEAD
    assert len(groups) == scen.params.NUM_LIVING
    assert np.array_equal(out.end_state.Ftime[groups], calls["y_end"][-1][n_bio:])
    for atol, rtol in calls["atol"]:
        assert len(atol) == n_bio + len(groups)
        assert np.all(atol[n_bio:] == 1e-2 * rtol)
    first, last = out.solver_info["months"][0], out.solver_info["months"][-1]
    assert first["ftime_range"] != last["ftime_range"]


def test_reusing_a_scenario_with_feeding_time_leaves_rk4_unchanged():
    scen = _ftime_scenario(reco_plain())
    ftime_before = scen.start_state.Ftime.copy()
    before = rsim_run(scen, method="RK4", years=range(1, 2))
    rsim_run(scen, method="lsoda", years=range(1, 2))
    assert np.array_equal(scen.start_state.Ftime, ftime_before)
    after = rsim_run(scen, method="RK4", years=range(1, 2))
    assert np.array_equal(before.out_Biomass, after.out_Biomass)
    assert np.array_equal(before.end_state.Ftime, after.end_state.Ftime)


def test_an_evaluation_leaves_the_months_ftime_alone():
    scen = _ftime_scenario(reco_plain())
    rhs, pd, fd, state, feeding, y0 = _build(scen)
    held = fd["Ftime"].copy()
    fd_before = copy.deepcopy(fd)
    y = _wiggled(y0, rhs.n_bio)
    y[rhs.n_bio :] = 5.0  # out of bounds: clipped in the copy only
    rhs(0.0, y)
    rhs.rates(y)
    assert np.array_equal(fd["Ftime"], held)
    assert np.array_equal(fd["Ftime"], fd_before["Ftime"])


def test_a_forced_group_feeds_the_others():
    scen = _ftime_scenario(reco_plain())
    rhs0, pd, fd, state, feeding, y0 = _build(scen)
    n_bio = rhs0.n_bio
    predator = 13
    prey = 12
    outs = []
    for factor in (1.0, 0.8):
        rhs, pd, fd, state, feeding, y0 = _build(
            scen, forced={predator: factor * float(pd["Bbase"][predator])}
        )
        state_before = state.copy()
        outs.append(rhs(0.0, y0))
        assert np.array_equal(state, state_before)
        assert outs[-1][predator - 1] == 0.0  # forced: no derivative
        k = n_bio + list(feeding.groups).index(predator)
        assert np.isfinite(outs[-1][k])
    assert outs[0][prey - 1] != outs[1][prey - 1]
    assert outs[0][n_bio:].tolist() != outs[1][n_bio:].tolist()


def test_set_month_needs_a_mask_for_every_group():
    scen = _ftime_scenario(reco_plain())
    rhs, pd, fd, state, feeding, y0 = _build(scen)
    n1 = pd["NUM_GROUPS"] + 1
    with pytest.raises(ValueError, match="NUM_GROUPS"):
        rhs.set_month(fd, state, np.zeros(rhs.n_bio, dtype=bool), np.zeros(n1))
    with pytest.raises(ValueError, match="NUM_GROUPS"):
        rhs.set_month(fd, state, np.zeros(n1, dtype=bool), np.zeros(rhs.n_bio))
