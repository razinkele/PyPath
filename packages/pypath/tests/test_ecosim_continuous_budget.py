"""Conservation on the continuous path (spec 5.5): the derivative's system
budget, the month budget in ``solver_info`` and the nutrient check."""

import numpy as np
import pytest

from pypath.core.ecosim import _run_setup, rsim_run
from pypath.core.ecosim_continuous import (
    ContinuousRHS,
    default_atol,
    feeding_time,
    integrate_month,
)
from pypath.core.ecosim_deriv import _rpath_detritus_derivs, deriv_vector
from tests.continuous_support import needs_reco, reco_plain, run_dicts, scenario

pytestmark = needs_reco

SWITCHES = (
    "RK4_FAST_EQUILIBRIUM",
    "STANZA_STAGE_RATES",
    "FTIME_START_OF_MONTH",
    "RK4_STEPS",
)


def _variant(scen):
    """PaddP 0.5, MoPred 0.5, start Ftime 0.8, EmigRate[12] 0.1 and
    Immigration 0.5 at group 12 (REco has none)."""
    p = scen.params
    nl = p.NUM_LIVING
    p.PaddP[1 : nl + 1] = 0.5
    p.MoPred[1 : nl + 1] = 0.5
    p.EmigRate[12] = 0.1
    immigration = np.zeros(p.NUM_GROUPS + 1)
    immigration[12] = 0.5
    p.Immigration = immigration
    scen.start_state.Ftime[:] = 0.8
    return scen


def _scen(case):
    rule = "rpath" if case == "rpath" else "ewe"
    scen = scenario(reco_plain(), rule=rule)
    return _variant(scen) if case == "ewe variant" else scen


def _build(scen, feeding=None, **kwargs):
    """``(rhs, pd, fd, y0)`` for month 1 of ``scen``; ``y0`` holds the base
    biomass of groups 1..n_bio, then the feeding times (with ``feeding``)."""
    state, pd, fd, fi = run_dicts(scen)
    n_bio = pd["NUM_LIVING"] + pd["NUM_DEAD"]
    switches = {k: getattr(scen.params, k, None) for k in SWITCHES}
    b0 = np.array(pd["Bbase"], dtype=float)
    rhs = ContinuousRHS(pd, fi, n_bio, b0, switches=switches, feeding=feeding, **kwargs)
    n1 = pd["NUM_GROUPS"] + 1
    rhs.set_month(fd, state, np.zeros(n1, dtype=bool), np.zeros(n1))
    y0 = b0[1 : n_bio + 1].copy()
    if feeding is not None:
        y0 = np.concatenate([y0, fd["Ftime"][feeding.groups]])
    return rhs, pd, fd, y0


def _alternating(y0):
    return y0 * np.where(np.arange(len(y0)) % 2 == 0, 1.3, 0.7)


def _scale(r):
    """T = sum |Production| + sum FoodGain."""
    return float(np.abs(r["Production"]).sum() + r["FoodGain"].sum())


def _gap(rhs, y):
    r = rhs.rates(y)
    scale = _scale(r)
    dy = rhs(0.0, y)
    return abs(float(dy[: rhs.n_bio].sum()) - rhs.exchange(r, y)), scale


@pytest.mark.parametrize("case", ["ewe", "rpath", "ewe variant"])
def test_the_derivative_closes_the_system_budget(case):
    rhs, pd, fd, y0 = _build(_scen(case))
    gap, scale = _gap(rhs, _alternating(y0))
    assert scale > 0
    assert gap <= 1e-12 * scale, (case, gap / scale)


def test_unclosed_corrections_leave_a_residual():
    """Without the corrections, other mortality reaches detritus at a rate the
    group does not lose (R2: unscaled by feeding time; R5: mo, not moTot), so
    the system budget leaks: exchange_rate is not derived from sum(dy)."""
    rhs, pd, fd, y0 = _build(_scen("ewe variant"), corrections=False)
    gap, scale = _gap(rhs, _alternating(y0))
    assert gap > 1e-6 * scale, gap / scale


DISCARDS, DETRITUS = 22, 21  # REco's pools


def _passing_scen(dpp=0.1):
    """REco under the EwE rule with Discards passing DetPassedProp x B on to
    Detritus along a link of DF 0.6 (REco itself has no link between pools;
    the EwE rule ignores a pool's links to itself and to outside)."""
    scen = _scen("ewe")
    p = scen.params
    k = int(np.flatnonzero(np.asarray(p.DetFrom) == DISCARDS)[0])
    p.DetTo[k] = DETRITUS
    p.DetFrac[k] = 0.6
    p.DetPassedProp[DISCARDS] = dpp
    return scen


def test_a_flow_between_pools_closes_the_budget():
    """On the corrected path (DF once) the passed flow is DetPassedProp x B,
    and the export is (DetritusOut - DetPassedProp) x B from the parameters."""
    rhs, pd, fd, y0 = _build(_passing_scen(0.1))
    y = _alternating(y0)
    r = rhs.rates(y)
    b_discards = y[DISCARDS - 1]
    still, *_ = _build(_passing_scen(0.0))
    passed = r["DetGain"][DETRITUS] - still.rates(y)["DetGain"][DETRITUS]
    assert passed == pytest.approx(0.1 * b_discards, rel=1e-12)
    det_out = rhs._pd["_det_out"][DISCARDS]
    assert det_out > 0.1  # no negative export here
    assert r["DetExport"][DISCARDS] == pytest.approx(
        (det_out - 0.1) * b_discards, rel=1e-12
    )
    gap, scale = _gap(rhs, y)
    assert gap <= 1e-12 * scale, gap / scale


def test_df_applied_twice_leaves_a_residual():
    """Without the corrections the flow is DetPassedProp x DF x B: 0.4 of
    DetPassedProp x B leaves Discards, reaches no pool and is not export."""
    rhs, pd, fd, y0 = _build(_passing_scen(0.1), corrections=False)
    assert "DETRITUS_DF_ONCE" not in rhs._pd
    y = _alternating(y0)
    r = rhs.rates(y)
    gap = float(rhs(0.0, y)[: rhs.n_bio].sum()) - rhs.exchange(r, y)
    assert gap == pytest.approx(-0.4 * 0.1 * y[DISCARDS - 1], rel=1e-9)


def test_negative_export_is_counted():
    """DetPassedProp above DetritusOut creates mass: the budget still closes
    (the export is negative), and every month counts the evaluations."""
    months = rsim_run(_passing_scen(0.1), method="lsoda", years=range(1, 2))
    assert all(m["negative_export"] == 0 for m in months.solver_info["months"])
    rhs, *_ = _build(_passing_scen(0.3))
    assert rhs._pd["_det_out"][DISCARDS] < 0.3
    out = rsim_run(_passing_scen(0.3), method="lsoda", years=range(1, 2))
    for m in out.solver_info["months"]:
        assert m["negative_export"] > 0, m["month"]
        assert abs(m["budget_residual"]) < 1e-5, (m["month"], m["budget_residual"])


def _n_bio(scen):
    return scen.params.NUM_LIVING + scen.params.NUM_DEAD


def test_the_month_budget_closes():
    scen = _scen("ewe variant")
    n_bio = _n_bio(scen)
    b0 = np.asarray(scen.params.B_BaseRef, dtype=float)[1 : n_bio + 1]
    out = rsim_run(
        scen,
        method="dop853",
        years=range(1, 2),
        solver_options={"rtol": 1e-10, "atol": default_atol(b0) * 1e-4},
    )
    month = out.solver_info["months"][0]
    assert abs(month["budget_residual"]) < 1e-8, month["budget_residual"]
    assert month["pool_budget_residual"] < 1e-7, month["pool_budget_residual"]


def _ftime_scen():
    scen = scenario(reco_plain(), rule="ewe")
    scen.params.FtimeAdj[1 : scen.params.NUM_LIVING + 1] = 0.5
    return scen


def test_solver_info_reports_the_budget():
    out = rsim_run(_ftime_scen(), method="lsoda", years=range(1, 2))
    assert out.solver_info["ftime_groups"]  # feeding-time states present
    months = out.solver_info["months"]
    assert len(months) == 12
    for m in months:
        assert abs(m["budget_residual"]) < 1e-5, (m["month"], m["budget_residual"])
        assert m["pool_budget_residual"] < 1e-5, m["month"]
        assert m["nutrient_floor"] == 0
        assert m["negative_export"] == 0
    scen = _ftime_scen()
    scen.forcing.ForcedBio[0:6, 12] = 0.8 * scen.params.B_BaseRef[12]
    months = rsim_run(scen, method="lsoda", years=range(1, 2)).solver_info["months"]
    for m in months[:6]:
        assert m["budget_residual"] is None and m["pool_budget_residual"] is None
    for m in months[6:]:
        assert abs(m["budget_residual"]) < 1e-5, (m["month"], m["budget_residual"])


def _nutrient_scen(shape):
    scen = _ftime_scen()
    scen.params.NutBaseFreeProp = 0.5
    n_months = scen.forcing.ForcedBio.shape[0]
    scen.forcing.NutForcing = np.full(n_months, shape)
    return scen


def test_free_plus_bound_nutrient_is_the_total():
    shape = 0.9
    scen = _nutrient_scen(shape)
    feeding = feeding_time(scen.params, _run_setup(scen, range(1, 2)).risk0)
    rhs, pd, fd, y0 = _build(scen, feeding=feeding)
    n_bio = rhs.n_bio
    assert fd["NutForcing"] == shape
    assert len(y0) > n_bio  # feeding-time states follow the biomass
    nodes = []
    real = rhs.rates

    def spy(y):
        r = real(y)
        nodes.append((np.array(y, copy=True), r))
        return r

    rhs.rates = spy
    atol = np.concatenate([default_atol(y0[:n_bio]), np.full(len(y0) - n_bio, 1e-8)])
    integrate_month(rhs, y0, 1.0 / 12.0, "lsoda", 1e-6, atol, n_bio=n_bio)
    assert len(nodes) == 6
    total = float(np.sum(pd["Bbase"][1 : n_bio + 1])) / 0.5 * shape
    for y, r in nodes:
        assert not r["NutFloor"]
        assert r["NutFree"] + y[:n_bio].sum() == pytest.approx(total, rel=1e-12)
    floors = {}
    for s in (1.0, 0.4):
        out = rsim_run(_nutrient_scen(s), method="lsoda", years=range(1, 2))
        floors[s] = out.solver_info["months"][0]["nutrient_floor"]
    assert floors[1.0] == 0
    assert floors[0.4] > 0


def test_the_nutrient_rates_are_off_without_nutrient_limitation():
    rhs, pd, fd, y0 = _build(_scen("rpath"))
    assert pd.get("NutBaseFreeProp") is None
    r = rhs.rates(y0)
    assert np.isnan(r["NutFree"]) and not r["NutFloor"]


# --- deferred minors ---------------------------------------------------------


def _one_living_two_pools(det_to_self_only):
    """Group 1 living, pools 2 and 3. Pool 2 links to itself (and, unless
    ``det_to_self_only``, to pool 3 too)."""
    n = 4  # Outside, 1 living, 2 pools
    det_from = [0, 1, 2]
    det_to = [0, 2, 2]
    det_frac = [0.0, 1.0, 1.0]
    if not det_to_self_only:
        det_from.append(2)
        det_to.append(3)
        det_frac.append(0.5)
    qq = np.zeros((n, n))
    qq[2, 1] = 0.4  # the living group eats pool 2
    bb = np.array([0.0, 2.0, 5.0, 3.0])
    return dict(
        deriv=np.zeros(n),
        QQ=qq,
        BB=bb,
        M0_arr=np.array([0.0, 0.3, 0.0, 0.0]),
        Unassim=np.array([0.0, 0.2, 0.0, 0.0]),
        det_from=np.array(det_from),
        det_to=np.array(det_to),
        det_frac=np.array(det_frac),
        fish_from=np.array([0]),
        fish_to=np.array([0]),
        link_fish_rate=np.array([0.0]),
        NUM_LIVING=1,
        NUM_DEAD=2,
        det_passed_prop=np.array([0.0, 0.0, 0.2, 0.0]),
        det_ba=np.zeros(n),
        det_out=np.full(n, np.nan),
        df_once=True,
    )


def test_df_once_without_other_pools_passes_nothing():
    kw = _one_living_two_pools(det_to_self_only=True)
    flows: dict = {}
    gain, loss = _rpath_detritus_derivs(**kw, flows=flows)
    assert np.all(np.isfinite(gain)) and np.all(np.isfinite(loss))
    assert gain[3] == 0.0  # pool 2 has no other pool: nothing passed
    # the export comes from the parameters, (DetritusOut - DetPassedProp) x B,
    # so the DetPassedProp x B passed nowhere is not hidden in it
    assert flows["DetExport"][2] == pytest.approx(
        (kw["det_out"][2] - 0.2) * kw["BB"][2]
    )
    kw = _one_living_two_pools(det_to_self_only=False)
    gain, loss = _rpath_detritus_derivs(**kw, flows=flows)
    assert gain[3] == pytest.approx(0.2 * 5.0)  # the passed share, once


def test_df_once_is_inert_without_detpassedprop():
    state, pd, fd, fi = run_dicts(scenario(reco_plain(), rule="rpath"))
    assert pd.get("DetPassedProp") is None
    plain = deriv_vector(state.copy(), dict(pd), fd, fi)
    once = deriv_vector(state.copy(), dict(pd, DETRITUS_DF_ONCE=True), fd, fi)
    assert np.array_equal(plain, once)


def test_flows_leave_the_detritus_derivative_unchanged():
    kw = _one_living_two_pools(det_to_self_only=False)
    a = dict(kw, deriv=np.zeros(4), det_out=np.full(4, np.nan))
    b = dict(kw, deriv=np.zeros(4), det_out=np.full(4, np.nan))
    ga, la = _rpath_detritus_derivs(**a)
    gb, lb = _rpath_detritus_derivs(**b, flows={})
    assert np.array_equal(ga, gb) and np.array_equal(la, lb)
    assert np.array_equal(a["deriv"], b["deriv"])


class _Dip:
    """dy = 0 with a dense output below 0 at the nodes; records what the
    rates see."""

    def __init__(self):
        self.stats = {}
        self.seen = []

    def __call__(self, t, y):
        return np.zeros_like(y)

    def rates(self, y):
        self.seen.append(np.array(y, copy=True))
        full = np.concatenate([[0.0], y])
        return {"FoodGain": full.copy(), "LossPropToB": 0.5 * np.ones_like(full)}


def test_quadrature_nodes_are_clipped_before_the_rates(monkeypatch):
    import pypath.core.ecosim_continuous as mod

    y0 = np.array([1.0, 2.0])

    class _Sol:
        status, message, nfev = 0, "stub", 1
        t = np.array([0.0, 1.0])
        y = np.column_stack([y0, y0])

        @staticmethod
        def sol(t):
            return np.array([-1e-12, 2.0])

    monkeypatch.setattr(mod, "solve_ivp", lambda *a, **k: _Sol())
    rhs = _Dip()
    res = integrate_month(rhs, y0, 1.0, "lsoda", 1e-8, np.full(2, 1e-10))
    assert len(rhs.seen) == 6
    assert all(y[0] == 0.0 for y in rhs.seen)
    assert res.loss_rate_mean[1] == 0.0  # B = 0 at every node: no division
    assert np.isnan(res.exchange_mean)  # the double has no exchange
    assert res.pool_net_mean is None  # nor detritus rates


def test_the_mean_state_falls_back_with_the_rates(caplog):
    y0 = np.array([2.0, 5.0])

    class _NanAway(_Dip):
        def __call__(self, t, y):
            return -0.7 * y

        def rates(self, y):
            out = super().rates(y)
            if y[0] != y0[0]:
                out["FoodGain"][1] = np.nan
            return out

    res = integrate_month(
        _NanAway(), y0, 1.0, "dop853", 1e-8, np.full(2, 1e-10), on_failure="warn"
    )
    assert res.degraded is True
    np.testing.assert_array_equal(res.y_mean, y0)
