"""Finite-difference Jacobians reused across monthly restarts (continuous path)."""

import copy
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import pypath.core.ecosim_continuous as cont
from pypath.core.ecosim import rsim_run
from pypath.core.ecosim_continuous import (
    ContinuousRHS,
    JacobianReuse,
    default_atol,
    integrate_month,
)
from tests.continuous_support import (
    needs_reco,
    reco_plain,
    reco_stanzas,
    run_dicts,
    scenario,
)

N = 3
PHYTOPLANKTON = 20  # REco's (240 /yr, flagged NoIntegrate)


class _Linear:
    """``f(t, y) = A y`` on three states, counting its calls."""

    A = np.array([[-1.0, 0.5, 0.0], [0.2, -2.0, 0.1], [0.0, 0.3, -0.5]])

    def __init__(self):
        self.calls = 0

    def __call__(self, t, y):
        self.calls += 1
        return self.A @ np.asarray(y, dtype=float)


class _Decay:
    """dy = -k y on states 1..n; rates are full length, index 0 = Outside."""

    stats: dict = {}

    def __call__(self, t, y):
        return -0.7 * y

    def rates(self, y):
        full = np.concatenate([[0.0], y])
        return {"FoodGain": full.copy(), "LossPropToB": 0.7 * full}


def _n_bio(scen):
    return int(scen.params.NUM_LIVING) + int(scen.params.NUM_DEAD)


def _cmco2_rhs(hden_tol):
    """REco at CmCo 2, month 1: ``(rhs, y0)`` with ``y0`` the base biomass."""
    scen = scenario(reco_plain(), effort=1.0)
    scen.params.CmCo[1 : scen.params.NUM_LIVING + 1] = 2.0
    state, pd, fd, fi = run_dicts(scen)
    n_bio = pd["NUM_LIVING"] + pd["NUM_DEAD"]
    b0 = np.array(pd["Bbase"], dtype=float)
    rhs = ContinuousRHS(pd, fi, n_bio, b0, hden_tol=hden_tol)
    n1 = pd["NUM_GROUPS"] + 1
    rhs.set_month(fd, state, np.zeros(n1, dtype=bool), np.zeros(n1))
    return rhs, b0[1 : n_bio + 1].copy()


def _alternating(y0):
    return y0 * np.where(np.arange(len(y0)) % 2 == 0, 0.2, 3.0)


@needs_reco
def test_the_fd_jacobian_matches_central_differences():
    """Forward differences at ``h_k = 1.49e-8 max(|y_k|, atol_k / rtol)``
    against central differences, column by column (relative column norm),
    on the columns whose norm exceeds 1e-8. Handling time is solved to 1e-14.

    Measured: 6.1e-5 (laguna, 2026-11-15), roundoff noise in the derivative
    (about 1e-12 relative per evaluation, the same at ``hden_tol`` 1e-12),
    which falls as the step grows (1.6e-6 at 1e-7). The bound 1e-3 leaves
    room for other hardware and still catches a transposed, mis-scaled or
    sign-flipped column (an error of order 1)."""
    rhs, y0 = _cmco2_rhs(hden_tol=1e-14)
    y = _alternating(y0)
    rtol = 1e-6
    atol = default_atol(y0, rtol)
    jr = JacobianReuse("bdf", atol, rtol)
    jac = jr.for_month(rhs)
    before = rhs.evals
    J = jac(0.0, y)
    n = len(y)
    assert rhs.evals - before == n + 1  # f at y and one per column
    assert jr.stats == {"jac_fresh": 1, "jac_reused": 0}
    ref = np.empty((n, n))
    for k in range(n):
        h = 6e-6 * max(abs(y[k]), atol[k] / rtol)
        yp, ym = y.copy(), y.copy()
        yp[k] += h
        ym[k] -= h
        ref[:, k] = (rhs(0.0, yp) - rhs(0.0, ym)) / (yp[k] - ym[k])
    norms = np.linalg.norm(ref, axis=0)
    cols = norms > 1e-8
    assert cols.sum() >= n // 2
    err = np.linalg.norm(J - ref, axis=0)[cols] / norms[cols]
    print(f"\nFD vs central: max relative column error {err.max():.2e}")
    assert err.max() <= 1e-3


def test_reuse_rules(monkeypatch):
    y = np.array([1.0, 2.0, 3.0])
    atol = np.full(N, 1e-8)
    fresh = N + 1  # evaluations of a fresh Jacobian: f at y and N columns

    # BDF and Radau: a month's first request reuses, later requests are fresh
    for method in ("bdf", "radau"):
        fun = _Linear()
        jr = JacobianReuse(method, atol, 1e-6)
        J1 = jr.for_month(fun)(0.0, y)  # no Jacobian yet: fresh
        assert fun.calls == fresh
        np.testing.assert_allclose(J1, _Linear.A, rtol=1e-6, atol=1e-8)
        assert jr.stats == {"jac_fresh": 1, "jac_reused": 0}
        jac = jr.for_month(fun)  # month 2
        assert jr.stats == {"jac_fresh": 0, "jac_reused": 0}
        J2 = jac(0.0, 1.5 * y)  # the first request reuses, however far y moved
        assert fun.calls == fresh
        np.testing.assert_array_equal(J2, J1)
        J2[:] = 0.0  # the caller's copy: the held Jacobian is unchanged
        jac(0.01, 1.5 * y)  # a later request computes fresh
        assert fun.calls == 2 * fresh
        assert jr.stats == {"jac_fresh": 1, "jac_reused": 1}
        J3 = jr.for_month(fun)(0.0, y)
        np.testing.assert_allclose(J3, _Linear.A, rtol=1e-6, atol=1e-8)

    # LSODA: reuses only when t advances and the state is within 10 % of the
    # Jacobian's; fresh at the same t or an earlier one (a retry: ODEPACK cuts
    # the step after a corrector failure) and after a 20 % move
    fun = _Linear()
    jr = JacobianReuse("lsoda", atol, 1e-6)
    jac = jr.for_month(fun)
    jac(0.0, y)
    assert fun.calls == fresh
    jac(0.1, 1.05 * y)
    assert fun.calls == fresh
    assert jr.stats == {"jac_fresh": 1, "jac_reused": 1}
    jac(0.1, 1.05 * y)  # the same t
    assert fun.calls == 2 * fresh
    jac(0.05, 1.05 * y)  # an earlier t, the state within 10 %
    assert fun.calls == 3 * fresh
    jac(0.08, 1.06 * y)  # t advances again, within 10 %: reused
    assert fun.calls == 3 * fresh
    jac(0.2, 1.05 * 1.2 * y)  # 20 % from the state of the last fresh one
    assert fun.calls == 4 * fresh
    assert jr.stats == {"jac_fresh": 4, "jac_reused": 2}
    jac = jr.for_month(fun)
    jac(0.0, 3.0 * y)  # the month's first request
    assert fun.calls == 4 * fresh
    assert jr.stats == {"jac_fresh": 0, "jac_reused": 1}

    # a held Jacobian of another shape is not reused
    four = JacobianReuse("bdf", 1e-8, 1e-6)
    four.for_month(lambda t, v: -np.asarray(v))(0.0, np.ones(4))
    fun = _Linear()
    four.for_month(fun)(0.0, y)
    assert fun.calls == fresh and four.stats["jac_fresh"] == 1

    # integrate_month passes jac to solve_ivp, except for DOP853 and None
    calls: list = []
    real = cont.solve_ivp

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cont, "solve_ivp", spy)
    y0 = np.array([2.0, 5.0])
    for method, with_jac in (("dop853", True), ("lsoda", False), ("bdf", True)):
        rhs = _Decay()
        jac = JacobianReuse(method, 1e-10, 1e-8).for_month(rhs) if with_jac else None
        integrate_month(rhs, y0, 1.0, method, 1e-8, np.full(2, 1e-10), jac=jac)
    assert "jac" not in calls[0] and "jac" not in calls[1]
    assert callable(calls[2]["jac"])

    with pytest.raises(ValueError, match="method"):
        JacobianReuse("rk4", atol, 1e-6)


def test_a_non_finite_jacobian_is_not_held():
    """A fresh Jacobian with a non-finite entry goes to the solver but is not
    held: the next month's first request forms a fresh one instead of
    reusing it (the matrix held before it is dropped too)."""
    y = np.array([1.0, 2.0, 3.0])
    for method in ("bdf", "lsoda"):
        fun = _Linear()
        jr = JacobianReuse(method, np.full(N, 1e-8), 1e-6)
        jr.for_month(fun)(0.0, y)  # month 1: a finite one, held

        def broken(t, v):
            # NaN once state 1 moves: column 1 of a fresh Jacobian
            return fun(t, v) if v[1] == y[1] else np.full(N, np.nan)

        jac = jr.for_month(broken)  # month 2
        assert np.all(np.isfinite(jac(0.0, y)))  # the first request reuses
        bad = jac(0.0, y)  # a later one (under LSODA a retry): fresh
        assert jr.stats == {"jac_fresh": 1, "jac_reused": 1}
        assert np.isnan(bad[:, 1]).all()
        assert np.all(np.isfinite(np.delete(bad, 1, axis=1)))
        good = jr.for_month(fun)(0.0, y)  # month 3: not the NaN matrix
        assert jr.stats == {"jac_fresh": 1, "jac_reused": 0}
        np.testing.assert_allclose(good, _Linear.A, rtol=1e-6, atol=1e-8)


class _Robertson:
    """Robertson's stiff kinetics, ``k1`` and ``k3`` changed between months
    (a jump the reused Jacobian has to survive), counting its calls."""

    def __init__(self, k1, k3):
        self.k1, self.k3, self.calls = k1, k3, 0

    def __call__(self, t, y):
        self.calls += 1
        a = self.k1 * y[0]
        b = self.k3 * y[1] * y[2]
        c = 3e7 * y[1] ** 2
        return np.array([-a + b, a - b - c, c])


#: Six "months" of 40/12 time units, ``(k1, k3)`` jumping at every restart.
_ROBERTSON_MONTHS = [
    (0.04, 1e4),
    (4.0, 1e5),
    (0.04, 1e3),
    (4.0, 1e5),
    (0.04, 1e3),
    (4.0, 1e4),
]


def test_lsoda_reuse_through_solve_ivp():
    """``JacobianReuse`` on SciPy's real LSODA (the default method), six
    monthly restarts of a stiff system with a jump in its rates at each:
    fresh and reused Jacobians both occur; every request at or before the
    previous ``t`` (a retry) is fresh, and every reuse is a month's first
    request or one at an advancing ``t``; the end state agrees with LSODA
    without a callable (its own Jacobians).

    Measured (SciPy 1.15.3 and 1.18.1 alike): 8 retries, all fresh; 38
    fresh, 47 reused; end states 1.5e-6 apart relative (atol 1e-10 added to
    the denominator), both rtol-1e-6 solutions; bounded at 1e-4."""
    rtol, atol, dt = 1e-6, 1e-10, 40.0 / 12.0
    jr = JacobianReuse("lsoda", np.full(3, atol), rtol)
    y = np.array([1.0, 0.0, 0.0])
    y_ref = y.copy()
    events: list = []
    fresh = reused = 0
    for k1, k3 in _ROBERTSON_MONTHS:
        fun = _Robertson(k1, k3)
        jac = jr.for_month(fun)
        last: list = []

        def spy(t, v, jac=jac, last=last):
            before = jr.stats["jac_fresh"]
            out = jac(t, v)
            kind = "fresh" if jr.stats["jac_fresh"] > before else "reused"
            if not last:
                where = "first"
            else:
                where = "advance" if t > last[0] else "retry"
            events.append((where, kind))
            last[:] = [t]
            return out

        sol = solve_ivp(
            fun, (0.0, dt), y, method="LSODA", rtol=rtol, atol=atol, jac=spy
        )
        assert sol.status == 0, sol.message
        y = sol.y[:, -1]
        fresh += jr.stats["jac_fresh"]
        reused += jr.stats["jac_reused"]
        ref = solve_ivp(
            _Robertson(k1, k3), (0.0, dt), y_ref, method="LSODA", rtol=rtol, atol=atol
        )
        y_ref = ref.y[:, -1]
    count = {e: events.count(e) for e in set(events)}
    end = float(np.max(np.abs(y - y_ref) / (np.abs(y_ref) + atol)))
    print(
        f"\nLSODA Jacobian requests {count}; fresh {fresh}, reused {reused}; end {end:.2e}"
    )
    assert fresh > 0 and reused > 0
    assert count.get(("retry", "fresh"), 0) > 0  # the retreat rule is exercised
    assert count.get(("retry", "reused"), 0) == 0  # ... and never reuses
    assert count.get(("advance", "reused"), 0) > 0  # the advance rule reuses
    assert count.get(("first", "reused"), 0) == len(_ROBERTSON_MONTHS) - 1
    assert end <= 1e-4


@needs_reco
def test_the_difference_step_floor(monkeypatch):
    """The finite-difference step is ``1.49e-8 x max(|y_k|, floor_k)``. On a
    run the biomass floor is ``atol / rtol`` capped at the default atol's
    ratio, ``0.01 x max(B0_k, 1e-6 x max B0)``: a large user atol (1e-3,
    ratio 1000) does not make the step large, a small one (1e-12) lowers the
    floor, and the default atol gives its own ratio exactly. Log states take
    a floor of 1 in ``z``; quasi-steady groups leave the slow states'."""
    seen: list = []
    real = JacobianReuse._fresh

    def spy(self, fun, t, y):
        seen.append(np.array(self._month_floor, dtype=float))
        return real(self, fun, t, y)

    monkeypatch.setattr(JacobianReuse, "_fresh", spy)
    scen = scenario(reco_plain(), rule="ewe", effort=2.0)
    n = _n_bio(scen)
    rtol = 1e-6
    cap = default_atol(scen.params.B_BaseRef[1 : n + 1], rtol) / rtol
    for opts, want in (
        ({}, cap),
        ({"atol": 1e-3}, cap),
        ({"atol": 1e-12}, np.full(n, 1e-12 / rtol)),
        ({"positivity": "log"}, np.ones(n)),
        (
            {"fast_groups": "quasi-steady", "qs_rate": 200.0},
            np.delete(cap, PHYTOPLANKTON - 1),
        ),
    ):
        seen.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rsim_run(
                copy.deepcopy(scen),
                method="bdf",
                years=range(1, 2),
                solver_options={"rtol": rtol, **opts},
            )
        assert out.solver_info["qs_groups"] == (
            [PHYTOPLANKTON] if "fast_groups" in opts else []
        )
        assert seen, opts
        for floor in seen:
            np.testing.assert_array_equal(floor, want, err_msg=str(opts))


def _effort_jump():
    """REco with stanzas, EwE rule, effort 2, x 5 from month 7, two years."""
    scen = scenario(reco_stanzas(), rule="ewe", effort=2.0, years=range(1, 3))
    scen.fishing.ForcedEffort[6:] = scen.fishing.ForcedEffort[6:] * 5.0
    return scen


@needs_reco
@pytest.mark.parametrize("method", ["bdf", "radau", "lsoda"])
def test_reuse_agrees_with_the_solver(method):
    """Reuse across the effort jump matches the solver's own Jacobians to 1e-5
    with no more evaluations. Spike: BDF 848 against 1377 evaluations
    (9.5e-10), Radau 920 against 1399 (1.6e-7), LSODA identical (REco forms
    no LSODA Jacobian)."""
    base = _effort_jump()
    outs = {}
    for mode in ("solver", "reuse"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outs[mode] = rsim_run(
                copy.deepcopy(base),
                method=method,
                years=range(1, 3),
                solver_options={"jacobian": mode},
            )
    n = _n_bio(base)
    reuse, solver = outs["reuse"], outs["solver"]
    diff = float(
        np.max(
            np.abs(
                reuse.out_Biomass[1:, 1 : n + 1] / solver.out_Biomass[1:, 1 : n + 1] - 1
            )
        )
    )
    ev = {mode: out.solver_info["rhs_evals"] for mode, out in outs.items()}
    months = reuse.solver_info["months"]
    fresh = sum(m["jac_fresh"] for m in months)
    reused = sum(m["jac_reused"] for m in months)
    print(
        f"\n{method}: rhs_evals solver {ev['solver']} reuse {ev['reuse']}; "
        f"jac_fresh {fresh}, jac_reused {reused}; max rel diff {diff:.2e}"
    )
    assert diff <= 1e-5
    assert ev["reuse"] <= ev["solver"]
    for out in outs.values():
        assert sum(m["clips"] for m in out.solver_info["months"]) == 0
    assert reuse.solver_info["jacobian"] == "reuse"
    assert solver.solver_info["jacobian"] == "solver"
    assert all(
        m["jac_fresh"] == m["jac_reused"] == 0 for m in solver.solver_info["months"]
    )
    if method == "lsoda":
        # REco keeps LSODA non-stiff (Adams throughout here): when it never
        # asks for a Jacobian the two runs are identical. A platform where
        # LSODA switches to stiff stepping once is held to the bounds above
        # only. LSODA's reuse is tested on a stiff system
        # (test_lsoda_reuse_through_solve_ivp) and on LT2022.
        if fresh == reused == 0:
            assert ev["reuse"] == ev["solver"]
            np.testing.assert_array_equal(reuse.out_Biomass, solver.out_Biomass)
    else:
        # every month after the first reuses at its first request
        assert all(m["jac_reused"] >= 1 for m in months[1:])


@needs_reco
def test_jacobian_columns_are_counted_evaluations(monkeypatch):
    """Each Jacobian column is a counted right-hand-side evaluation, so the
    deriv-call identity holds with Jacobians formed (BDF), and ``rhs_evals``
    covers SciPy's ``nfev`` plus ``n_state + 1`` per fresh Jacobian."""
    calls = {"n": 0}
    real = cont.deriv_vector

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cont, "deriv_vector", counting)
    scen = scenario(reco_plain(), rule="ewe", effort=2.0)
    scen.params.CmCo[1 : scen.params.NUM_LIVING + 1] = 2.0
    out = rsim_run(scen, method="bdf", years=range(1, 2))
    info = out.solver_info
    months = info["months"]
    fresh = sum(m["jac_fresh"] for m in months)
    assert fresh > 0
    # +1: the DetritusOut pre-fill; + one per month: the link-loop check
    assert calls["n"] == info["rhs_evals"] + 1 + len(months)
    n_state = _n_bio(scen) + len(info["ftime_groups"])
    assert info["rhs_evals"] >= sum(m["nfev"] for m in months) + fresh * (n_state + 1)


@needs_reco
def test_jacobian_option():
    scen = scenario(reco_plain())
    with pytest.raises(ValueError, match="jacobian"):
        rsim_run(
            copy.deepcopy(scen),
            method="lsoda",
            years=range(1, 2),
            solver_options={"jacobian": "x"},
        )
    with pytest.raises(ValueError, match="jacobian"):
        cont.continuous_options("bdf", {"jacobian": None}, _n_bio(scen))
    assert cont.continuous_options("bdf", None, _n_bio(scen))["jacobian"] == "reuse"
    for opts, want in (({}, "reuse"), ({"jacobian": "solver"}, "solver")):
        out = rsim_run(
            copy.deepcopy(scen), method="bdf", years=range(1, 2), solver_options=opts
        )
        assert out.solver_info["jacobian"] == want
        months = out.solver_info["months"]
        assert all({"jac_fresh", "jac_reused"} <= set(m) for m in months)
        if want == "solver":
            assert all(m["jac_fresh"] == m["jac_reused"] == 0 for m in months)
        else:
            assert sum(m["jac_fresh"] + m["jac_reused"] for m in months) > 0


@pytest.mark.integration
@pytest.mark.slow
def test_lt2022_lsoda_reuse_agrees_with_the_solver():
    """LT2022 makes LSODA stiff, so its reuse rule is exercised: two years of
    LSODA with ``"reuse"`` against ``"solver"`` agree to 1e-5 with no more
    evaluations and no clips."""
    from tests.test_ecosim_continuous_validation import LT2022, _load_ewe

    base = _load_ewe(LT2022, scenario=16, timeseries_dataset=35)
    outs = {}
    for mode in ("solver", "reuse"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outs[mode] = rsim_run(
                copy.deepcopy(base),
                method="lsoda",
                years=range(1, 3),
                solver_options={"jacobian": mode},
            )
    n = _n_bio(base)
    reuse, solver = outs["reuse"], outs["solver"]
    diff = float(
        np.max(
            np.abs(
                reuse.out_Biomass[1:, 1 : n + 1] / solver.out_Biomass[1:, 1 : n + 1] - 1
            )
        )
    )
    ev = {mode: out.solver_info["rhs_evals"] for mode, out in outs.items()}
    months = reuse.solver_info["months"]
    fresh = sum(m["jac_fresh"] for m in months)
    reused = sum(m["jac_reused"] for m in months)
    print(
        f"\nLT2022 lsoda, two years: rhs_evals solver {ev['solver']} reuse "
        f"{ev['reuse']}; jac_fresh {fresh}, jac_reused {reused}; max rel diff "
        f"{diff:.2e}"
    )
    assert fresh + reused > 0
    assert diff <= 1e-5
    assert ev["reuse"] <= ev["solver"]
    for out in outs.values():
        assert sum(m["clips"] for m in out.solver_info["months"]) == 0


@pytest.mark.integration
@pytest.mark.slow
def test_lt2022_bdf_evaluations():
    """Ten years of BDF on LT2022 with reuse: at most 100 right-hand-side
    evaluations a month (spike: 86.4), no clips, and few fresh Jacobians
    (spike: 0.23 a month on average, at most 5)."""
    from tests.test_ecosim_continuous_validation import LT2022, _load_ewe

    scen = _load_ewe(LT2022, scenario=16, timeseries_dataset=35)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rsim_run(scen, method="bdf", years=range(1, 11))
    info = out.solver_info
    months = info["months"]
    assert len(months) == 120 and all(m["status"] == 0 for m in months)
    fresh = np.array([m["jac_fresh"] for m in months])
    reused = np.array([m["jac_reused"] for m in months])
    per_month = info["rhs_evals"] / 120
    print(
        f"\nLT2022 BDF reuse: rhs_evals {info['rhs_evals']} ({per_month:.1f} per "
        f"month); jac_fresh mean {fresh.mean():.2f}, max {fresh.max()}; "
        f"jac_reused mean {reused.mean():.2f}; clips "
        f"{sum(m['clips'] for m in months)}"
    )
    assert per_month <= 100
    assert sum(m["clips"] for m in months) == 0
    assert fresh.mean() <= 0.5 and fresh.max() <= 8
