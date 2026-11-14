"""First-step reuse, tolerances and the timing budget (continuous path)."""

import copy
import os
import time

import numpy as np
import pytest

import pypath.core.ecosim_continuous as cont
from pypath.core.ecosim import rsim_run
from pypath.core.ecosim_continuous import (
    ContinuousRHS,
    default_atol,
    integrate_month,
)
from tests.continuous_support import needs_reco, reco_plain, scenario

pytestmark = needs_reco

# Each evaluation is one derivative pass plus the Hden solve (6-9 kernel
# passes), so 1.5x cannot hold; 2.5x still catches a second derivative call
# per evaluation or a fallback to hybr.
EVAL_COST_BOUND = 2.5
RK4_RATIO_BOUND = 12.0


class Decay:
    """dy = -k y on states 1..n; rates are full length, index 0 = Outside."""

    stats: dict = {}

    def __call__(self, t, y):
        return -0.7 * y

    def rates(self, y):
        full = np.concatenate([[0.0], y])
        return {"FoodGain": full.copy(), "LossPropToB": 0.7 * full}


def _spy_solve_ivp(monkeypatch, fail_first=False):
    """Record the keyword arguments of every ``solve_ivp`` call; with
    ``fail_first`` the first call's result is reported as failed."""
    calls: list = []
    real = cont.solve_ivp

    def spy(*args, **kwargs):
        calls.append(kwargs)
        res = real(*args, **kwargs)
        if fail_first and len(calls) == 1:
            res.status = -1
            res.message = "forced failure"
        return res

    monkeypatch.setattr(cont, "solve_ivp", spy)
    return calls


def test_the_next_month_starts_from_the_last_step(monkeypatch):
    calls = _spy_solve_ivp(monkeypatch)
    out = rsim_run(scenario(reco_plain()), method="lsoda", years=range(1, 2))
    months = out.solver_info["months"]
    assert len(calls) == len(months) >= 3
    assert "first_step" not in calls[0] or calls[0]["first_step"] is None
    assert months[0]["first_step"] is None
    for m in (1, 2):
        assert months[m - 1]["last_step"] > 0
        assert calls[m]["first_step"] == months[m - 1]["last_step"]
        assert months[m]["first_step"] == months[m - 1]["last_step"]


def test_a_failed_month_leaves_the_next_first_step_to_the_solver(monkeypatch):
    calls = _spy_solve_ivp(monkeypatch, fail_first=True)
    out = rsim_run(
        scenario(reco_plain()),
        method="lsoda",
        years=range(1, 2),
        solver_options={"on_failure": "warn"},
    )
    months = out.solver_info["months"]
    assert months[0]["status"] < 0
    assert calls[1].get("first_step") is None
    assert months[1]["first_step"] is None
    # month 2 was clean, so month 3 reuses its last step
    assert calls[2]["first_step"] == months[1]["last_step"]


def test_first_step_is_passed_through(monkeypatch):
    calls = _spy_solve_ivp(monkeypatch)
    res = integrate_month(
        Decay(),
        np.array([2.0, 5.0]),
        1.0,
        "lsoda",
        1e-8,
        np.full(2, 1e-10),
        first_step=0.25,
    )
    assert calls[0]["first_step"] == 0.25
    assert res.last_step is not None and 0.0 < res.last_step <= 1.0


def test_last_step_is_the_last_full_step():
    res = integrate_month(
        Decay(), np.array([2.0, 5.0]), 1.0, "dop853", 1e-12, np.full(2, 1e-14)
    )
    assert res.steps >= 2
    from scipy.integrate import solve_ivp

    t = solve_ivp(
        Decay(),
        (0.0, 1.0),
        np.array([2.0, 5.0]),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        max_step=1.0,
    ).t
    assert res.last_step == pytest.approx(np.diff(t)[-2])
    # a one-step month reports its only step
    one = integrate_month(
        Decay(),
        np.array([2.0, 5.0]),
        1.0,
        "dop853",
        1e-1,
        np.full(2, 1e1),
        first_step=1.0,
    )
    assert one.steps == 1 and one.last_step == pytest.approx(1.0)


def test_default_atol_follows_rtol():
    b0 = np.array([1e-6, 1.0, 100.0])
    np.testing.assert_allclose(
        default_atol(b0, rtol=1e-10), [1e-16, 1e-12, 1e-10], rtol=1e-12
    )
    np.testing.assert_allclose(default_atol(b0), [1e-12, 1e-8, 1e-6], rtol=1e-12)


def test_the_run_default_atol_follows_rtol():
    scen = scenario(reco_plain())
    out = rsim_run(
        scen, method="lsoda", years=range(1, 2), solver_options={"rtol": 1e-10}
    )
    b0 = np.asarray(scen.params.B_BaseRef, dtype=float)
    n_bio = scen.params.NUM_LIVING + scen.params.NUM_DEAD
    np.testing.assert_allclose(
        out.solver_info["atol"], default_atol(b0[1 : n_bio + 1], rtol=1e-10)
    )


def _cmco(scen, value):
    scen.params.CmCo[1 : scen.params.NUM_LIVING + 1] = value
    return scen


def test_one_derivative_per_rhs_evaluation(monkeypatch):
    calls = {"n": 0}
    real = cont.deriv_vector

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cont, "deriv_vector", counting)
    scen = _cmco(scenario(reco_plain(), rule="ewe", effort=2.0), 2.0)
    out = rsim_run(scen, method="lsoda", years=range(1, 2))
    info = out.solver_info
    # +1: the DetritusOut pre-fill at the base state, outside any evaluation
    assert calls["n"] == info["rhs_evals"] + 1
    assert info["rhs_evals"] > sum(m["nfev"] for m in info["months"])
    assert max(m["hden_passes_max"] for m in info["months"]) <= 10


def _cpu(fn):
    t0 = time.process_time()
    result = fn()
    return time.process_time() - t0, result


def _exact_call_cpu(scen):
    """Min CPU of one single-pass ``deriv_vector`` call at the base state."""
    from pypath.core.ecosim import _month_forcing, _run_setup
    from pypath.core.ecosim_deriv import deriv_vector

    setup = _run_setup(scen, range(1, 2))
    n_bio = scen.params.NUM_LIVING + scen.params.NUM_DEAD
    rhs = ContinuousRHS(
        setup.params_dict, setup.fishing_dict, n_bio, scen.params.B_BaseRef
    )
    pd = rhs._pd
    pd.pop("_hden_solver", None)  # Hden0 as given: one pass, no solve
    forcing = _month_forcing(scen, setup, 1, setup.ftime)
    base = np.array(setup.state, dtype=float)
    samples = []
    for _ in range(200):
        t0 = time.process_time()
        deriv_vector(base.copy(), pd, forcing, setup.fishing_dict, rates={})
        samples.append(time.process_time() - t0)
    return min(samples)


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize("cmco", [1000.0, 2.0])
def test_reco_is_within_its_budget(cmco):
    scen = _cmco(scenario(reco_plain(), rule="ewe", effort=2.0), cmco)
    years = range(1, 2)

    def run(method):
        s = copy.deepcopy(scen)
        cpu, out = _cpu(lambda: rsim_run(s, method=method, years=years))
        return cpu, out

    run("RK4")  # warm-up of each method, untimed
    run("lsoda")
    rk4, lsoda = [], []
    evals = None
    for _ in range(5):
        rk4.append(run("RK4")[0])
        cpu, out = run("lsoda")
        lsoda.append(cpu)
        evals = out.solver_info["rhs_evals"]
    ratio = min(lsoda) / min(rk4)
    exact = _exact_call_cpu(scen)
    per_eval = min(lsoda) / evals
    report = (
        f"CmCo {cmco}: RK4 samples {rk4}, LSODA samples {lsoda}, "
        f"ratio {ratio:.2f}, rhs_evals {evals}, CPU/eval {per_eval:.3e}, "
        f"exact call {exact:.3e} (x{per_eval / exact:.2f}), "
        f"loadavg {os.getloadavg()}"
    )
    print(report)
    assert ratio <= RK4_RATIO_BOUND, report
    assert per_eval <= EVAL_COST_BOUND * exact, report
