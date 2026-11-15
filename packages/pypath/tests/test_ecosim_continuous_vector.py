"""The vectorised consumption pass of the continuous path (phase 3)."""

import numpy as np
import pytest

import pypath.core.ecosim_deriv as deriv_mod
from pypath.core.ecosim import rsim_run
from pypath.core.ecosim_deriv import (
    HdenKernel,
    _compute_consumption_sparse_python,
    _compute_consumption_vector,
    deriv_vector,
)
from tests.continuous_support import needs_reco, reco_plain, scenario
from tests.test_ecosim_continuous_hden import (
    _kernel_call,
    _mediated,
    _pattern,
    _perturbed,
    _setup,
)

pytestmark = needs_reco

# HandleSwitch exponents other than 1 go through numpy's array power, which
# can differ from the loop's scalar power in the last bit (SIMD pow)
VECTOR_CASES = {
    "reco": True,  # REco's own arrays: pooling arrays at 0, HandleSwitch 1
    "zero_prey": True,
    "pooling": True,
    "handle_self_only": True,
    "handle_switch": False,
    "eat_eff": True,
    "mediation": True,
    "combined": False,
}

SENTINEL = -7.0


def _case_params(case):
    params, forcing, fishing = _setup(reco_plain())
    y = _perturbed(params)
    active = np.asarray(params["ActiveLink"]) != 0
    if case in ("pooling", "combined"):
        params["HandleSelf"] = np.where(active, 0.5, 0.0)
        params["ScrambleSelf"] = np.where(active, 0.5, 0.0)
    if case == "handle_self_only":
        params["HandleSelf"] = np.where(active, 0.5, 0.0)
        params["ScrambleSelf"] = np.zeros_like(params["HandleSelf"])
    if case in ("handle_switch", "combined"):
        params["HandleSwitch"] = np.where(active, 2.0, 0.0)
    if case in ("eat_eff", "combined"):
        params["_eat_eff"] = np.full(len(y), 0.7)
    if case in ("mediation", "combined"):
        params["_mediation"] = _mediated(params)
    if case == "zero_prey":
        y[4] = 0.0
    return y, params, forcing, fishing


def _both_kernels(inputs, pred_yy):
    """QQ of the link loop and of its vectorised twin, from a sentinel fill."""
    out = []
    for kernel in (_compute_consumption_sparse_python, _compute_consumption_vector):
        n = len(inputs.BB)
        qq = np.full((n, n), SENTINEL)
        kernel(
            qq,
            inputs.BB,
            inputs.VV,
            inputs.DD,
            inputs.QQbase,
            inputs.preyYY,
            pred_yy,
            inputs.link_prey,
            inputs.link_pred,
            len(inputs.link_prey),
            ScrambleSelf=inputs.ScrambleSelf,
            HandleSelf=inputs.HandleSelf,
            PredPredWeight=inputs.PredPredWeight,
            PreyPreyWeight=inputs.PreyPreyWeight,
            HandleSwitch=inputs.HandleSwitch,
            COUPLED=inputs.COUPLED,
            med_multipliers=inputs.med_multipliers,
            eat_eff=inputs.eat_eff,
        )
        out.append(qq)
    return out


@pytest.mark.parametrize("case", list(VECTOR_CASES))
def test_the_vector_pass_equals_the_link_loop(case):
    y, params, forcing, fishing = _case_params(case)
    seen, rates = _kernel_call(
        y, params, forcing, fishing, lambda k: k.hden0 * _pattern(len(k.hden0))
    )
    inputs = seen["inputs"]
    lp, lq = inputs.link_prey, inputs.link_pred
    # the case reaches the kernel
    if case == "reco":
        assert np.all(inputs.ScrambleSelf == 0.0) and np.all(inputs.HandleSelf == 0)
        switch = inputs.HandleSwitch[lp, lq]
        assert set(np.unique(switch)) == {0.0, 1.0}
        assert np.count_nonzero(switch == 1.0) >= 100
    elif case == "zero_prey":
        assert np.any(inputs.BB[lp] == 0.0)
    elif case in ("pooling", "combined"):
        assert np.all(inputs.ScrambleSelf[lp, lq] == 0.5)
        assert HdenKernel(inputs).coupled()
    elif case == "handle_self_only":
        assert np.all(inputs.HandleSelf[lp, lq] == 0.5)
        assert not HdenKernel(inputs).coupled()
    if case in ("handle_switch", "combined"):
        assert np.all(inputs.HandleSwitch[lp, lq] == 2.0)
    if case in ("eat_eff", "combined"):
        assert np.any(inputs.eat_eff == 0.7)
    if case in ("mediation", "combined"):
        assert np.sum(inputs.med_multipliers[lp, lq] != 1.0) == 3

    pred_yy = inputs.predYY * (inputs.hden0 / seen["h"])
    qq_loop, qq_vec = _both_kernels(inputs, pred_yy)
    written = qq_loop != SENTINEL
    assert np.count_nonzero(qq_loop[written]) >= 100
    # the twin writes exactly the links the loop writes
    np.testing.assert_array_equal(qq_vec != SENTINEL, written)
    if case == "zero_prey":
        assert np.any(~written[lp, lq])  # the loop skips the zero prey's links
    if VECTOR_CASES[case]:
        np.testing.assert_array_equal(qq_vec, qq_loop)
    else:
        np.testing.assert_allclose(qq_vec, qq_loop, rtol=1e-13, atol=0)

    # through deriv_vector: the flag selects the twin for the final pass
    def solved(k):
        return k.hden0 * _pattern(len(k.hden0))

    p = dict(params, _hden_exact=True, _hden_solver=solved)
    p["_hden"] = np.array(params["_hden0"], dtype=float, copy=True)
    vec_rates = {}
    d_loop = deriv_vector(y.copy(), p, forcing, fishing, rates={})
    d_vec = deriv_vector(
        y.copy(), dict(p, _KERNEL_VECTOR=True), forcing, fishing, rates=vec_rates
    )
    if VECTOR_CASES[case]:
        np.testing.assert_array_equal(d_vec, d_loop)
        np.testing.assert_array_equal(
            vec_rates["HdenConsumption"], rates["HdenConsumption"]
        )
    else:
        np.testing.assert_allclose(d_vec, d_loop, rtol=1e-12, atol=1e-15)


def _spy(monkeypatch):
    calls = {"n": 0}
    real = deriv_mod._compute_consumption_vector

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(deriv_mod, "_compute_consumption_vector", spy)
    return calls


def test_the_parity_paths_never_take_the_vector_pass(monkeypatch):
    calls = _spy(monkeypatch)
    for method in ("RK4", "AB"):
        rsim_run(scenario(reco_plain(), rule="ewe"), method=method, years=range(1, 2))
    assert calls["n"] == 0
    out = rsim_run(
        scenario(reco_plain(), rule="ewe"), method="lsoda", years=range(1, 2)
    )
    # every evaluation's final pass; the monthly check runs the loop
    assert calls["n"] == out.solver_info["rhs_evals"] > 0


@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_kernel_check_is_reported(rule):
    """Under the EwE rule the check compares HdenConsumption, under Rpath's
    (no handling time) FoodGain."""
    scen = scenario(reco_plain(), rule=rule)
    scen.params.CmCo[1 : scen.params.NUM_LIVING + 1] = 2.0
    out = rsim_run(scen, method="lsoda", years=range(1, 2))
    months = out.solver_info["months"]
    assert len(months) == 12
    checks = [m["kernel_check"] for m in months]
    print(rule, "kernel_check", checks)
    for m, check in zip(months, checks):
        assert np.isfinite(check) and 0.0 <= check <= 1e-13, (m["month"], check)
