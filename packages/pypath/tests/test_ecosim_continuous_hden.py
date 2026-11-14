"""Handling time as a fixed point of the state (continuous path)."""

import numpy as np
import pytest

from pypath.core.ecosim_continuous import (
    hden_index,
    iterate_hden,
    recalibrated_hden0,
    solve_hden,
    solve_hden_kernel,
)
from pypath.core.ecosim_deriv import HdenKernel, KernelInputs, deriv_vector
from pypath.core.mediation import MediationCollection, MediationLink, MediationShape
from tests.continuous_support import needs_reco, run_dicts, scenario
from tests.continuous_support import reco_plain as _reco_plain


@pytest.fixture(scope="module")
def reco_plain():
    return _reco_plain()


def _setup(reco, cmco=2.0):
    scen = scenario(reco, effort=1.0)
    nl = scen.params.NUM_LIVING
    scen.params.CmCo[1 : nl + 1] = cmco
    _, params, forcing, fishing = run_dicts(scen)
    params["_hden0"] = recalibrated_hden0(params, params["_cmco_qb"])
    return params, forcing, fishing


def _perturbed(params):
    y = np.array(params["Bbase"], dtype=float)
    nl = len(y) - 1
    y[1 : nl + 1] *= np.where(np.arange(nl) % 2 == 0, 0.2, 3.0)[: len(y) - 1]
    return y


def _residual(y, params, forcing, fishing, h):
    """Residual of ``h`` under the full derivative (no ``_hden_solver``)."""
    p = dict(params, _hden=h.copy(), _hden_exact=True)
    p.pop("_hden_solver", None)
    r = {}
    deriv_vector(y.copy(), p, forcing, fishing, rates=r)
    idx = hden_index(y, params, forcing)
    cq = params["_cmco_qb"]
    act = (cq > 0) & (idx > 0)
    g = 1.0 + h[act] * r["HdenConsumption"][act] / (cq[act] * idx[act])
    return np.max(np.abs(g - h[act]) / h[act])


@needs_reco
def test_the_recalibrated_hden0_is_the_base_fixed_point(reco_plain):
    params, forcing, fishing = _setup(reco_plain)
    y = np.array(params["Bbase"], dtype=float)
    idx = hden_index(y, params, forcing)
    h, n = solve_hden(y, params, forcing, fishing, idx, params["_cmco_qb"])
    print("base fixed point evaluations:", n)
    np.testing.assert_allclose(h, params["_hden0"], rtol=1e-12, atol=0)

    scen = scenario(reco_plain, effort=1.0)
    nl = scen.params.NUM_LIVING
    scen.params.CmCo[1 : nl + 1] = 1000.0
    _, big, _, _ = run_dicts(scen)
    ewe = np.array(big["_hden0"], dtype=float)  # EwE's CmCo / (CmCo - 1)
    h0 = recalibrated_hden0(big, big["_cmco_qb"])
    sel = (np.asarray(big["QQbase"])[0, :] == 0) & (np.asarray(big["_cmco_qb"]) > 0)
    assert sel.any()
    np.testing.assert_allclose(h0[sel], ewe[sel], rtol=1e-12)


@needs_reco
def test_a_perturbed_state_converges(reco_plain):
    params, forcing, fishing = _setup(reco_plain)
    y = _perturbed(params)
    idx = hden_index(y, params, forcing)
    h, n = solve_hden(y, params, forcing, fishing, idx, params["_cmco_qb"])
    print("perturbed state evaluations:", n)
    assert _residual(y, params, forcing, fishing, h) < 1e-10


@needs_reco
def test_no_fixed_point_raises_a_clear_error(reco_plain):
    params, forcing, fishing = _setup(reco_plain)
    y = _perturbed(params)
    idx = hden_index(y, params, forcing)
    with pytest.raises(RuntimeError, match="handling time"):
        solve_hden(
            y,
            params,
            forcing,
            fishing,
            idx,
            params["_cmco_qb"],
            max_iter=1,
            fallback=False,
        )


@needs_reco
def test_solving_does_not_touch_params(reco_plain):
    params, forcing, fishing = _setup(reco_plain)
    dicts = (params, forcing, fishing)

    def snap(d):
        return {
            k: np.array(v, copy=True) for k, v in d.items() if isinstance(v, np.ndarray)
        }

    before = [snap(d) for d in dicts]
    keys = [set(d) for d in dicts]
    y = _perturbed(params)
    idx = hden_index(y, params, forcing)
    solve_hden(y, params, forcing, fishing, idx, params["_cmco_qb"])
    for d, snapshot, k in zip(dicts, before, keys):
        assert set(d) == k
        for name, arr in snapshot.items():
            assert np.array_equal(d[name], arr, equal_nan=True), name
    assert "_hden_exact" not in params


@needs_reco
def test_a_fallback_result_above_tol_raises(reco_plain):
    params, forcing, fishing = _setup(reco_plain)
    y = _perturbed(params)
    idx = hden_index(y, params, forcing)
    with pytest.raises(RuntimeError, match="handling time"):
        solve_hden(
            y,
            params,
            forcing,
            fishing,
            idx,
            params["_cmco_qb"],
            tol=1e-30,
            max_iter=3,
        )


def test_hden_index_uses_the_stanza_ratio_where_finite():
    y = np.array([0.0, 2.0, 3.0])
    ratio = np.array([np.nan, np.nan, 0.5])
    out = hden_index(
        y, {"Bbase": np.array([0.0, 1.0, 4.0])}, {"StanzaPredRatio": ratio}
    )
    np.testing.assert_array_equal(out, [0.0, 2.0, 2.0])


def test_recalibrated_hden0_rejects_a_ceiling_below_the_base_consumption(reco_plain):
    """``q0 >= cmco_qb`` has no positive fixed point: name the groups."""
    params, _, _ = _setup(reco_plain)
    cmco_qb = np.array(params["_cmco_qb"], dtype=float)
    g = int(np.flatnonzero(cmco_qb > 0)[0])
    cmco_qb[g] = 1e-9
    with pytest.raises(ValueError, match=rf"handling time.*\[{g}\]"):
        recalibrated_hden0(params, cmco_qb)


# ---------------------------------------------------------------------------
# The vectorised kernel and the bracketed iteration
# ---------------------------------------------------------------------------


def _pattern(n):
    """Hden factors 1.3, 0.8, 1.3, ... per group."""
    return np.where(np.arange(n) % 2 == 0, 1.3, 0.8)


def _kernel_call(y, params, forcing, fishing, hden_of=None):
    """One ``deriv_vector`` call with a stub ``_hden_solver``.

    The stub records the ``KernelInputs`` and returns ``hden_of(inputs)``
    (default: the inputs' Hden0), computing ``HdenKernel(inputs).q`` at that
    Hden inside the call. Returns ``(seen, rates)``.
    """
    seen = {}

    def stub(inputs):
        h = inputs.hden0.copy() if hden_of is None else hden_of(inputs)
        seen.update(inputs=inputs, h=h, q=HdenKernel(inputs).q(h))
        return h

    p = dict(params, _hden_exact=True, _hden_solver=stub)
    p["_hden"] = np.array(params["_hden0"], dtype=float, copy=True)
    rates = {}
    deriv_vector(np.array(y, dtype=float), p, forcing, fishing, rates=rates)
    return seen, rates


class _GroupMediation(MediationCollection):
    """Group multipliers only. ``compute_fleet_multipliers`` returns
    ``n_fleets`` values, but ``fishing_mortality`` indexes them by the
    1-based gear number (an IndexError at the last gear, before this task);
    one value more keeps the fishing part out of this test."""

    def compute_fleet_multipliers(self, BB, Bbase, n_fleets):
        return np.ones(n_fleets + 1)


def _mediated(params):
    """A mediation collection: group 12 mediates the first three links."""
    prey, pred = params["_link_prey"], params["_link_pred"]
    pairs = [(a, b) for a, b in zip(prey, pred) if a >= 1 and b >= 1][:3]
    shape = MediationShape(1, "m", np.array([0.0, 1.0, 4.0]), np.array([0.4, 1, 1.6]))
    links = [MediationLink(1, 11, prey_idx=a - 1, pred_idx=b - 1) for a, b in pairs]
    return _GroupMediation(shapes=[shape], links=links)


KERNEL_CASES = [
    "base",
    "zero_prey",
    "pooling",
    "handle_switch",
    "eat_eff",
    "mediation",
    "dense_loop",
]


@needs_reco
@pytest.mark.parametrize("case", KERNEL_CASES)
def test_the_vector_kernel_matches_the_link_loop(reco_plain, case):
    params, forcing, fishing = _setup(reco_plain)
    y = _perturbed(params)
    active = np.asarray(params["ActiveLink"]) != 0
    if case == "zero_prey":
        y[4] = 0.0
    elif case == "pooling":
        params["HandleSelf"] = np.where(active, 0.5, 0.0)
        params["ScrambleSelf"] = np.where(active, 0.5, 0.0)
    elif case == "handle_switch":
        params["HandleSwitch"] = np.where(active, 2.0, 0.0)
    elif case == "eat_eff":
        params["_eat_eff"] = np.full(len(y), 0.7)
    elif case == "mediation":
        params["_mediation"] = _mediated(params)
    elif case == "dense_loop":  # no link arrays: the dense loop's links
        del params["_link_prey"], params["_link_pred"]
    seen, rates = _kernel_call(
        y, params, forcing, fishing, lambda k: k.hden0 * _pattern(len(k.hden0))
    )
    inputs = seen["inputs"]
    assert isinstance(inputs, KernelInputs)
    lp, lq = inputs.link_prey, inputs.link_pred
    # the case reaches the kernel
    if case == "zero_prey":
        assert np.any(inputs.BB[lp] == 0.0)
    elif case == "pooling":
        assert np.all(inputs.ScrambleSelf[lp, lq] == 0.5)
    elif case == "handle_switch":
        assert np.all(inputs.HandleSwitch[lp, lq] == 2.0)
    elif case == "eat_eff":
        consumers = np.asarray(params["PP_type"])[lq] == 0
        assert np.all(inputs.eat_eff[lq[consumers & (lq > 0)]] == 0.7)
    elif case == "mediation":
        assert np.sum(inputs.med_multipliers[lp, lq] != 1.0) == 3
    elif case == "dense_loop":
        assert lp.min() >= 1 and lq.max() <= params["NUM_LIVING"]
    q_loop = rates["HdenConsumption"]
    assert np.count_nonzero(q_loop) >= 10
    np.testing.assert_allclose(seen["q"], q_loop, rtol=1e-13, atol=0)
    # rates["Hden"] is the stub's Hden, used by the final kernel pass
    np.testing.assert_array_equal(rates["Hden"], seen["h"])


def _states(params):
    """The alternating state and five log-normal ones (sigma 0.7)."""
    base = np.array(params["Bbase"], dtype=float)
    nl = int(params["NUM_LIVING"]) + int(params["NUM_DEAD"])
    rng = np.random.default_rng(0)
    out = [_perturbed(params)]
    for _ in range(5):
        y = base.copy()
        y[1 : nl + 1] *= np.exp(0.7 * rng.standard_normal(nl))
        out.append(y)
    return out


@needs_reco
def test_kernel_and_full_solves_agree(reco_plain):
    params, forcing, fishing = _setup(reco_plain)
    for y in _states(params):
        seen, _ = _kernel_call(y, params, forcing, fishing)
        hk, nk = solve_hden_kernel(seen["inputs"])
        idx = hden_index(y, params, forcing)
        hf, nf = solve_hden(y, params, forcing, fishing, idx, params["_cmco_qb"])
        print("kernel passes", nk, "full-derivative passes", nf)
        assert np.max(np.abs(hk / hf - 1.0)) < 1e-11


def _kernel_map(inputs):
    """``(g, h0, active)`` of the kernel fixed point, as ``solve_hden_kernel``."""
    cq = np.asarray(inputs.cmco_qb, dtype=float)
    idx = np.asarray(inputs.index, dtype=float)
    active = (cq > 0) & (idx > 0)
    denom = np.where(active, cq * idx, 1.0)
    kernel = HdenKernel(inputs)

    def g(h):
        out = np.ones_like(h)
        out[active] = 1.0 + h[active] * kernel.q(h)[active] / denom[active]
        return out

    h0 = np.where(active, inputs.hden0, 1.0)
    return g, h0, active


@needs_reco
def test_the_bracketed_iteration_needs_few_passes(reco_plain):
    params, forcing, fishing = _setup(reco_plain)
    seen, _ = _kernel_call(_perturbed(params), params, forcing, fishing)
    g, h0, active = _kernel_map(seen["inputs"])
    h, passes = iterate_hden(g, h0, active)
    hn, plain = iterate_hden(g, h0, active, accel="none", max_iter=300, fallback=False)
    print("bracketed passes", passes, "undamped passes", plain)
    assert passes <= 12
    assert passes < plain / 2
    np.testing.assert_allclose(h, hn, rtol=1e-11, atol=0)

    def double(hv):
        return 1.0 + 0.9 * hv / (0.1 + hv)

    root = (1.8 + np.sqrt(1.8**2 + 0.4)) / 2  # h^2 - 1.8 h - 0.1 = 0
    h0 = np.array([1.0, 1.5, 5.0, 40.0])
    h, n = iterate_hden(double, h0, np.ones(4, dtype=bool))
    np.testing.assert_allclose(h, root, rtol=1e-12)

    def coupled(hv):
        s = hv / (0.1 + hv)
        return 1.0 + 0.45 * s + 0.45 * s[::-1]

    h, n = iterate_hden(coupled, h0, np.ones(4, dtype=bool), coupled=True)
    np.testing.assert_allclose(h, root, rtol=1e-12)
    # an inactive component keeps its start value
    act = np.array([True, False, True, True])
    h, n = iterate_hden(double, h0, act)
    assert h[1] == 1.5
    np.testing.assert_allclose(h[act], root, rtol=1e-12)


def test_a_non_positive_hden_is_never_accepted():
    def g(hv):
        return -1.0 - hv  # fixed point -0.5

    with pytest.raises(RuntimeError, match=r"handling time.*groups \[0, 1\]"):
        iterate_hden(g, np.array([2.0, 3.0]), np.ones(2, dtype=bool))
