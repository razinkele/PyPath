"""Handling time as a fixed point of the state (continuous path)."""

import numpy as np
import pytest

from pypath.core.ecosim_continuous import hden_index, recalibrated_hden0, solve_hden
from pypath.core.ecosim_deriv import deriv_vector
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
    p = dict(params, _hden=h.copy(), _hden_exact=True)
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
    before = (params["_hden"].copy(), params["_hden0"].copy())
    y = _perturbed(params)
    idx = hden_index(y, params, forcing)
    solve_hden(y, params, forcing, fishing, idx, params["_cmco_qb"])
    assert np.array_equal(params["_hden"], before[0])
    assert np.array_equal(params["_hden0"], before[1])
    assert "_hden_exact" not in params


def test_hden_index_uses_the_stanza_ratio_where_finite():
    y = np.array([0.0, 2.0, 3.0])
    ratio = np.array([np.nan, np.nan, 0.5])
    out = hden_index(
        y, {"Bbase": np.array([0.0, 1.0, 4.0])}, {"StanzaPredRatio": ratio}
    )
    np.testing.assert_array_equal(out, [0.0, 2.0, 2.0])
