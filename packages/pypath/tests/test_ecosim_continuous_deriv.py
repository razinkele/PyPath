"""Opt-in derivative flags of the continuous pathway (exact Hden, R5, DF once)."""

import numpy as np
import pytest

from pypath.core.ecosim_deriv import _rpath_detritus_derivs, deriv_vector
from tests.continuous_support import needs_reco, run_dicts, scenario
from tests.continuous_support import reco_plain as _reco_plain


@pytest.fixture(scope="module")
def reco_plain():
    return _reco_plain()


@needs_reco
def test_exact_hden_leaves_hden_unchanged_and_uses_it(reco_plain):
    scen = scenario(reco_plain)
    nl = scen.params.NUM_LIVING
    scen.params.CmCo[1 : nl + 1] = 2.0
    state, params, forcing, fishing = run_dicts(scen)
    assert "_hden0" in params and "_hden" in params and "_cmco_qb" in params
    h = 1.3 * np.array(params["_hden0"], dtype=float)
    params["_hden"] = h.copy()
    params["_hden_exact"] = True
    r1, r2 = {}, {}
    d1 = deriv_vector(state.copy(), params, forcing, fishing, rates=r1)
    d2 = deriv_vector(state.copy(), params, forcing, fishing, rates=r2)
    np.testing.assert_array_equal(params["_hden"], h)
    np.testing.assert_array_equal(d1, d2)
    # the kernel ran once at H: compare with a frozen, non-exact call at H
    # whose first pass is the same and whose second pass is at the updated H
    assert r1["HdenConsumption"].shape == (scen.params.NUM_GROUPS + 1,)
    np.testing.assert_array_equal(r1["HdenConsumption"], r2["HdenConsumption"])
    # the exact call equals the first kernel pass of the iterating path:
    # an iterating call that starts at H and is frozen runs its last pass at
    # the updated Hden, so it differs from the exact result
    p2 = dict(params, _hden=h.copy(), _hden_exact=False, _hden_frozen=True)
    r3 = {}
    deriv_vector(state.copy(), p2, forcing, fishing, rates=r3)
    assert not np.allclose(r3["HdenConsumption"], r1["HdenConsumption"], rtol=1e-9)
    assert (r1["HdenConsumption"] > 0).any()


@needs_reco
def test_r5_closure_sends_mo_tot_to_detritus(reco_plain):
    def rates_for(flag):
        scen = scenario(reco_plain, effort=0.0)
        nl = scen.params.NUM_LIVING
        scen.params.UnassimRespFrac[1 : nl + 1] = 0.0
        scen.params.PaddP = np.full(scen.params.NUM_GROUPS + 1, 0.5)
        state, params, forcing, fishing = run_dicts(scen)
        assert params["_mo_tot"] is not None
        params["DETRITUS_MO_TOT"] = flag
        rates = {}
        deriv_vector(state.copy(), params, forcing, fishing, rates=rates)
        return rates, nl, scen.params.NUM_GROUPS

    on, nl, ng = rates_for(True)
    off, _, _ = rates_for(False)
    living = slice(1, nl + 1)
    dead = slice(nl + 1, ng + 1)
    mzero_on = on["MzeroLoss"][living].sum()
    np.testing.assert_allclose(on["DetGain"][dead].sum(), mzero_on, rtol=1e-12)
    assert abs(off["DetGain"][dead].sum() - off["MzeroLoss"][living].sum()) > 1e-6


NUM_LIVING, NUM_DEAD = 1, 2
BB = np.array([0.0, 1.0, 10.0, 5.0])
QQ = np.zeros((4, 4))
QQ[2, 1] = 0.3


def _flows(df_once, pass_on=True):
    return _rpath_detritus_derivs(
        np.zeros(4),
        QQ,
        BB,
        np.array([0.0, 0.5, 0.1, 0.0]),
        np.array([0.0, 0.2, 0.0, 0.0]),
        np.array([0, 1, 2, 2, 3]),
        np.array([0, 2, 3, 2, 3]),
        np.array([0.0, 1.0, 0.5, 0.5, 1.0]),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        NUM_LIVING,
        NUM_DEAD,
        det_passed_prop=np.array([0.0, 0.0, 0.2, 0.0]),
        det_ba=np.zeros(4),
        det_out=np.array([0.0, 0.0, 0.02, 0.03]),
        pass_on=pass_on,
        df_once=df_once,
    )


def test_df_once_counts_the_fate_once():
    gain_once, _ = _flows(True)
    gain_twice, _ = _flows(False)
    assert gain_once[3] == pytest.approx(2.0)  # 0.2 * 10 * 0.5 / 0.5
    assert gain_twice[3] == pytest.approx(1.0)  # 0.2 * 10 * 0.5
    # independent of pass_on: dropping the received flow still works
    assert _flows(True, pass_on=False)[0][3] == 0.0
