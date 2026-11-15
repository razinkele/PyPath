"""Scramble/handling suite weights against Rpath's (REco), and pooled runs."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecosim import rsim_run, rsim_scenario
from tests.continuous_support import needs_reco, reco_plain, reco_stanzas, scenario

CSV = (
    Path(__file__).parent / "data" / "rpath_reference" / "ecosim" / "suite_weights.csv"
)
POOLED_RK4 = CSV.parent / "biomass_pooled_rk4.csv"

pytestmark = needs_reco


def test_suite_weights_match_rpath():
    ref = pd.read_csv(CSV)
    model, params = reco_plain()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scen = rsim_scenario(model, params, years=range(1, 3))
    p = scen.params
    assert np.array_equal(p.PreyFrom, ref["from"].to_numpy())
    assert np.array_equal(p.PreyTo, ref["to"].to_numpy())
    assert np.all(ref["ppw"] >= 0) and np.all(ref["pyw"] >= 0)
    np.testing.assert_allclose(p.PredPredWeight, ref["ppw"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(p.PreyPreyWeight, ref["pyw"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("cmco", [2.0, 1000.0])
def test_pooled_reco_runs(cmco):
    scen = scenario(reco_plain(), rule="ewe", effort=2.0, years=range(1, 3))
    p = scen.params
    p.ScrambleSelf = np.full(len(p.PreyFrom), 0.5)
    p.HandleSelf = np.full(len(p.PreyFrom), 0.5)
    p.CmCo[1 : p.NUM_LIVING + 1] = cmco
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rsim_run(scen, method="lsoda", years=range(1, 3))
    months = out.solver_info["months"]
    assert len(months) == 24
    assert max(m["hden_passes_max"] for m in months) <= 30
    nl = p.NUM_LIVING
    ratio = out.out_Biomass[1:, 1 : nl + 1] / p.B_BaseRef[1 : nl + 1]
    assert np.min(ratio) > 0.5


def _pooled_rk4(weight):
    """PyPath's RK4 run of REco with stanzas, the "rpath" rule, ScrambleSelf
    and HandleSelf ``weight`` on every link and effort x 2, ten years."""
    model, params = reco_stanzas()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scen = rsim_scenario(model, params, years=range(1, 11), fast_group_rule="rpath")
        p = scen.params
        p.ScrambleSelf = np.full(len(p.ScrambleSelf), weight)
        p.HandleSelf = np.full(len(p.HandleSelf), weight)
        scen.fishing.ForcedEffort = scen.fishing.ForcedEffort * 2.0
        out = rsim_run(scen, method="RK4", years=range(1, 11))
    return scen, out


def test_pooled_rk4_trajectory_matches_rpath():
    """Rpath 1.1.0's trajectory of a pooled model (README): REco with stanzas,
    ScrambleSelf and HandleSelf 0.5 (``rsim.params``' ``scrambleselfwt`` and
    ``handleselfwt``), effort x 2, ten years of RK4. PyPath's monthly biomass
    agrees within 1e-4 relative in every group and month: measured 3.6e-5
    (JuvFlatfish1, month 120), where the same run without pooling agrees
    with Rpath's to 2.6e-5 (the same group and month), so the rest is the
    stanza groups', not the pooling's. Without the pooling (weights 0) the
    trajectory misses this reference by 0.18 (weights 0.3: 0.079), so the
    bound tells them apart."""
    ref = pd.read_csv(POOLED_RK4)
    scen, out = _pooled_rk4(0.5)
    k = ref.shape[1]
    assert list(ref.columns[1:]) == list(scen.params.spname[1:k])
    want = ref.to_numpy(dtype=float)[:, 1:]
    assert want.shape == (120, k - 1)
    got = out.out_Biomass[1:121, 1:k]  # PyPath's row 0 is the start state
    err = np.abs(got / want - 1)
    print(f"\npooled RK4 against Rpath 1.1.0: max relative error {err.max():.2e}")
    assert err.max() <= 1e-4
