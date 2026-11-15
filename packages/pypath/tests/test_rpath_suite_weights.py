"""Scramble/handling suite weights against Rpath's (REco), and pooled runs."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecosim import rsim_run, rsim_scenario
from tests.continuous_support import needs_reco, reco_plain, scenario

CSV = (
    Path(__file__).parent / "data" / "rpath_reference" / "ecosim" / "suite_weights.csv"
)

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
