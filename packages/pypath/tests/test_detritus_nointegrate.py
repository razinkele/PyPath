"""Fast-turnover (NoIntegrate) and detritus groups follow Rpath.

Rpath's deriv_vector computes a full derivative for every group, including
detritus and the fast-turnover groups; NoIntegrate only selects Adams-
Bashforth's fast-equilibrium update. RK4 integrates every group. At the
Ecopath balance every derivative is zero, so a run starts at equilibrium.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pypath.core.ecosim as ecosim_mod
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

RE = Path(__file__).parent / "data" / "rpath_reference"


@pytest.fixture(scope="module")
def scenario():
    model_df = pd.read_csv(RE / "ecopath" / "model_params.csv")
    params = create_rpath_params(
        model_df["Group"].tolist(), [int(t) for t in model_df["Type"].tolist()]
    )
    params.model = model_df
    params.diet = pd.read_csv(RE / "ecopath" / "diet_matrix.csv")
    return rsim_scenario(
        rpath(params), params, years=range(1, 3), fast_group_rule="rpath"
    )


@pytest.fixture(scope="module")
def run_inputs(scenario):
    """The state, params, forcing and fishing rsim_run hands deriv_vector."""
    captured = {}
    original = ecosim_mod.deriv_vector

    def spy(state, params, forcing, fishing, *args, **kwargs):
        captured.setdefault("args", (state.copy(), params, forcing, fishing))
        return original(state, params, forcing, fishing, *args, **kwargs)

    ecosim_mod.deriv_vector = spy
    try:
        rsim_run(scenario, method="RK4", years=range(1, 2))
    finally:
        ecosim_mod.deriv_vector = original
    return captured["args"]


def test_every_group_starts_at_equilibrium(scenario, run_inputs):
    """dB/dt = 0 at the Ecopath balance, detritus and producers included."""
    state, params, forcing, fishing = run_inputs
    deriv = ecosim_mod.deriv_vector(state.copy(), params, forcing, fishing)
    n_bio = scenario.params.NUM_LIVING + scenario.params.NUM_DEAD
    rel = np.abs(deriv[1 : n_bio + 1]) / state[1 : n_bio + 1]
    worst = int(np.argmax(rel)) + 1
    assert rel.max() < 1e-9, (scenario.params.spname[worst], rel.max())


def test_no_integrate_follows_rpaths_turnover_rule(scenario):
    """Rpath: fast equilibrium when M0 * B > 2 * 12 * steps_per_month.

    In REco that is Zooplankton, Phytoplankton and Detritus; Discards turns
    over slowly and is integrated like any other group.
    """
    names = list(scenario.params.spname)
    n_bio = scenario.params.NUM_LIVING + scenario.params.NUM_DEAD
    fast = {
        names[i] for i in range(1, n_bio + 1) if scenario.params.NoIntegrate[i] != 0
    }
    assert fast == {"Zooplankton", "Phytoplankton", "Detritus"}


def test_no_integrate_does_not_mask_the_derivative(scenario, run_inputs):
    """NoIntegrate selects AB's update; it must not change deriv_vector."""
    state, params, forcing, fishing = run_inputs
    perturbed = state.copy()
    d_idx = list(scenario.params.spname).index("Discards")
    perturbed[d_idx] *= 1.5
    flagged = dict(params)
    flagged["NoIntegrate"] = np.ones_like(np.asarray(params["NoIntegrate"]))
    plain = dict(params)
    plain["NoIntegrate"] = np.zeros_like(np.asarray(params["NoIntegrate"]))
    d_flagged = ecosim_mod.deriv_vector(perturbed.copy(), flagged, forcing, fishing)
    d_plain = ecosim_mod.deriv_vector(perturbed.copy(), plain, forcing, fishing)
    np.testing.assert_array_equal(d_flagged, d_plain)
    assert d_plain[d_idx] < 0  # extra discards decay and are eaten
