"""Tests that NoIntegrate detritus groups have zero derivatives."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import (
    _build_active_link_matrix,
    _build_link_matrix,
    rsim_scenario,
)
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.params import create_rpath_params

RE = Path(__file__).parent / "data" / "rpath_reference"


@pytest.fixture()
def setup():
    """Load model data and build scenario + state vector."""
    model_df = pd.read_csv(RE / "ecopath" / "model_params.csv")
    params = create_rpath_params(
        model_df["Group"].tolist(), [int(t) for t in model_df["Type"].tolist()]
    )
    params.model = model_df
    params.diet = pd.read_csv(RE / "ecopath" / "diet_matrix.csv")

    pypath_model = rpath(params)
    scenario = rsim_scenario(pypath_model, params, years=range(1, 101))

    rpath_df = pd.read_csv(RE / "ecosim" / "biomass_trajectory_rk4.csv")
    row = rpath_df.iloc[0]
    state_r = np.zeros(scenario.params.NUM_GROUPS + 1)
    for i, g in enumerate(model_df["Group"].tolist(), start=1):
        if g in row:
            state_r[i] = float(row[g])

    pdict = {
        "NUM_GROUPS": scenario.params.NUM_GROUPS,
        "NUM_LIVING": scenario.params.NUM_LIVING,
        "NUM_DEAD": scenario.params.NUM_DEAD,
        "NUM_GEARS": scenario.params.NUM_GEARS,
        "PB": scenario.params.PBopt,
        "QB": scenario.params.FtimeQBOpt,
        "M0": scenario.params.MzeroMort.copy(),
        "Unassim": scenario.params.UnassimRespFrac,
        "ActiveLink": _build_active_link_matrix(scenario.params),
        "VV": _build_link_matrix(scenario.params, scenario.params.VV),
        "DD": _build_link_matrix(scenario.params, scenario.params.DD),
        "QQbase": _build_link_matrix(scenario.params, scenario.params.QQ),
        "Bbase": scenario.params.B_BaseRef,
        "PP_type": scenario.params.PP_type,
        "spname": scenario.params.spname,
        "NoIntegrate": scenario.params.NoIntegrate,
    }

    forcing = {
        "Ftime": scenario.start_state.Ftime.copy(),
        "ForcedBio": (
            scenario.forcing.ForcedBio[0]
            if hasattr(scenario.forcing, "ForcedBio")
            else np.zeros(scenario.params.NUM_GROUPS + 1)
        ),
        "ForcedPrey": (
            getattr(
                scenario.forcing,
                "ForcedPrey",
                np.ones(scenario.params.NUM_GROUPS + 1),
            )[0]
            if hasattr(scenario.forcing, "ForcedPrey")
            else np.ones(scenario.params.NUM_GROUPS + 1)
        ),
    }

    fishing = {
        "FishQ": scenario.params.FishQ,
        "FishFrom": scenario.params.FishFrom,
        "FishThrough": getattr(scenario.params, "FishThrough", None),
    }

    d_idx = pdict["spname"].index("Discards")

    return state_r, pdict, forcing, fishing, d_idx


def test_nointegrate_detritus_derivative_is_zero(setup):
    """Derivatives at Rpath start should be zero for NoIntegrate detritus."""
    state_r, pdict, forcing, fishing, d_idx = setup
    k1 = deriv_vector(state_r.copy(), pdict, forcing, fishing)
    assert abs(k1[d_idx]) == 0.0


def test_rk4_step_leaves_nointegrate_detritus_unchanged(setup):
    """An RK4 step should leave Discards unchanged when NoIntegrate is set."""
    state_r, pdict, forcing, fishing, d_idx = setup
    dt = 1.0 / 12.0

    k1 = deriv_vector(state_r.copy(), pdict, forcing, fishing)
    k2 = deriv_vector(state_r + 0.5 * dt * k1, pdict, forcing, fishing)
    k3 = deriv_vector(state_r + 0.5 * dt * k2, pdict, forcing, fishing)
    k4 = deriv_vector(state_r + dt * k3, pdict, forcing, fishing)

    s_next = state_r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    assert abs(s_next[d_idx] - state_r[d_idx]) < 1e-9
