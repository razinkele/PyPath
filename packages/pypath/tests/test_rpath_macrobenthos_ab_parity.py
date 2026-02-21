from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

BASE = Path("tests/data/rpath_reference")
ECOPATH_DIR = BASE / "ecopath"
ECOSIM_DIR = BASE / "ecosim"


def test_macrobenthos_ab_initial_parity_roundtrip():
    """Run a short AB scenario and assert early-step parity vs Rpath AB

    This test runs 1 year (12 months) of the AB integrator and compares
    the Macrobenthos trajectory against the Rpath-provided AB CSV for the
    same initial conditions. It also verifies that the instrumentation
    callback is invoked and produces numeric payloads for the requested
    group, allowing future assertions on intermediate derivative history.
    """
    model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

    groups = model_df["Group"].tolist()
    params = create_rpath_params(groups, model_df["Type"].tolist())
    params.model = model_df
    params.diet = diet_df

    # Instrument Macrobenthos
    params.INSTRUMENT_GROUPS = ["Macrobenthos"]
    instrumented = []

    def cb(payload):
        instrumented.append(payload)

    params.instrument_callback = cb

    pypath_model = rpath(params)

    # Short scenario: 2 years (24 monthly steps) to give Adams-Bashforth
    # a minimal history window while still keeping the run fast.
    scenario = rsim_scenario(pypath_model, params, years=range(1, 3))

    out_ab = rsim_run(scenario, method="AB", years=range(1, 3))

    rpath_traj_ab = pd.read_csv(ECOSIM_DIR / "biomass_trajectory_ab.csv")

    # Compare the first few months of Macrobenthos
    macrob_idx = groups.index("Macrobenthos")
    macrob_out_idx = macrob_idx + 1  # PyPath output has a leading 'Outside' column

    rvals = rpath_traj_ab["Macrobenthos"].values
    pvals = out_ab.out_Biomass[: len(rvals), macrob_out_idx]

    L = min(len(rvals), len(pvals), 6)  # check up to first 6 months

    # Allow a small tolerance for early-step parity (numerical integrator
    # differences at the 1e-6 level are acceptable but larger differences
    # should be investigated).
    assert np.allclose(rvals[:L], pvals[:L], rtol=2e-6, atol=1e-9), (
        "Macrobenthos AB early-step parity failed: rvals[:L] != pvals[:L]"
    )

    # Instrumentation callback must have been invoked at least once
    assert len(instrumented) > 0, "Instrumentation callback was not invoked"

    # Validate payload structure and numeric finiteness
    payload = instrumented[0]
    assert payload.get("method") == "AB"
    assert isinstance(payload.get("groups"), list)
    # The group index must correspond to Macrobenthos (0-based index)
    assert macrob_idx in payload["groups"]
    # deriv_current and new_state should be numeric lists and finite
    assert np.isfinite(np.asarray(payload["deriv_current"]).astype(float)).all()
    assert np.isfinite(np.asarray(payload["new_state"]).astype(float)).all()
