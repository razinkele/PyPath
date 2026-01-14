from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

BASE = Path("tests/data/rpath_reference")
ECOPATH_DIR = BASE / "ecopath"


def test_instrumentation_accepts_zero_based_indices():
    model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
    groups = model_df["Group"].tolist()
    params = create_rpath_params(groups, model_df["Type"].tolist())
    params.model = model_df

    # use a numeric 0-based index for Macrobenthos
    macrob_idx = groups.index("Macrobenthos")
    params.INSTRUMENT_GROUPS = [macrob_idx]

    instrumented = []

    def cb(payload):
        instrumented.append(payload)

    params.instrument_callback = cb

    pypath_model = rpath(params)
    scenario = rsim_scenario(pypath_model, params, years=range(1, 3))
    _out_ab = rsim_run(scenario, method="AB", years=range(1, 3))

    assert len(instrumented) > 0
    # Prefer verifying the params attribute, fall back to checking payload group names
    normalized_attr = getattr(scenario.params, 'INSTRUMENT_GROUPS', None)
    assert (
        (isinstance(normalized_attr, (list, tuple)) and macrob_idx in normalized_attr)
        or any(any(groups[g] == 'Macrobenthos' for g in p.get('groups', [])) for p in instrumented)
    ), "macrob_idx not found in params.INSTRUMENT_GROUPS or in any instrumentation payload"
    # At least one payload should contain finite derivs for the requested groups
    assert any(np.isfinite(np.asarray(p.get("deriv_current", [])).astype(float)).all() for p in instrumented if p.get("groups")), "No payload contains finite derivs for instrumented groups"
