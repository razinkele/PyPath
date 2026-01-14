from pathlib import Path

import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

BASE = Path("tests/data/rpath_reference")
ECOPATH_DIR = BASE / "ecopath"
ECOSIM_DIR = BASE / "ecosim"


def test_instrumentation_1based_numeric_warns_and_converts():
    """Passing numeric 1-based indices should emit a DeprecationWarning and be converted to 0-based."""
    model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

    groups = model_df["Group"].tolist()
    params = create_rpath_params(groups, model_df["Type"].tolist())
    params.model = model_df
    params.diet = diet_df

    # Use a 1-based numeric index for Macrobenthos (legacy usage)
    macrob_idx = groups.index("Macrobenthos")
    legacy_one_based = macrob_idx + 1

    params.INSTRUMENT_GROUPS = [legacy_one_based]
    # Explicitly opt in to legacy 1-based numeric conversion for this test
    params.INSTRUMENT_ASSUME_1BASED = True
    instrumented = []

    def cb(payload):
        instrumented.append(payload)

    params.instrument_callback = cb

    pypath_model = rpath(params)
    scenario = rsim_scenario(pypath_model, params, years=range(1, 3))

    # The run should emit a DeprecationWarning and still produce instrumentation
    with pytest.warns(DeprecationWarning):
        _out_ab = rsim_run(scenario, method="AB", years=range(1, 3))

    assert len(instrumented) > 0, "Instrumentation callback was not invoked"
    # Validate that the legacy 1-based index was converted to the correct 0-based group index.
    # Prefer checking that the params attribute was normalized; fall back to checking payload groups.
    normalized_attr = getattr(scenario.params, 'INSTRUMENT_GROUPS', None)
    assert (
        (isinstance(normalized_attr, (list, tuple)) and macrob_idx in normalized_attr)
        or any(any(groups[g] == "Macrobenthos" for g in p.get("groups", [])) for p in instrumented)
    ), "macrob_idx not found in params.INSTRUMENT_GROUPS or in any instrumentation payload"
