import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params


def test_detritus_link_coverage():
    """Ensure each detritus group has at least one incoming link (DetTo or FishTo).

    This captures cases where a detritus column receives no sources from living groups
    and no fish discards link, which can lead to consumption-without-inputs and late
    biomass collapse.
    """
    ECOPATH_DIR = "tests/data/rpath_reference/ecopath"
    model_df = pd.read_csv(ECOPATH_DIR + "/model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR + "/diet_matrix.csv")

    params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
    params.model = model_df
    params.diet = diet_df

    r = rpath(params)
    scenario = rsim_scenario(r, params, years=range(1, 3))

    rs = scenario.params
    det_to = rs.DetTo
    fish_to = rs.FishTo if hasattr(rs, 'FishTo') else np.array([])

    # For each detritus column (1-based expectation in DetTo), check presence
    for det_idx in range(rs.NUM_DEAD):
        expected_det_to = rs.NUM_LIVING + det_idx + 1  # 1-based target
        has_det_link = expected_det_to in det_to.tolist()
        has_fish_link = expected_det_to in fish_to.tolist() if fish_to.size > 0 else False
        assert (
            has_det_link or has_fish_link
        ), f"Detritus index {det_idx} (expected DetTo={expected_det_to}) has no DetTo or FishTo links"
