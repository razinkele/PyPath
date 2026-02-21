from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import (
    _build_active_link_matrix,
    _build_link_matrix,
    _normalize_fishing_input,
    rsim_run,
    rsim_scenario,
)
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.params import create_rpath_params


def test_m0_unchanged_when_fishing_correct():
    """Verify that M0 is NOT nudged when fishing link arrays are correctly
    passed to the derivative computation.

    With the fix that copies FishFrom/FishThrough/FishQ from params to
    fishing_dict, the equilibrium derivative is exactly zero at the Ecopath
    balance, so the M0 adjustment logic should be a no-op.
    """

    # Load model
    ECOPATH_DIR = str(Path(__file__).parent / "data" / "rpath_reference" / "ecopath")
    model_df = pd.read_csv(ECOPATH_DIR + "/model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR + "/diet_matrix.csv")
    params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
    params.model = model_df
    params.diet = diet_df

    # Build balanced rpath and scenario
    r = rpath(params)
    scenario = rsim_scenario(r, params, years=range(1, 101))

    rs = scenario.params
    m0_before = rs.MzeroMort.copy()

    # Build params_dict and fishing_dict to verify equilibrium
    n_groups = rs.NUM_GROUPS + 1
    params_dict = {
        'NUM_GROUPS': rs.NUM_GROUPS,
        'NUM_LIVING': rs.NUM_LIVING,
        'NUM_DEAD': rs.NUM_DEAD,
        'NUM_GEARS': rs.NUM_GEARS,
        'PB': rs.PBopt,
        'QB': rs.FtimeQBOpt,
        'M0': rs.MzeroMort.copy(),
        'Unassim': rs.UnassimRespFrac,
        'ActiveLink': _build_active_link_matrix(rs),
        'VV': _build_link_matrix(rs, rs.VV),
        'DD': _build_link_matrix(rs, rs.DD),
        'QQbase': _build_link_matrix(rs, rs.QQ),
        'Bbase': rs.B_BaseRef,
        'PP_type': rs.PP_type,
    }

    forcing0 = {
        'Ftime': scenario.start_state.Ftime.copy(),
        'ForcedBio': np.where(scenario.forcing.ForcedBio[0] > 0, scenario.forcing.ForcedBio[0], 0),
        'ForcedMigrate': scenario.forcing.ForcedMigrate[0],
        'ForcedEffort': (scenario.fishing.ForcedEffort[0] if 0 < len(scenario.fishing.ForcedEffort) else np.ones(rs.NUM_GEARS + 1)),
    }

    # Build the correct fishing_dict (with FishFrom/FishQ from params)
    fishing_dict = _normalize_fishing_input(scenario.fishing, n_groups)
    if not fishing_dict.get('FishFrom', []):
        fishing_dict['FishFrom'] = getattr(rs, 'FishFrom', np.array([0]))
        fishing_dict['FishThrough'] = getattr(rs, 'FishThrough', np.array([0]))
        fishing_dict['FishQ'] = getattr(rs, 'FishQ', np.array([0.0]))

    # Pre-check: equilibrium derivative with correct fishing should be ~zero
    state = scenario.start_state.Biomass.copy()
    init_deriv = deriv_vector(state, params_dict, forcing0, fishing_dict)
    for i in range(1, rs.NUM_LIVING + 1):
        assert abs(init_deriv[i]) < 1e-8, (
            f"Group {rs.spname[i]} has non-zero equilibrium derivative {init_deriv[i]:.3e} "
            f"even with correct fishing"
        )

    # Run rsim_run (which includes M0 adjustment logic)
    _ = rsim_run(scenario, method='RK4', years=range(1, 2))

    # M0 should NOT have changed since equilibrium was already exact
    m0_after = scenario.params.MzeroMort
    np.testing.assert_allclose(
        m0_before, m0_after, atol=1e-10,
        err_msg="M0 was unexpectedly modified even though equilibrium was exact"
    )
