import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import (
    _build_active_link_matrix,
    _build_link_matrix,
    rsim_run,
    rsim_scenario,
)
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.params import create_rpath_params


def test_m0_adjustment_persisted():
    """Verify that small initial-derivative M0 adjustments are persisted to params.MzeroMort

    This ensures diagnostics built from `scenario.params` see the same M0 used by the integrator
    and that the adjusted groups have near-zero derivatives after the adjustment."""

    # Load model
    ECOPATH_DIR = "tests/data/rpath_reference/ecopath"
    model_df = pd.read_csv(ECOPATH_DIR + "/model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR + "/diet_matrix.csv")
    params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
    params.model = model_df
    params.diet = diet_df

    # Build balanced rpath and scenario
    r = rpath(params)
    scenario = rsim_scenario(r, params, years=range(1, 101))

    rs = scenario.params

    # Build params_dict before any adjustment (copy original M0)
    params_dict_pre = {
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

    # Compute initial derivative with original M0
    fish_base = {'FishingMort': np.zeros(rs.NUM_GROUPS + 1)}
    # DEBUG: request breakdown for Seabirds inside this pre-run derivative call
    seab_idx = rs.spname.index('Seabirds')
    params_dict_pre['TRACE_DEBUG_GROUPS'] = [seab_idx]
    params_dict_pre['spname'] = rs.spname
    init_deriv_pre = deriv_vector(scenario.start_state.Biomass.copy(), params_dict_pre, forcing0, fish_base)

    ADJUST_DERIV_MAX = 1e-3
    to_adjust = [i for i in range(1, rs.NUM_GROUPS + 1) if np.isfinite(init_deriv_pre[i]) and abs(init_deriv_pre[i]) < ADJUST_DERIV_MAX and scenario.start_state.Biomass[i] > 0]

    # Sanity: expect at least one adjusted group (Seabirds)
    assert len(to_adjust) > 0, "No groups identified for M0 adjustment (test setup unexpected)"

    # DEBUG: print pre-run derivative and key params for inspection
    seab_idx = rs.spname.index('Seabirds')
    print(f"TEST-DEBUG: pre-run init_deriv_pre[Seabirds idx={seab_idx}] = {init_deriv_pre[seab_idx]:.6e}")
    print(f"TEST-DEBUG: pre-run init_deriv_pre[:10]={init_deriv_pre[:10]}")
    QQbase_pre = _build_link_matrix(rs, rs.QQ)
    print(f"TEST-DEBUG: pre-run M0 seabirds = {params_dict_pre['M0'][seab_idx]:.6e}")
    print(f"TEST-DEBUG: pre-run PB seabirds = {params_dict_pre['PB'][seab_idx]:.6e} QB = {params_dict_pre['QB'][seab_idx]:.6e} B = {rs.B_BaseRef[seab_idx]:.6e}")
    print(f"TEST-DEBUG: pre-run QQ consumption col_sum = {float(np.nansum(QQbase_pre[:, seab_idx])):.6e} predation row_sum = {float(np.nansum(QQbase_pre[seab_idx, :])):.6e}")
    print(f"TEST-DEBUG: pre-run forcing0 ForcedEffort[:4]={forcing0['ForcedEffort'][:4]} ForcedBio[:4]={forcing0['ForcedBio'][:4]}")
    print(f"TEST-DEBUG: pre-run fish_base length={len(fish_base['FishingMort'])} sample={fish_base['FishingMort'][:4]}")

    # Run rsim_run to trigger M0 adjustment logic (rsim_run persists adjusted M0)
    _ = rsim_run(scenario, method='RK4', years=range(1, 2))  # 1 year is sufficient

    # After run, params.MzeroMort should contain adjusted M0 values
    params_after = scenario.params

    # Build params_dict using persisted M0
    params_dict_post = params_dict_pre.copy()
    params_dict_post['M0'] = params_after.MzeroMort

    # Compute derivative with updated M0
    init_deriv_post = deriv_vector(scenario.start_state.Biomass.copy(), params_dict_post, forcing0, fish_base)

    # For each adjusted group, derivative should now be approximately zero
    for idx in to_adjust:
        # Allow small numerical residuals; exact zero may not be achievable due to
        # floating point arithmetic and approximations in component calculations.
        assert abs(init_deriv_post[idx]) < 1e-04, f"Group {idx} derivative not sufficiently small after M0 persistence: {init_deriv_post[idx]:.3e}"

    # Additionally, check at least one named group (Seabirds) was adjusted
    seab_idx = rs.spname.index('Seabirds')
    assert seab_idx in to_adjust, "Expected Seabirds to be in the adjustment set"

    # Confirm persisted M0 changed for Seabirds
    assert not np.isclose(params_dict_pre['M0'][seab_idx], params_after.MzeroMort[seab_idx]), "Seabirds M0 not persisted/changed"
