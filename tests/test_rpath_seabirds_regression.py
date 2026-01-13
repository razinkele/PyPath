"""Regression test for Seabirds RK4 mismatch vs Rpath reference.

This test encodes the current failing behavior: it asserts that PyPath's annual
Seabirds trajectory matches the Rpath reference to high precision. It will
fail until the underlying drift is fixed, serving as a reproducible regression
capture.
"""
import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

REFERENCE_DIR = 'tests/data/rpath_reference'
ECOPATH_DIR = REFERENCE_DIR + '/ecopath'
ECOSIM_DIR = REFERENCE_DIR + '/ecosim'


def test_seabirds_match_rpath_rk4():
    model_df = pd.read_csv(ECOPATH_DIR + '/model_params.csv')
    diet_df = pd.read_csv(ECOPATH_DIR + '/diet_matrix.csv')

    groups = model_df['Group'].tolist()
    types = model_df['Type'].tolist()

    params = create_rpath_params(groups, types)
    params.model = model_df
    params.diet = diet_df

    pypath_model = rpath(params)
    scenario = rsim_scenario(pypath_model, params, years=range(1, 101))

    if 'Seabirds' not in scenario.params.spname:
        return

    sidx = scenario.params.spname.index('Seabirds')

    out = rsim_run(scenario, method='rk4', years=range(1, 101))
    biom = out.out_Biomass  # monthly snapshots

    # aggregate to annual by taking monthly means per year (12-month blocks)
    biom_series = biom[:, sidx]
    n_full_years = len(biom_series) // 12
    if n_full_years == 0:
        pytest.skip("Not enough months to aggregate to annual")
    p_pYear = biom_series[: n_full_years * 12].reshape(n_full_years, 12).mean(axis=1)

    rpath_traj = pd.read_csv(ECOSIM_DIR + '/biomass_trajectory_rk4.csv')
    r_pSeab = rpath_traj['Seabirds'].values
    r_pYear = r_pSeab[: p_pYear.shape[0] * 12].reshape(-1, 12).mean(axis=1)

    L = min(len(r_pYear), len(p_pYear))
    if np.std(r_pYear[:L]) < 1e-12 or np.std(p_pYear[:L]) < 1e-12:
        corr = 1.0 if np.allclose(r_pYear[:L], p_pYear[:L], atol=1e-12) else np.nan
    else:
        corr = np.corrcoef(r_pYear[:L], p_pYear[:L])[0, 1]

    rel_err = abs(p_pYear[-1] - r_pYear[-1]) / (r_pYear[-1] if r_pYear[-1] > 0 else 1.0)

    # Expected: very close match. This will fail until the drift is fixed.
    assert corr > 0.99, f"Seabirds annual correlation is too low: {corr:.6f}"
    assert rel_err < 0.01, f"Seabirds relative final error too large: {rel_err:.6f}"
