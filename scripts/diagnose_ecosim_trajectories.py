"""
Run diagnostics for AB and RK4 Ecosim trajectories and compare shapes & NaNs to R reference
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

ECOSIM_DIR = Path('tests/data/rpath_reference/ecosim')
ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')

# Load balanced model
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df.copy()
params.diet = diet_df
rpath_model, diag = rpath(params, debug=True)

# Utility to run and summarize a run

def summarize_run(method, years):
    print(f"\n--- Running method={method} years={len(years)} ---")
    scenario = rsim_scenario(rpath_model, params, years=years)
    out = rsim_run(scenario, method=method, years=years)
    # out.out_Biomass : 2D array shape (n_years, n_groups)
    B = out.out_Biomass
    print('out_Biomass shape:', B.shape)
    nan_counts = np.sum(np.isnan(B), axis=0)
    print('NaNs per group (first 10):', nan_counts[:10])
    last = B[-1, :]
    print('Last-year min/max (non-nan):', np.nanmin(last), np.nanmax(last))
    return {'method': method, 'years': len(years), 'shape': B.shape, 'nan_counts': nan_counts.tolist(), 'last': np.nan_to_num(last).tolist(), 'B': B}

# Compare to R reference for AB and RK4
r_ab = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_ab.csv')
r_rk4 = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_rk4.csv')

res_ab = summarize_run('AB', range(1, 101))
res_rk4 = summarize_run('RK4', range(1, 101))

print('\nR AB shape (rows, cols):', r_ab.shape)
print('R RK4 shape (rows, cols):', r_rk4.shape)

# check group count parity
r_groups_ab = r_ab.columns[1:].tolist()
print('R AB groups:', len(r_groups_ab))
print('Model groups:', len(rpath_model.Group))

# Per-group correlation checks
import math

corrs = {}
for col_idx, gname in enumerate(r_groups_ab):
    rvals = r_ab[gname].values
    pvals = res_ab['B'][:, col_idx]
    # trim to min length
    nmin = min(len(rvals), len(pvals))
    rtrim = rvals[:nmin]
    ptrim = pvals[:nmin]
    # handle NaNs
    if np.all(np.isnan(ptrim)) or np.all(np.isnan(rtrim)):
        corr = float('nan')
    else:
        valid = ~np.isnan(ptrim) & ~np.isnan(rtrim)
        if np.sum(valid) < 2:
            corr = float('nan')
        else:
            try:
                corr = float(np.corrcoef(rtrim[valid], ptrim[valid])[0, 1])
            except Exception:
                corr = float('nan')
    corrs[gname] = corr

    # sort by NaN flag then by correlation value (lowest first)
    # avoid complex one-liners to keep debugging simple
items = list(corrs.items())
items_sorted = sorted(items, key=lambda x: (math.isnan(x[1]), x[1] if not math.isnan(x[1]) else 0))
low = items_sorted[:10]
print('\nLowest correlations (first 10):')
for g, c in low:
    print(g, c)

# Save results for investigation
out = {
    'res_ab': {'shape': res_ab['shape'], 'nan_counts': res_ab['nan_counts']},
    'res_rk4': {'shape': res_rk4['shape'], 'nan_counts': res_rk4['nan_counts']},
    'r_ab_shape': r_ab.shape,
    'r_rk4_shape': r_rk4.shape,
    'group_correlations': corrs,
}
json.dump(out, open('tests/data/rpath_reference/ecosim/diagnose_ecosim_summary.json', 'w'), indent=2)
print('\nSaved diagnostics to tests/data/rpath_reference/ecosim/diagnose_ecosim_summary.json')
