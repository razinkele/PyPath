"""Minimal parity runner: silent by default, prints concise per-group metrics.

Usage: python scripts/run_parity_minimal.py --years 100 --report-top 10
"""
import os
import argparse
os.environ['PYPATH_SILENCE_DEBUG'] = '1'

import numpy as np
import pandas as pd
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run

RE = Path('tests/data/rpath_reference')
ECOPATH_DIR = RE / 'ecopath'
ECOSIM_DIR = RE / 'ecosim'

parser = argparse.ArgumentParser()
parser.add_argument('--years', type=int, default=100)
parser.add_argument('--report-top', type=int, default=10)
args = parser.parse_args()
ny = args.years

# load reference
rpath_df = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_rk4.csv')
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')

params = create_rpath_params(model_df['Group'].tolist(), [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
py_rpath = rpath(params)
scenario = rsim_scenario(py_rpath, params, years=range(1, ny + 1))

print(f"Running PyPath RK4 for {ny} years (quiet)...")
try:
    out = rsim_run(scenario, method='rk4', years=range(1, ny + 1))
except Exception as e:
    print(f"ERROR: simulation failed: {e}")
    raise

py_monthly = out.out_Biomass
months = py_monthly.shape[0]
n_years = months // 12
py_yearly = py_monthly[: n_years * 12].reshape((n_years, 12, -1)).mean(axis=1)

rpath_months = rpath_df.shape[0]
min_months = min(months, rpath_months)
r_df_trim = rpath_df.iloc[:min_months]
r_yearly = r_df_trim.groupby(r_df_trim.index // 12).mean().iloc[:n_years]

group_names = model_df['Group'].tolist()
num_groups = len(group_names)
results = []
for i, gname in enumerate(group_names, start=1):
    if gname not in r_yearly:
        continue
    r_series = r_yearly[gname].values[:n_years]
    p_series = py_yearly[:, i][:n_years]
    if np.all(np.isnan(r_series)) or np.all(np.isnan(p_series)):
        corr = np.nan
    else:
        corr = np.corrcoef(r_series, p_series)[0, 1]
    denom = np.maximum(np.abs(r_series), 1e-12)
    rel_errs = np.abs(p_series - r_series) / denom
    max_rel = np.nanmax(rel_errs) if rel_errs.size else np.nan
    mean_rel = np.nanmean(rel_errs) if rel_errs.size else np.nan
    results.append((gname, corr, max_rel, mean_rel))

res_df = pd.DataFrame(results, columns=['group', 'corr', 'max_rel', 'mean_rel'])
res_df = res_df.sort_values('corr')

# Save concise report
out_csv = Path('tmp_parity_results.csv')
res_df.to_csv(out_csv, index=False)

# Print summary
sname = 'Seabirds'
row = res_df[res_df['group'] == sname]
print('\n=== Seabirds parity summary ===')
if not row.empty:
    print(row.to_string(index=False))
else:
    print('Seabirds not found in results')

print(f"\nTop {args.report_top} worst-correlated groups:")
print(res_df.head(args.report_top).to_string(index=False))

n_bad = ((res_df['corr'] < 0.99) | (res_df['max_rel'] > 0.05)).sum()
print(f"\nGroups failing parity (corr<0.99 or max_rel>0.05): {n_bad} / {len(res_df)}")
print('\nDone')