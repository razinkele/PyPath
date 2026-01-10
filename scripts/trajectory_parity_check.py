"""Compare PyPath RK4 long-run trajectory to Rpath reference (annualized)

Usage: python scripts/trajectory_parity_check.py --years 100 --report-top 10
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RE = Path('tests/data/rpath_reference')
ECOPATH_DIR = RE / 'ecopath'
ECOSIM_DIR = RE / 'ecosim'

parser = argparse.ArgumentParser()
parser.add_argument('--years', type=int, default=100)
parser.add_argument('--report-top', type=int, default=10)
parser.add_argument('--quiet', action='store_true', help='suppress debug prints during run')
args = parser.parse_args()
ny = args.years

# If requested, signal ecosim module to silence debug prints via env var
if args.quiet:
    import os

    os.environ['PYPATH_SILENCE_DEBUG'] = '1'

# Import pypath components after possible env var set so modules can respect it
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run

# load reference
rpath_df = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_rk4.csv')
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')

# build scenario
params = create_rpath_params(model_df['Group'].tolist(), [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
py_rpath = rpath(params)
scenario = rsim_scenario(py_rpath, params, years=range(1, ny + 1))

# run PyPath RK4
print('Running PyPath RK4 for', ny, 'years...')
out = rsim_run(scenario, method='rk4', years=range(1, ny + 1))
py_monthly = out.out_Biomass  # months x groups+1

# Convert monthly to annual series (mean per year)
months = py_monthly.shape[0]
n_years = months // 12
py_yearly = py_monthly[: n_years * 12].reshape((n_years, 12, -1)).mean(axis=1)  # n_years x groups+1

# Extract Rpath reference monthly -> annual (same approach)
# rpath_df rows correspond to months; ensure length matches or trim
rpath_months = rpath_df.shape[0]
min_months = min(months, rpath_months)
r_df_trim = rpath_df.iloc[:min_months]
r_yearly = r_df_trim.groupby(r_df_trim.index // 12).mean().iloc[:n_years]

# Compute per-group correlation and max relative error
group_names = model_df['Group'].tolist()
num_groups = len(group_names)
results = []
for i, gname in enumerate(group_names, start=1):
    # rpath annual for group
    if gname not in r_yearly:
        continue
    r_series = r_yearly[gname].values[:n_years]
    p_series = py_yearly[:, i][:n_years]
    # correlation
    if np.all(np.isnan(r_series)) or np.all(np.isnan(p_series)):
        corr = np.nan
    else:
        corr = np.corrcoef(r_series, p_series)[0, 1]
    # max relative error
    denom = np.maximum(np.abs(r_series), 1e-12)
    rel_errs = np.abs(p_series - r_series) / denom
    max_rel = np.nanmax(rel_errs) if rel_errs.size else np.nan
    mean_rel = np.nanmean(rel_errs) if rel_errs.size else np.nan
    results.append((gname, corr, max_rel, mean_rel))

res_df = pd.DataFrame(results, columns=['group', 'corr', 'max_rel', 'mean_rel'])
res_df = res_df.sort_values('corr')

# report
# Restore print if we silenced debug prints earlier
if args.quiet:
    import builtins

    builtins.print = _original_print

sname = 'Seabirds'
row = res_df[res_df['group'] == sname]
print('\n=== Seabirds parity summary ===')
if not row.empty:
    print(row.to_string(index=False))
else:
    print('Seabirds not found in results')

# show worst correlations
print(f"\nTop {args.report_top} worst-correlated groups:")
print(res_df.head(args.report_top).to_string(index=False))

# summary pass/fail counts
n_bad = ((res_df['corr'] < 0.99) | (res_df['max_rel'] > 0.05)).sum()
print(f"\nGroups failing parity (corr<0.99 or max_rel>0.05): {n_bad} / {len(res_df)}")

print('\nDone')
