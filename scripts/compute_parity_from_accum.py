"""Compute parity metrics given an accumulated yearly results CSV.

Usage: python scripts/compute_parity_from_accum.py --report-top 10
"""
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--accum', default='tmp_parity_results_accum.csv')
parser.add_argument('--rpath', default='tests/data/rpath_reference/ecosim/biomass_trajectory_rk4.csv')
parser.add_argument('--report-top', type=int, default=10)
args = parser.parse_args()

accum = pd.read_csv(args.accum)
rpath = pd.read_csv(args.rpath)

completed_years = sorted(accum['year'].unique())
ny = len(completed_years)

min_months = min(ny*12, rpath.shape[0])
r_df_trim = rpath.iloc[:min_months]
r_yearly = r_df_trim.groupby(r_df_trim.index // 12).mean().iloc[:ny]

results = []
for gname in [c for c in accum.columns if c != 'year']:
    if gname not in r_yearly:
        continue
    r_series = r_yearly[gname].values[:ny]
    p_series = accum.sort_values('year')[gname].values[:ny]
    if np.all(np.isnan(r_series)) or np.all(np.isnan(p_series)):
        corr = np.nan
    else:
        corr = np.corrcoef(r_series, p_series)[0,1]
    denom = np.maximum(np.abs(r_series), 1e-12)
    rel_errs = np.abs(p_series - r_series) / denom
    max_rel = np.nanmax(rel_errs) if rel_errs.size else np.nan
    mean_rel = np.nanmean(rel_errs) if rel_errs.size else np.nan
    results.append((gname, corr, max_rel, mean_rel))

par_df = pd.DataFrame(results, columns=['group','corr','max_rel','mean_rel']).sort_values('corr')
par_df.to_csv('tmp_parity_final_report.csv', index=False)

print('\n=== Seabirds parity summary ===')
if 'Seabirds' in par_df['group'].values:
    print(par_df[par_df['group']=='Seabirds'].to_string(index=False))
else:
    print('Seabirds not found')

print(f"\nTop {args.report_top} worst-correlated groups:")
print(par_df.head(args.report_top).to_string(index=False))

n_bad = ((par_df['corr'] < 0.99) | (par_df['max_rel'] > 0.05)).sum()
print(f"\nGroups failing parity (corr<0.99 or max_rel>0.05): {n_bad} / {len(par_df)}")
print('\nDone')
