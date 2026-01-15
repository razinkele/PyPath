from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
DIAG_DIR = REPO / 'tests' / 'data' / 'rpath_reference' / 'ecosim' / 'diagnostics'
RDIR = REPO / 'tests' / 'data' / 'rpath_reference' / 'ecosim'

# Load Py diagnostics and R reference
rk4 = pd.read_csv(DIAG_DIR / 'seabirds_components_rk4.csv')
ab = pd.read_csv(DIAG_DIR / 'seabirds_components_ab.csv')
qq_rk4 = pd.read_csv(DIAG_DIR / 'seabirds_qq_rk4.csv')
qq_ab = pd.read_csv(DIAG_DIR / 'seabirds_qq_ab.csv')

r_rk4 = pd.read_csv(RDIR / 'biomass_trajectory_rk4.csv')
if 'Seabirds' not in r_rk4.columns:
    raise SystemExit('R RK4 reference missing Seabirds')

r_ab = pd.read_csv(RDIR / 'biomass_trajectory_ab.csv')

# Align lengths
n = len(rk4)
rvals_rk4 = r_rk4['Seabirds'].values[:n]
pyvals_rk4 = rk4['biomass'].values

n2 = len(ab)
rvals_ab = r_ab['Seabirds'].values[:n2]
pyvals_ab = ab['biomass'].values

# Compute correlations with components
components = ['production', 'predation_loss', 'consumption_by_predator', 'fish_loss', 'm0_loss', 'derivative']
print('RK4: correlations between R Seabirds and Py components')
for c in components:
    series = rk4[c].values
    if np.std(series) == 0 or np.std(rvals_rk4) == 0:
        corr = 1.0 if np.allclose(series, rvals_rk4, atol=1e-12) else np.nan
    else:
        corr = np.corrcoef(rvals_rk4, series)[0,1]
    print(f'  {c}: corr={corr:.4f}, mean={np.nanmean(series):.6g}, std={np.nanstd(series):.6g}')

print('\nAB: correlations between R Seabirds and Py components')
for c in components:
    series = ab[c].values
    if np.std(series) == 0 or np.std(rvals_ab) == 0:
        corr = 1.0 if np.allclose(series, rvals_ab, atol=1e-12) else np.nan
    else:
        corr = np.corrcoef(rvals_ab, series)[0,1]
    print(f'  {c}: corr={corr:.4f}, mean={np.nanmean(series):.6g}, std={np.nanstd(series):.6g}')

# Find months with largest biomass difference RK4
diff_rk4 = pyvals_rk4 - rvals_rk4
idx_max = np.argmax(np.abs(diff_rk4))
print(f'\nRK4 largest biomass diff month={rk4.iloc[idx_max].month}, diff={diff_rk4[idx_max]:.6g}')
print('Components at that month:')
print(rk4.iloc[idx_max][components])
print('\nQQ contributions (top 5 prey) at that month:')
qq_row = qq_rk4.iloc[idx_max][1:]
prey_names = qq_rk4.columns[1:]
qqs = list(zip(prey_names, qq_row.values))
qqs_sorted = sorted(qqs, key=lambda t: abs(t[1]), reverse=True)[:5]
for name, val in qqs_sorted:
    print(f'  {name}: {val:.6g}')

# Same for AB
diff_ab = pyvals_ab - rvals_ab
idx_max_ab = np.argmax(np.abs(diff_ab))
print(f'\nAB largest biomass diff month={ab.iloc[idx_max_ab].month}, diff={diff_ab[idx_max_ab]:.6g}')
print('Components at that month:')
print(ab.iloc[idx_max_ab][components])
print('\nQQ contributions (top 5 prey) at that month:')
qq_row_ab = qq_ab.iloc[idx_max_ab][1:]
prey_names_ab = qq_ab.columns[1:]
qqs_ab = list(zip(prey_names_ab, qq_row_ab.values))
qqs_ab_sorted = sorted(qqs_ab, key=lambda t: abs(t[1]), reverse=True)[:5]
for name, val in qqs_ab_sorted:
    print(f'  {name}: {val:.6g}')

# Save a brief summary JSON
summary = {
    'rk4': {
        'corr_components': {c: float(np.corrcoef(rvals_rk4, rk4[c].values)[0,1]) if np.std(rk4[c].values) > 0 else None for c in components},
        'max_diff_month': int(rk4.iloc[idx_max].month),
        'max_diff': float(diff_rk4[idx_max]),
    },
    'ab': {
        'corr_components': {c: float(np.corrcoef(rvals_ab, ab[c].values)[0,1]) if np.std(ab[c].values) > 0 else None for c in components},
        'max_diff_month': int(ab.iloc[idx_max_ab].month),
        'max_diff': float(diff_ab[idx_max_ab]),
    },
}
import json

with open(DIAG_DIR / 'seabirds_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
print('\nSaved summary to', DIAG_DIR / 'seabirds_summary.json')
