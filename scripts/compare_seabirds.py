import pandas as pd
import numpy as np
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run

RE = Path('tests/data/rpath_reference')
model_df = pd.read_csv(RE/'ecopath'/'model_params.csv')
diet_df = pd.read_csv(RE/'ecopath'/'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = diet_df
pypath_model = rpath(params)
scenario = rsim_scenario(pypath_model, params, years=range(1,101))
out = rsim_run(scenario, method='rk4', years=range(1,101))
biom = out.out_Biomass
if 'Seabirds' not in scenario.params.spname:
    print('No Seabirds in test data');
    raise SystemExit(0)
sidx = scenario.params.spname.index('Seabirds')
biom_series = biom[:, sidx]

# aggregate to annual by taking monthly means per year (12-month blocks)
n_full_years = len(biom_series) // 12
p_pYear = biom_series[: n_full_years * 12].reshape(n_full_years, 12).mean(axis=1)

rpath_traj = pd.read_csv(RE/'ecosim'/'biomass_trajectory_rk4.csv')
r_pSeab = rpath_traj['Seabirds'].values
r_pYear = r_pSeab[: p_pYear.shape[0] * 12].reshape(-1, 12).mean(axis=1)

L = min(len(r_pYear), len(p_pYear))
if np.std(r_pYear[:L]) < 1e-12 or np.std(p_pYear[:L]) < 1e-12:
    corr = 1.0 if np.allclose(r_pYear[:L], p_pYear[:L], atol=1e-12) else np.nan
else:
    corr = np.corrcoef(r_pYear[:L], p_pYear[:L])[0,1]
rel_err = abs(p_pYear[-1] - r_pYear[-1]) / (r_pYear[-1] if r_pYear[-1] > 0 else 1.0)
print(f'Correlation={corr:.6f} rel_err={rel_err:.6f}')
# find years with largest absolute differences
absdiff = np.abs(p_pYear[:L] - r_pYear[:L])
max_idx = int(np.argmax(absdiff))
print('Year with max abs diff:', max_idx+1, 'absdiff=', absdiff[max_idx], 'p=', p_pYear[max_idx], 'r=', r_pYear[max_idx])
# print first and last 20 years for inspection
print('\nFirst 20 years (p vs r):')
for i in range(min(20, L)):
    print(f'Year {i+1:3d}: p={p_pYear[i]:.6f} r={r_pYear[i]:.6f} diff={p_pYear[i]-r_pYear[i]:.6f}')
print('\nLast 20 years (p vs r):')
for i in range(max(0, L-20), L):
    print(f'Year {i+1:3d}: p={p_pYear[i]:.6f} r={r_pYear[i]:.6f} diff={p_pYear[i]-r_pYear[i]:.6f}')
