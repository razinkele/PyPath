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

rpath_traj = pd.read_csv(RE/'ecosim'/'biomass_trajectory_rk4.csv')
names = scenario.params.spname

# aggregate monthly to annual means
n_groups = len(names)
ann_p = {}
ann_r = {}
for i,name in enumerate(names):
    if name not in rpath_traj.columns:
        # skip gear groups or other non-overlapping names
        continue
    series = biom[:, i]
    n_full_years = len(series)//12
    if n_full_years==0:
        continue
    p = series[:n_full_years*12].reshape(n_full_years, 12).mean(axis=1)
    r = rpath_traj[name].values[:n_full_years*12].reshape(n_full_years, 12).mean(axis=1)
    ann_p[name] = p
    ann_r[name] = r

# compute correlations and rel_err for each species
corrs = []
for name in ann_p.keys():
    p = ann_p[name]
    r = ann_r[name]
    L = min(len(p), len(r))
    if np.std(r[:L]) < 1e-12 or np.std(p[:L]) < 1e-12:
        corr = 1.0 if np.allclose(r[:L], p[:L], atol=1e-12) else np.nan
    else:
        corr = np.corrcoef(r[:L], p[:L])[0,1]
    rel_err = abs(p[-1]-r[-1])/(r[-1] if r[-1]>0 else 1.0)
    corrs.append((name,corr,rel_err, p[-1], r[-1]))

corrs_sorted = sorted(corrs, key=lambda x: (0 if np.isnan(x[1]) else x[1]))
print('Lowest correlations:')
for name,corr,rel_err, p_last, r_last in corrs_sorted[:10]:
    print(f'{name:20s} corr={corr:.6f} rel_err={rel_err:.4f} p_last={p_last:.6f} r_last={r_last:.6f}')

print('\nHighest relative errors:')
corrs_sorted_err = sorted(corrs, key=lambda x: x[2], reverse=True)
for name,corr,rel_err, p_last, r_last in corrs_sorted_err[:10]:
    print(f'{name:20s} rel_err={rel_err:.4f} corr={corr:.6f} p_last={p_last:.6f} r_last={r_last:.6f}')
