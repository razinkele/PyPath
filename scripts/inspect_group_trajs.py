from pathlib import Path

import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

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

for name in ['Foragefish1','Foragefish2','Discards','Seabirds','Seals']:
    if name not in scenario.params.spname or name not in rpath_traj.columns:
        continue
    idx = scenario.params.spname.index(name)
    series = biom[:, idx]
    n_full_years = len(series)//12
    p = series[:n_full_years*12].reshape(n_full_years,12).mean(axis=1)
    r = rpath_traj[name].values[:n_full_years*12].reshape(n_full_years,12).mean(axis=1)
    print('\nGroup:', name)
    for i in range(min(30,len(p))):
        print(f'Year {i+1:3d}: p={p[i]:.6f} r={r[i]:.6f} diff={p[i]-r[i]:.6f}')
