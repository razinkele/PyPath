import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

model_df = pd.read_csv('tests/data/rpath_reference/ecopath/model_params.csv')
diet_df = pd.read_csv('tests/data/rpath_reference/ecopath/diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df
pypath_model = rpath(params)
scenario = rsim_scenario(pypath_model, params, years=range(1, 101))
rpath_traj = pd.read_csv('tests/data/rpath_reference/ecosim/biomass_trajectory_rk4.csv')

out = rsim_run(scenario, method='RK4', years=range(1, 101))
pytraj = out.out_Biomass
seabirds_col_py = 1
r_vals = rpath_traj['Seabirds'].values
p_vals = pytraj[:len(r_vals), seabirds_col_py]
print('len r', len(r_vals), 'len p', len(p_vals))
print('corr', np.corrcoef(r_vals, p_vals)[0,1])
print('R first 10:', r_vals[:10])
print('PY first 10:', p_vals[:10])
print('R last 10:', r_vals[-10:])
print('PY last 10:', p_vals[-10:])
