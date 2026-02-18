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
py_biom = out.out_Biomass
month = 1199
prey_idx = scenario.params.spname.index('Discards')
print('month',month,'prey',prey_idx,'Bpy',py_biom[month][prey_idx],'Bbase',scenario.params.B_BaseRef[prey_idx])
print('Seabirds index',scenario.params.spname.index('Seabirds'),'Seabirds Bpy',py_biom[month][scenario.params.spname.index('Seabirds')],'Bbase_seabirds',scenario.params.B_BaseRef[scenario.params.spname.index('Seabirds')])
