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

rpath_df = pd.read_csv(RE/'ecosim'/'biomass_trajectory_rk4.csv')

discard_idx = scenario.params.spname.index('Discards')
print('Discard index', discard_idx)

print('PyPath Discards (months around 1..5):')
for m in range(1,6):
    print(m, py_biom[m][discard_idx])

print('\nPyPath Discards (last 5 months):')
for m in range(1196,1201):
    print(m, py_biom[m][discard_idx])

print('\nRpath Discards (first 5 months):')
for m in range(1,6):
    print(m, float(rpath_df.iloc[m]['Discards']))

print('\nRpath Discards (last 5 months):')
for m in range(1196,1201):
    print(m, float(rpath_df.iloc[m]['Discards']))
