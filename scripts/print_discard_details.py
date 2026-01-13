import pandas as pd
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _build_active_link_matrix, _build_link_matrix

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
state_py = py_biom[month]
prey_idx = scenario.params.spname.index('Discards')
seab_idx = scenario.params.spname.index('Seabirds')
print('month',month)
print('Discard Bpy:', state_py[prey_idx])
print('Discard Bbase:', scenario.params.B_BaseRef[prey_idx])
print('Seabirds Bpy:', state_py[seab_idx])
print('Seabirds Bbase:', scenario.params.B_BaseRef[seab_idx])
print('ForcedPrey for Discards (if any):', scenario.forcing.ForcedBio[month][prey_idx] if hasattr(scenario.forcing,'ForcedBio') else 'N/A')
