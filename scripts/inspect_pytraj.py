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
out = rsim_run(scenario, method='RK4', years=range(1, 101))
pytraj = out.out_Biomass
try:
    header = out.out_headers
except Exception:
    header = None
print('pypath_model attrs sample=', [a for a in dir(pypath_model) if 'sp' in a or 'name' in a][:20])
print('scenario attrs sample=', [a for a in dir(scenario) if 'sp' in a or 'name' in a][:20])
print('out attrs sample=', [a for a in dir(out) if 'out' in a or 'names' in a or 'header' in a or 'sp' in a][:40])
print('header sample=', header[:10] if header else None)
print('pytraj shape=', pytraj.shape)
print('pytraj first row sample=', pytraj[0,:10])
print('\nfirst 20 groups in model df:')
for i,name in enumerate(model_df['Group'].tolist()[:20]):
    print(i+1, name)
