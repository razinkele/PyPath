import pandas as pd, numpy as np
from pypath.core.ecosim import rsim_scenario, rsim_run
from pypath.core.params import create_rpath_params
from pathlib import Path

REPO=Path(__file__).parent.parent
ECOPATH_DIR = REPO / 'tests' / 'data' / 'rpath_reference' / 'ecopath'
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df

from pypath.core.ecopath import rpath
r = rpath(params)
from pypath.core.ecosim import rsim_scenario
s = rsim_scenario(r, params)
out = rsim_run(s, method='RK4', years=range(1,101))
ref = pd.read_csv(REPO / 'tests' / 'data' / 'rpath_reference' / 'ecosim' / 'biomass_trajectory_rk4.csv')
col='Seabirds'; idx=list(ref.columns[1:]).index(col)
py = out.out_Biomass[:,idx]
r = ref[col].values[:len(py)]
print('len', len(py))
print('py first 20', py[:20].round(6))
print('r first 20', r[:20].round(6))
print('py last 20', py[-20:].round(6))
print('r last 20', r[-20:].round(6))
print('corr', numpy.corrcoef(r, py)[0,1])
print('py min/max', py.min(), py.max())
