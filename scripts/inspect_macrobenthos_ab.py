from pathlib import Path

import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

BASE=Path('tests/data/rpath_reference')
ECOPATH_DIR=BASE/'ecopath'
ECOSIM_DIR=BASE/'ecosim'
model_df=pd.read_csv(ECOPATH_DIR/'model_params.csv')
diet_df=pd.read_csv(ECOPATH_DIR/'diet_matrix.csv')
groups=model_df['Group'].tolist()
params=create_rpath_params(groups, model_df['Type'].tolist())
params.model=model_df
params.diet=diet_df
pypath_model=rpath(params)
scenario=rsim_scenario(pypath_model, params, years=range(1,3))
out_ab=rsim_run(scenario, method='AB', years=range(1,3))
rpath_traj_ab=pd.read_csv(ECOSIM_DIR/'biomass_trajectory_ab.csv')
macrob_idx=groups.index('Macrobenthos')
macrob_out_idx=macrob_idx+1
rvals=rpath_traj_ab['Macrobenthos'].values
pvals=out_ab.out_Biomass[:len(rvals), macrob_out_idx]
print('rvals[:8]=', rvals[:8])
print('pvals[:8]=', pvals[:8])
print('diff    =', pvals[:8]-rvals[:8])
print('rel diff=', (pvals[:8]-rvals[:8])/rvals[:8])
