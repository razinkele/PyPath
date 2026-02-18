import json
from pathlib import Path

REPO=Path('.')
ECO=REPO/'tests'/'data'/'rpath_reference'/'ecosim'/'ecosim_params.json'
with open(ECO) as fh:
    j=json.load(fh)
print('Keys in R ecosim params:', list(j.keys()))
print('\nR MzeroMort (first 10):', j.get('MzeroMort')[:10])
pp_type = j.get('PP_type')
print('R PP_type present?', pp_type is not None)
if pp_type is not None:
    print('R PP_type (first 10):', pp_type[:10])
print('R ForcedBio present?', 'ForcedBio' in j)
print('\nPreyFrom len', len(j.get('PreyFrom', [])))
print('PreyTo len', len(j.get('PreyTo', [])))
print('QQ len', len(j.get('QQ', [])))

import pandas as pd

model=pd.read_csv(REPO/'tests'/'data'/'rpath_reference'/'ecopath'/'model_params.csv')
print('\nGroups (first 10):', model['Group'].tolist()[:10])

# R arrays are 1-based with a leading zero placeholder; Seabirds is index 1
print('\nR Mzero for Seabirds (index 1):', j.get('MzeroMort')[1])
print('R start_biomass for Seabirds (index 1):', j.get('start_biomass')[1])
print('R start_ftime for Seabirds (index 1):', j.get('start_ftime')[1])

# Now load PyPath scenario and compare
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params

model_df=pd.read_csv(REPO/'tests'/'data'/'rpath_reference'/'ecopath'/'model_params.csv')
diet_df=pd.read_csv(REPO/'tests'/'data'/'rpath_reference'/'ecopath'/'diet_matrix.csv')
params=create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model=model_df
params.diet=diet_df
r=rpath(params)
scenario=rsim_scenario(r, params)
print('\nPy MzeroMort first 10:', scenario.params.MzeroMort[:10])
print('Py PP_type first 10:', scenario.params.PP_type[:10])
print('Py start_state biomass Seabirds:', scenario.start_state.Biomass[1])
# Check if scenario forcing has ForcedBio non-zero for seabirds
f= scenario.forcing.ForcedBio
print('Scenario ForcedBio seabirds (first 12 months):', [float(x[1]) for x in f[:12]])
print('Scenario ForcedBio seabirds (last months sample):', [float(x[1]) for x in f[-12:]])
print('\nDone')
