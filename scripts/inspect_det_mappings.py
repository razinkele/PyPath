from pathlib import Path

import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params

REF = Path('tests/data/rpath_reference')
ECOPATH_DIR = REF / 'ecopath'
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')

groups = model_df['Group'].tolist()
params = create_rpath_params(groups, [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = diet_df

model = rpath(params)
scenario = rsim_scenario(model, params, years=range(1, 101))
print('Model row for Discards:')
print(model_df[model_df['Group']=='Discards'])

print('DetFrac pre-run shape:', getattr(scenario.params, 'DetFrac', None))
print('DetFrom sample:', getattr(scenario.params, 'DetFrom', None)[:50])
print('DetTo sample:', getattr(scenario.params, 'DetTo', None)[:50])
print('NumDetLinks:', getattr(scenario.params, 'NumDetLinks', None))
# show triples where DetTo == 2 (detritus index 2)
det_from = getattr(scenario.params, 'DetFrom', None)
det_to = getattr(scenario.params, 'DetTo', None)
det_frac = getattr(scenario.params, 'DetFrac', None)
if det_from is not None and det_to is not None and det_frac is not None:
    for k, (f, t, fr) in enumerate(zip(det_from, det_to, det_frac)):
        if t == 2:
            print('link', k, 'from', f, 'to', t, 'frac', fr)
