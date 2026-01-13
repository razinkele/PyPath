from pathlib import Path
import pandas as pd
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
RE = Path('tests/data/rpath_reference')
model_df = pd.read_csv(RE / 'ecopath' / 'model_params.csv')
params = create_rpath_params(model_df['Group'].tolist(), [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
pypath_model = rpath(params)
print('NoIntegrate length:', len(pypath_model.params.NoIntegrate))
print('NoIntegrate sample (first 30):', pypath_model.params.NoIntegrate[:30])
print('Index of Discards:', pypath_model.params.spname.index('Discards'))
print('NoIntegrate[Discards]=', pypath_model.params.NoIntegrate[pypath_model.params.spname.index('Discards')])