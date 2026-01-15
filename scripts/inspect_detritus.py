"""
Print detritus input and consumption values from diagnostics and compare to R ref.
"""
import json
from pathlib import Path

import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params

ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')

rref = json.load(open(ECOPATH_DIR / 'balanced_model.json'))
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')

params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df.copy()
params.diet = diet_df
model, diag = rpath(params, debug=True)

# det inputs/cons
detinputs = diag['detinputs']
detcons = diag['detcons']
# dead_idx order
types = model_df['Type'].values.astype(float)
dead_idx = [i for i, t in enumerate(types) if t == 2]

d = []
for i, det_global in enumerate(dead_idx):
    d.append({'idx': det_global, 'name': model_df.loc[det_global,'Group'], 'detinput': float(detinputs[i]), 'detcons': float(detcons[i]), 'ee': float(model.EE[det_global]), 'r_ee': float(rref['EE'][det_global])})

print('Detritus diagnostics:', d)
json.dump(d, open(ECOPATH_DIR / 'detritus_diag.json', 'w'), indent=2)
print('Saved detritus_diag.json')
