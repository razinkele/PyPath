from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params

ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')

# Load model and diet
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')

groups = model_df['Group'].tolist()
types = model_df['Type'].tolist()

params = create_rpath_params(groups, types)
params.model = model_df
params.diet = diet_df

rpath_obj, diag = rpath(params, debug=True)
iterations = diag.get('iterations', [])
print(f'Found {len(iterations)} iterations')
outdir = ECOPATH_DIR
for it in iterations:
    i = it['iter']
    np.save(outdir / f'iter_{i}_A.npy', it['A'])
    np.save(outdir / f'iter_{i}_bvec.npy', it['b_vec'])
    # Also save x/ee/biomass
    np.save(outdir / f'iter_{i}_x.npy', it['x'])
    np.save(outdir / f'iter_{i}_ee.npy', it['ee'])
    np.save(outdir / f'iter_{i}_biomass.npy', it['biomass'])
print('Iteration dumps saved to', outdir)
