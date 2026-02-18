from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params

ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
groups = model_df['Group'].tolist()
ngroups = len(groups)
types = model_df['Type'].values.astype(float)
living_idx = np.where(types < 2)[0]
params = create_rpath_params(groups, types.tolist())
params.model = model_df.copy()
params.diet = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
model, diag = rpath(params, debug=True)
B_py = np.array(model.Biomass, dtype=float)[living_idx]
QB_py = np.array(model.QB, dtype=float)[living_idx]
print('living_idx', living_idx)
for i, idx in enumerate(living_idx):
    print(i, idx, groups[idx], 'B=', B_py[i], 'QB=', QB_py[i], 'B*QB=', B_py[i]*QB_py[i])
