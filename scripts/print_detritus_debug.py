from pathlib import Path

import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params

REPO = Path(__file__).parent.parent
ECOPATH_DIR = REPO / "tests" / "data" / "rpath_reference" / "ecopath"

model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

groups = model_df["Group"].tolist()
types = model_df["Type"].tolist()

params = create_rpath_params(groups, types)
params.model = model_df
params.diet = diet_df

rpath_obj, diag = rpath(params, debug=True)

print('detinputs:', diag['detinputs'])
print('detcons:', diag['detcons'])
print('det_pb:', diag['det_pb'])
print('det_b:', diag['det_b'])

for d_idx, det_idx in enumerate([i for i,t in enumerate(types) if t==2]):
    print('\nDetritus group', det_idx, groups[det_idx])
    print('  detinputs:', diag['detinputs'][d_idx])
    print('  detcons:', diag['detcons'][d_idx])
    print('  det_pb (computed):', diag['det_pb'][d_idx])
    print('  det_b (computed):', diag['det_b'][d_idx])
    print('  final pb:', rpath_obj.PB[det_idx])
    print('  final biomass:', rpath_obj.Biomass[det_idx])
