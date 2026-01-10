from pathlib import Path
import numpy as np
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

print("Groups:", groups)
print("Living indices:", diag['living_idx'])

for i, idx in enumerate(diag['living_idx']):
    name = groups[idx]
    before = diag['biomass_before'][idx]
    after = diag['biomass_after'][idx]
    xval = diag['x'][i]
    print(f"Idx {idx} ({name}): before={before}, solved_x={xval}, after={after}")

print('\nDetritus groups:')
for j in range(len(model_df)):
    if types[j] == 2:
        print(j, groups[j], 'input_biomass', model_df['Biomass'].values[j], 'computed_pb', rpath_obj.PB[j], 'computed_biomass', rpath_obj.Biomass[j])
