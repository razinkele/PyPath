import json
from pathlib import Path

import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params

REFERENCE_DIR = Path("tests/data/rpath_reference")
ECOPATH_DIR = REFERENCE_DIR / "ecopath"

model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

groups = model_df["Group"].tolist()
types = model_df["Type"].tolist()
params = create_rpath_params(groups, types)
params.model = model_df
params.diet = diet_df

model = rpath(params)

print(model)
print("\nFirst 10 groups summary:\n", model.summary().head(10))

# Load reference
with open(ECOPATH_DIR / "balanced_model.json","r") as f:
    ref = json.load(f)

for i, grp in enumerate(ref['Group'][:20]):
    print(i, grp, 'ref_B=', ref['Biomass'][i], 'py_B=', model.Biomass[i],
          'ref_pb=', ref['PB'][i], 'py_pb=', model.PB[i], 'ref_ee=', ref['EE'][i], 'py_ee=', model.EE[i])
