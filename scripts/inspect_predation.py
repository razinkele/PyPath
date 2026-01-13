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

nliving = len([t for t in types if t<2])
# show predators bio_qb and their diet column for predators affecting prey index 3 (JuvRoundfish1)
prey_idx = 3
print('Prey:', groups[prey_idx])
print('b_vec[prey]:', diag['b_vec'][prey_idx])

# For each predator j, show qb, biomass, bio_qb and nodetrdiet at prey index
for j_local, j_global in enumerate(diag['living_idx']):
    qb_val = diag['qb'][j_global]
    biomass_val = diag['biomass_after'][j_global]
    # Safely compute bio_qb, treating NaN as zero (non-consuming predators)
    bio_qb = np.nan_to_num(biomass_val * qb_val, nan=0.0)
    diet_frac = diag['nodetrdiet'][prey_idx, j_local]
    contribution = diet_frac * bio_qb
    if contribution != 0 and not np.isnan(contribution):
        print(f'predator idx {j_global} ({groups[j_global]}): qb={qb_val}, biomass={biomass_val}, bio_qb={bio_qb}, diet_frac={diet_frac}, contrib={contribution}')

sum_contribs = np.nan_to_num(diag['nodetrdiet'][prey_idx,:] * (np.nan_to_num(diag['biomass_after'][diag['living_idx']]) * np.nan_to_num(diag['qb'][diag['living_idx']])))
print('\nSum of contributions:', np.sum(sum_contribs) )
