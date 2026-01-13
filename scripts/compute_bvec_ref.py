import json
from pathlib import Path

import numpy as np
import pandas as pd

ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')
ref=json.load(open(ECOPATH_DIR / 'balanced_model.json'))
df=pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
# predators are living group columns in diet
groups = ref['Group']
diet_df = df
prey_to_row = {r:i for i,r in enumerate(diet_df['Group'].tolist())}
cols = [g for g in groups if g in diet_df.columns]
diet_values = np.zeros((len(groups)+1, len(cols)))
for new_row, group in enumerate(groups):
    if group in prey_to_row:
        diet_values[new_row,:] = diet_df.loc[prey_to_row[group], cols].values
if 'Import' in prey_to_row and len(diet_df['Group']) > len(groups):
    diet_values[len(groups), :] = diet_df.loc[prey_to_row['Import'], cols].values
# Replace NA/NaN with 0 for missing diet entries
diet_values = np.nan_to_num(diet_values, nan=0.0)
import_row = diet_values[len(groups), :]
nliving = 20
# normalize
nodetrdiet = np.zeros((nliving, nliving))
for j in range(nliving):
    denom = 1.0 - import_row[j] if (1.0 - import_row[j]) > 0 else 1.0
    for i in range(nliving):
        nodetrdiet[i,j] = diet_values[i,j] / denom
B = np.array(ref['Biomass'])
QB = np.array(ref['QB'])
bio_qb = B * QB
b_vec_ref = np.zeros(nliving)
for i in range(nliving):
    b_vec_ref[i] = np.sum(nodetrdiet[i,:] * bio_qb[:nliving])
print('b_vec_ref[3]=', b_vec_ref[3])
print('sum of diag nodetrdiet contributions[3]=', np.sum(nodetrdiet[3,:] * bio_qb[:nliving]))
print('Total contributions row 3 details:')
for j in range(nliving):
    if nodetrdiet[3,j] * bio_qb[j] != 0:
        print(j, cols[j], 'diet_frac', nodetrdiet[3,j], 'bio_qb', bio_qb[j], 'contrib', nodetrdiet[3,j]*bio_qb[j])
print('\nDone')
