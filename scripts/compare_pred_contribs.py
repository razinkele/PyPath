import json
from pathlib import Path
import numpy as np
import pandas as pd

ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
rref = json.load(open(ECOPATH_DIR / 'balanced_model.json'))
groups = model_df['Group'].tolist()
ngroups = len(groups)
types = model_df['Type'].values.astype(float)
living_idx = np.where(types < 2)[0]

pred_cols = [g for g in groups if g in diet_df.columns]
diet_values = np.zeros((ngroups + 1, len(pred_cols)))
prey_to_row = {name: i for i, name in enumerate(diet_df['Group'].tolist())}
for new_row_idx, group_name in enumerate(groups):
    if group_name in prey_to_row:
        old_row = prey_to_row[group_name]
        diet_values[new_row_idx, :] = diet_df.loc[old_row, pred_cols].values.astype(float)
if 'Import' in prey_to_row and len(diet_df['Group']) > ngroups:
    diet_values[ngroups, :] = diet_df.loc[prey_to_row['Import'], pred_cols].values.astype(float)

diet_values = np.nan_to_num(diet_values, nan=0.0)
for col_idx, grp_idx in enumerate(living_idx):
    if 0 < types[grp_idx] < 1:
        mix_q = 1 - types[grp_idx]
        diet_values[:, col_idx] *= mix_q

nodetrdiet = np.zeros((len(living_idx), len(living_idx)))
for j, pred_global in enumerate(living_idx):
    pred_name = groups[pred_global]
    col_idx = pred_cols.index(pred_name)
    import_frac = diet_values[ngroups, col_idx] if diet_values.shape[0] > ngroups else 0.0
    denom = (1 - import_frac) if (1 - import_frac) > 0 else 1.0
    for i, prey_global in enumerate(living_idx):
        nodetrdiet[i, j] = diet_values[prey_global, col_idx] / denom

from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
params = create_rpath_params(groups, types.tolist())
params.model = model_df.copy()
params.diet = diet_df
model, diag = rpath(params, debug=True)

B_py = np.array(model.Biomass, dtype=float)[living_idx]
QB_py = np.array(model.QB, dtype=float)[living_idx]
BQB_py = B_py * QB_py

B_r = np.array(rref['Biomass'], dtype=float)[living_idx]
QB_r = np.array(rref['QB'], dtype=float)[living_idx]
BQB_r = B_r * QB_r

idx_global = 3
idx_local = list(living_idx).index(idx_global)
print('Pred contributions using PyPath BQB:')
for j in range(len(living_idx)):
    if nodetrdiet[idx_local,j] > 0:
        print(j, groups[living_idx[j]], 'frac', nodetrdiet[idx_local,j], 'BQB_py', BQB_py[j], 'contrib', nodetrdiet[idx_local,j]*BQB_py[j])
print('Sum:', (nodetrdiet[idx_local,:]*BQB_py).sum())
print('\nPred contributions using rref BQB:')
for j in range(len(living_idx)):
    if nodetrdiet[idx_local,j] > 0:
        print(j, groups[living_idx[j]], 'frac', nodetrdiet[idx_local,j], 'BQB_r', BQB_r[j], 'contrib', nodetrdiet[idx_local,j]*BQB_r[j])
print('Sum:', (nodetrdiet[idx_local,:]*BQB_r).sum())
