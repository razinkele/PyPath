"""
Inspect b_vec components (consumption, catch, bioacc) for comparison between PyPath run and R reference.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath

ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')

rref = json.load(open(ECOPATH_DIR / 'balanced_model.json'))
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')

groups = model_df['Group'].tolist()
ngroups = len(groups)
types = model_df['Type'].values.astype(float)
living_idx = np.where(types < 2)[0]

# build nodetrdiet same as other scripts
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

params = create_rpath_params(groups, types.tolist())
params.model = model_df.copy()
params.diet = diet_df
model, diag = rpath(params, debug=True)

# PyPath B, QB for living idx
B_py = np.array(model.Biomass, dtype=float)[living_idx]
QB_py = np.array(model.QB, dtype=float)[living_idx]
cons_py = nodetrdiet * (B_py * QB_py)[np.newaxis, :]
cons_py_sum = np.sum(cons_py, axis=1)

# landings/discards
fleet_idx = np.where(types == 3)[0]
landmat = np.zeros((ngroups, len(fleet_idx)))
discardmat = np.zeros((ngroups, len(fleet_idx)))
if len(fleet_idx) > 0:
    fleet_names = [groups[i] for i in fleet_idx]
    for g, name in enumerate(fleet_names):
        if name in model_df.columns:
            landmat[:, g] = model_df[name].values.astype(float)
        disc = f"{name}.disc"
        if disc in model_df.columns:
            discardmat[:, g] = model_df[disc].values.astype(float)
landmat = np.nan_to_num(landmat, nan=0.0)
discardmat = np.nan_to_num(discardmat, nan=0.0)
catch_py = np.sum(landmat + discardmat, axis=1)[living_idx]

bioacc_py = np.nan_to_num(model_df['BioAcc'].values.astype(float))[living_idx]

b_vec_py = catch_py + bioacc_py + cons_py_sum

# R ref components
B_r = np.array(rref['Biomass'], dtype=float)[living_idx]
QB_r = np.array(rref['QB'], dtype=float)[living_idx]
cons_r = nodetrdiet * (B_r * QB_r)[np.newaxis, :]
cons_r_sum = np.sum(cons_r, axis=1)
catch_r = np.sum(landmat + discardmat, axis=1)[living_idx]
# bioacc was taken from model_df earlier
b_vec_r = catch_r + np.nan_to_num(model_df['BioAcc'].values.astype(float))[living_idx] + cons_r_sum

# Print components for group 3 (global index 3)
idx_global = 3
if idx_global in living_idx:
    idx_local = list(living_idx).index(idx_global)
    print('Group', groups[idx_global], 'local idx', idx_local)
    print('PyPath: B=', B_py[idx_local], 'QB=', QB_py[idx_local], 'cons=', cons_py_sum[idx_local], 'catch=', catch_py[idx_local], 'bioacc=', bioacc_py[idx_local], 'b_vec=', b_vec_py[idx_local], 'EE=', model.EE[idx_global])
    print('Rref:   B=', B_r[idx_local], 'QB=', QB_r[idx_local], 'cons=', cons_r_sum[idx_local], 'catch=', catch_r[idx_local], 'bioacc=', model_df.loc[idx_global,'BioAcc'], 'b_vec=', b_vec_r[idx_local], 'EE=', rref['EE'][idx_global])
else:
    print('Group 3 is not a living group.')

# Save summary
out = {
    'group': groups[idx_global],
    'py': {
        'B': float(B_py[idx_local]), 'QB': float(QB_py[idx_local]), 'cons': float(cons_py_sum[idx_local]), 'catch': float(catch_py[idx_local]), 'bioacc': float(bioacc_py[idx_local]), 'b_vec': float(b_vec_py[idx_local]), 'EE': float(model.EE[idx_global])
    },
    'r': {
        'B': float(B_r[idx_local]), 'QB': float(QB_r[idx_local]), 'cons': float(cons_r_sum[idx_local]), 'catch': float(catch_r[idx_local]), 'bioacc': float(model_df.loc[idx_global,'BioAcc']), 'b_vec': float(b_vec_r[idx_local]), 'EE': float(rref['EE'][idx_global])
    }
}
import json
json.dump(out, open(ECOPATH_DIR / 'bvec_component_group3.json', 'w'), indent=2)
print('Saved bvec components to', ECOPATH_DIR / 'bvec_component_group3.json')
