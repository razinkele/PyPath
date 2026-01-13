import numpy as np
import pandas as pd
from pathlib import Path
ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')

# load R ref
import json
rref = json.load(open(ECOPATH_DIR / 'balanced_model.json'))
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')

groups = model_df['Group'].tolist()
ngroups = len(groups)
types = model_df['Type'].values.astype(float)
living_idx = np.where(types < 2)[0]
nliving = len(living_idx)

# build nodetrdiet as in compare_intermediates
pred_cols = [g for g in groups if g in diet_df.columns]
diet_values = np.zeros((ngroups+1, len(pred_cols)))
prey_to_row = {name:i for i,name in enumerate(diet_df['Group'].tolist())}
for new_row, group_name in enumerate(groups):
    if group_name in prey_to_row:
        old_row = prey_to_row[group_name]
        diet_values[new_row,:] = diet_df.loc[old_row, pred_cols].values.astype(float)
if 'Import' in prey_to_row and len(diet_df['Group']) > ngroups:
    diet_values[ngroups,:] = diet_df.loc[prey_to_row['Import'], pred_cols].values.astype(float)

diet_values = np.nan_to_num(diet_values, nan=0.0)
for col_idx, grp_idx in enumerate(living_idx):
    if 0 < types[grp_idx] < 1:
        mix_q = 1 - types[grp_idx]
        diet_values[:, col_idx] *= mix_q

nodetrdiet = np.zeros((nliving, nliving))
for j, pred_global in enumerate(living_idx):
    pred_name = groups[pred_global]
    col_idx = pred_cols.index(pred_name)
    import_frac = diet_values[ngroups, col_idx] if diet_values.shape[0] > ngroups else 0.0
    denom = (1 - import_frac) if (1 - import_frac) > 0 else 1.0
    for i, prey_global in enumerate(living_idx):
        nodetrdiet[i,j] = diet_values[prey_global, col_idx] / denom

B = np.array(rref['Biomass'], dtype=float)
PB = np.array(rref['PB'], dtype=float)
QB = np.array(rref['QB'], dtype=float)

bio_qb = np.nan_to_num(B[living_idx] * QB[living_idx], nan=0.0)
cons = nodetrdiet * bio_qb[np.newaxis, :]

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

living_catch = np.sum(landmat + discardmat, axis=1)[living_idx]

b_vec_ref = living_catch + np.nan_to_num(model_df['BioAcc'].values.astype(float))[living_idx] + np.sum(cons, axis=1)

# Build A_ref
original_no_b = np.isnan(model_df['Biomass'].values.astype(float))
original_no_ee = np.isnan(model_df['EE'].values.astype(float))
A_ref = np.zeros((nliving, nliving))
for i in range(nliving):
    idx_global = living_idx[i]
    if original_no_ee[idx_global]:
        if not np.isnan(B[idx_global]):
            A_ref[i,i] = B[idx_global] * PB[idx_global]
        else:
            A_ref[i,i] = PB[idx_global] * rref['EE'][idx_global]
    else:
        A_ref[i,i] = PB[idx_global] * rref['EE'][idx_global]
qb_dc = nodetrdiet * np.nan_to_num(QB[living_idx])[np.newaxis, :]
for j in range(nliving):
    if original_no_b[living_idx[j]]:
        A_ref[:, j] -= qb_dc[:, j]

# load iteration 1
A_it1 = np.load(ECOPATH_DIR / 'iter_1_A.npy')
b_it1 = np.load(ECOPATH_DIR / 'iter_1_bvec.npy')

absA = np.abs(A_ref - A_it1)
print('Iter1 A: max abs diff', absA.max())

diff_b = np.abs(b_vec_ref - b_it1)
print('Iter1 b: max abs diff', diff_b.max())

# show largest entries diffs
import numpy as np
idxs = np.unravel_index(np.argsort(absA.ravel())[::-1][:10], absA.shape)
for i,j in zip(idxs[0], idxs[1]):
    print('A diff at', (i,j), 'ref', A_ref[i,j], 'it1', A_it1[i,j], 'absdiff', absA[i,j])

idxs_b = np.argsort(diff_b)[::-1][:10]
for i in idxs_b:
    print('b diff at', i, 'ref', b_vec_ref[i], 'it1', b_it1[i], 'absdiff', diff_b[i])
