"""
Greedy multi-flip heuristic: starting from original model, iteratively pick the single flip (cap EE=1, set biomass NaN)
that most reduces A/b distance to R reference, apply it, and repeat until no improvement or until max_flips.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params

ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')

rref = json.load(open(ECOPATH_DIR / 'balanced_model.json'))
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')

groups = model_df['Group'].tolist()
ngroups = len(groups)
types = model_df['Type'].values.astype(float)
living_idx = np.where(types < 2)[0]
nliving = len(living_idx)

# Compute R-like A_ref and b_vec_ref (reuse compute from compare scripts)
# Simplified: we'll load A_ref and b_vec_ref using the helpers in compare_intermediates if available
# For brevity, reuse the approach from heuristic_flip_relaxed (copying necessary parts)

# Build nodetrdiet
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

nodetrdiet = np.zeros((nliving, nliving))
for j, pred_global in enumerate(living_idx):
    pred_name = groups[pred_global]
    col_idx = pred_cols.index(pred_name)
    import_frac = diet_values[ngroups, col_idx] if diet_values.shape[0] > ngroups else 0.0
    denom = (1 - import_frac) if (1 - import_frac) > 0 else 1.0
    for i, prey_global in enumerate(living_idx):
        nodetrdiet[i, j] = diet_values[prey_global, col_idx] / denom

B = np.array(rref['Biomass'], dtype=float)
PB = np.array(rref['PB'], dtype=float)
QB = np.array(rref['QB'], dtype=float)

# compute b_vec_ref
bio_qb_ref = np.nan_to_num(B[living_idx] * QB[living_idx], nan=0.0)
cons_ref = nodetrdiet * bio_qb_ref[np.newaxis, :]

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

living_catch_ref = np.sum(landmat + discardmat, axis=1)[living_idx]

b_vec_ref = living_catch_ref + np.nan_to_num(model_df['BioAcc'].values.astype(float))[living_idx] + np.sum(cons_ref, axis=1)

# build A_ref
original_no_b = np.isnan(model_df['Biomass'].values.astype(float))
original_no_ee = np.isnan(model_df['EE'].values.astype(float))
A_ref = np.zeros((nliving, nliving))
for i in range(nliving):
    idx_global = living_idx[i]
    if original_no_ee[idx_global]:
        if not np.isnan(B[idx_global]):
            A_ref[i, i] = B[idx_global] * PB[idx_global]
        else:
            A_ref[i, i] = PB[idx_global] * rref['EE'][idx_global]
    else:
        A_ref[i, i] = PB[idx_global] * rref['EE'][idx_global]
qb_dc_ref = nodetrdiet * np.nan_to_num(QB[living_idx])[np.newaxis, :]
for j in range(nliving):
    if original_no_b[living_idx[j]]:
        A_ref[:, j] -= qb_dc_ref[:, j]

# Start from original model
working_df = model_df.copy()
params = create_rpath_params(groups, types.tolist())
params.model = working_df.copy()
params.diet = diet_df
base_model, base_diag = rpath(params, debug=True)
base_A = base_diag.get('iterations', [base_diag])[-1]['A'] if base_diag.get('iterations') else base_diag['A']
base_b = base_diag.get('iterations', [base_diag])[-1]['b_vec'] if base_diag.get('iterations') else base_diag['b_vec']

def score_A_b(A_mod, b_mod):
    score_A = np.nanmax(np.abs(A_ref - A_mod))
    score_b = np.nanmax(np.abs(b_vec_ref - b_mod))
    return score_A + score_b, score_A, score_b

current_score, _, _ = score_A_b(base_A, base_b)

max_flips = 5
applied_flips = []

for flip_num in range(max_flips):
    # identify candidate pool (any living group with EE>1.0 in current state)
    params_tmp = create_rpath_params(groups, types.tolist())
    params_tmp.model = working_df.copy()
    params_tmp.diet = diet_df
    model_tmp, diag_tmp = rpath(params_tmp, debug=True)
    current_ee = model_tmp.EE
    candidates = [int(idx) for idx in living_idx if current_ee[idx] > 1.0 and idx not in applied_flips]
    if not candidates:
        print('No candidates with EE>1 found. Stopping.')
        break
    print(f'Flip round {flip_num+1}, candidates: {candidates}')

    best_local = None
    best_local_score = current_score
    best_local_details = None

    for cand in candidates:
        mod_df = working_df.copy()
        mod_df.loc[mod_df['Group'] == groups[cand], 'EE'] = 1.0
        mod_df.loc[mod_df['Group'] == groups[cand], 'Biomass'] = np.nan
        params_test = create_rpath_params(groups, types.tolist())
        params_test.model = mod_df
        params_test.diet = diet_df
        try:
            mod_model, mod_diag = rpath(params_test, debug=True)
        except Exception as e:
            print(f'Candidate {cand} raised error: {e}')
            continue
        iter_snap = mod_diag.get('iterations', [])
        if iter_snap:
            snap = iter_snap[0]
            A_mod = snap['A']
            b_mod = snap['b_vec']
        else:
            A_mod = mod_diag['A']
            b_mod = mod_diag['b_vec']
        sc, scA, scb = score_A_b(A_mod, b_mod)
        print(f' - cand {cand}: score={sc:.4f} (A={scA:.4f}, b={scb:.4f})')
        if sc < best_local_score:
            best_local = cand
            best_local_score = sc
            best_local_details = (A_mod, b_mod, scA, scb)

    if best_local is None:
        print('No flip improves the score. Stopping.')
        break

    # apply best_local flip
    print(f'Applying flip {best_local} improved score {current_score:.4f} -> {best_local_score:.4f}')
    working_df.loc[working_df['Group'] == groups[best_local], 'EE'] = 1.0
    working_df.loc[working_df['Group'] == groups[best_local], 'Biomass'] = np.nan
    applied_flips.append(best_local)
    current_score = best_local_score

# Final run
params_final = create_rpath_params(groups, types.tolist())
params_final.model = working_df.copy()
params_final.diet = diet_df
final_model, final_diag = rpath(params_final, debug=True)

out = {
    'applied_flips': applied_flips,
    'final_score': float(current_score)
}
json.dump(out, open(ECOPATH_DIR / 'heuristic_multi_flip_results.json', 'w'), indent=2)
np.save(ECOPATH_DIR / 'heuristic_multi_flip_final_A.npy', final_diag['A'])
np.save(ECOPATH_DIR / 'heuristic_multi_flip_final_bvec.npy', final_diag['b_vec'])
print('Done. Applied flips:', applied_flips)
