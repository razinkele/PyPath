"""
Relaxed heuristic flip simulation: try flipping each candidate EE>1 under several candidate-selection policies
(1) original_no_b & original_no_ee (strict, previous)
(2) original_no_b
(3) original_no_ee
(4) any EE>1
Run rpath for each flip and choose the flip minimizing the difference to R-like reference intermediates (A and b_vec).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params

ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')

# Load reference balanced model and build R-like A_ref/b_ref
rref = json.load(open(ECOPATH_DIR / 'balanced_model.json'))
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')

groups = model_df['Group'].tolist()
ngroups = len(groups)
types = model_df['Type'].values.astype(float)
living_idx = np.where(types < 2)[0]
nliving = len(living_idx)

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

# compute R-like A_ref and b_vec_ref using balanced rref
B = np.array(rref['Biomass'], dtype=float)
PB = np.array(rref['PB'], dtype=float)
QB = np.array(rref['QB'], dtype=float)

bio_qb_ref = np.nan_to_num(B[living_idx] * QB[living_idx], nan=0.0)
cons_ref = nodetrdiet * bio_qb_ref[np.newaxis, :]

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

# Run base PyPath rpath to collect diagnostics
params = create_rpath_params(groups, types.tolist())
params.model = model_df.copy()
params.diet = diet_df
base_model, base_diag = rpath(params, debug=True)

# Determine candidates under various policies
cand_sets = {}
base_ee = base_model.EE
cand_sets['strict_orig_b_and_orig_ee'] = [idx for idx in living_idx if original_no_b[idx] and original_no_ee[idx] and base_ee[idx] > 1.0]
cand_sets['orig_b'] = [idx for idx in living_idx if original_no_b[idx] and base_ee[idx] > 1.0]
cand_sets['orig_ee'] = [idx for idx in living_idx if original_no_ee[idx] and base_ee[idx] > 1.0]
cand_sets['any_ee_gt1'] = [idx for idx in living_idx if base_ee[idx] > 1.0]

print('Candidate sets:')
for k, v in cand_sets.items():
    print(f' - {k}: {v}')

# Flatten unique candidates and keep track of source policy
candidate_to_policies = {}
for policy, cids in cand_sets.items():
    for c in cids:
        candidate_to_policies.setdefault(c, []).append(policy)

if not candidate_to_policies:
    print('No candidates found under any policy. Exiting.')
    raise SystemExit(0)

candidates = sorted(candidate_to_policies.keys())
print('Union candidates:', candidates)

# Evaluate each candidate by running rpath with that single flip applied
best = None
best_score = float('inf')
results = []
for cand in candidates:
    mod_params = create_rpath_params(groups, types.tolist())
    mod_df = model_df.copy()
    # Apply flip: set EE to 1.0 (cap) and set biomass to NaN to force solving for biomass
    mod_df.loc[mod_df['Group'] == groups[cand], 'EE'] = 1.0
    mod_df.loc[mod_df['Group'] == groups[cand], 'Biomass'] = np.nan
    mod_params.model = mod_df
    mod_params.diet = diet_df

    try:
        mod_model, mod_diag = rpath(mod_params, debug=True)
    except Exception as e:
        print(f'Flip candidate {cand} caused error: {e}')
        continue

    # Use the first iteration snapshot if available, else final A/b
    iter_snap = mod_diag.get('iterations', [])
    if iter_snap:
        snap = iter_snap[0]
        A_mod = snap['A']
        b_mod = snap['b_vec']
    else:
        A_mod = mod_diag['A']
        b_mod = mod_diag['b_vec']

    # Compute score = sum of max abs diffs for A and b (weighted)
    score_A = np.nanmax(np.abs(A_ref - A_mod))
    score_b = np.nanmax(np.abs(b_vec_ref - b_mod))
    score = score_A + score_b
    results.append({'cand': int(cand), 'policies': candidate_to_policies.get(cand, []), 'score': float(score), 'score_A': float(score_A), 'score_b': float(score_b)})
    print(f'Candidate {cand}: score={score:.4f} (A={score_A:.4f}, b={score_b:.4f}) policies={candidate_to_policies.get(cand, [])}')
    if score < best_score:
        best = cand
        best_score = score

if best is None:
    print('No successful candidate flips.')
    raise SystemExit(1)

print(f'Best flip: {best} (policies={candidate_to_policies.get(best, [])}) with score {best_score}')

# Apply best flip to original model and re-run rpath to produce final model
final_params = create_rpath_params(groups, types.tolist())
final_df = model_df.copy()
final_df.loc[final_df['Group'] == groups[best], 'EE'] = 1.0
final_df.loc[final_df['Group'] == groups[best], 'Biomass'] = np.nan
final_params.model = final_df
final_params.diet = diet_df
final_model, final_diag = rpath(final_params, debug=True)

# Save decision and diagnostics
out = {
    'best_flip': int(best),
    'best_policies': candidate_to_policies.get(best, []),
    'best_score': float(best_score),
    'results': results
}
json.dump(out, open(ECOPATH_DIR / 'heuristic_flip_relaxed_results.json', 'w'), indent=2)

# Save final diagnostics to file
np.save(ECOPATH_DIR / 'heuristic_final_relaxed_A.npy', final_diag['A'])
np.save(ECOPATH_DIR / 'heuristic_final_relaxed_bvec.npy', final_diag['b_vec'])

print('Applied best relaxed flip and saved final diagnostics.')
