import json
from pathlib import Path
import numpy as np
import pandas as pd

ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')

# Tolerances
RTOL = 1e-8
ATOL = 1e-8

# Load R balanced model
rpath_ref = json.load(open(ECOPATH_DIR / 'balanced_model.json'))
# Load original model params and diet
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')

groups = model_df['Group'].tolist()
ngroups = len(groups)

types = model_df['Type'].values.astype(float)
living_idx = np.where(types < 2)[0]
nliving = len(living_idx)

# Build diet_values (rows = groups + Import, cols = living predators)
pred_cols = [g for g in groups if g in diet_df.columns]
diet_values = np.zeros((ngroups + 1, len(pred_cols)))
prey_name_to_row = {name: i for i, name in enumerate(diet_df['Group'].tolist())}
for new_row_idx, group_name in enumerate(groups):
    if group_name in prey_name_to_row:
        old_row = prey_name_to_row[group_name]
        diet_values[new_row_idx, :] = diet_df.loc[old_row, pred_cols].values.astype(float)
if 'Import' in prey_name_to_row and len(diet_df['Group']) > ngroups:
    import_row_idx = prey_name_to_row['Import']
    diet_values[ngroups, :] = diet_df.loc[import_row_idx, pred_cols].values.astype(float)
# replace NA
diet_values = np.nan_to_num(diet_values, nan=0.0)

# Adjust mixotrophs
for col_idx, grp_idx in enumerate(living_idx):
    if 0 < types[grp_idx] < 1:
        mix_q = 1 - types[grp_idx]
        diet_values[:, col_idx] *= mix_q

# nodetrdiet
nodetrdiet = np.zeros((nliving, nliving))
for j, pred_global in enumerate(living_idx):
    # find column index in pred_cols matching this predator
    pred_name = groups[pred_global]
    col_idx = pred_cols.index(pred_name)
    import_frac = diet_values[ngroups, col_idx] if diet_values.shape[0] > ngroups else 0.0
    denom = (1 - import_frac) if (1 - import_frac) > 0 else 1.0
    for i, prey_global in enumerate(living_idx):
        nodetrdiet[i, j] = diet_values[prey_global, col_idx] / denom

# Balanced arrays from rpath_ref
B = np.array(rpath_ref['Biomass'], dtype=float)
PB = np.array(rpath_ref['PB'], dtype=float)
QB = np.array(rpath_ref['QB'], dtype=float)
EE = np.array(rpath_ref['EE'], dtype=float)

# bio_qb
bio_qb = np.nan_to_num(B[living_idx] * QB[living_idx], nan=0.0)
# cons
cons = nodetrdiet * bio_qb[np.newaxis, :]

# Compute landings/discards from original model
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
totcatch = np.sum(landmat + discardmat, axis=1)

bioacc = np.nan_to_num(model_df['BioAcc'].values.astype(float), nan=0.0)

living_catch = totcatch[living_idx]
living_bioacc = bioacc[living_idx]

b_vec_ref = living_catch + living_bioacc + np.sum(cons, axis=1)

# Build A_ref
original_no_b = np.isnan(model_df['Biomass'].values.astype(float))
original_no_ee = np.isnan(model_df['EE'].values.astype(float))

A_ref = np.zeros((nliving, nliving))
for i in range(nliving):
    idx_global = living_idx[i]
    if original_no_ee[idx_global]:
        if not np.isnan(B[idx_global]):
            A_ref[i, i] = B[idx_global] * PB[idx_global]
        else:
            # fallback
            A_ref[i, i] = PB[idx_global] * EE[idx_global]
    else:
        A_ref[i, i] = PB[idx_global] * EE[idx_global]

qb_dc = nodetrdiet * np.nan_to_num(QB[living_idx])[np.newaxis, :]
for j in range(nliving):
    if original_no_b[living_idx[j]]:
        A_ref[:, j] -= qb_dc[:, j]

# Load PyPath diagnostics (saved earlier)
diag_A_file = ECOPATH_DIR / 'diag_A.npy'
diag_b_file = ECOPATH_DIR / 'diag_bvec.npy'
if diag_A_file.exists() and diag_b_file.exists():
    A_p = np.load(diag_A_file)
    b_p = np.load(diag_b_file)
else:
    raise SystemExit('PyPath diagnostics not found; run scripts/debug_ecopath.py with debug=True first')

# Compare A matrices
abs_diff_A = np.abs(A_ref - A_p)
max_abs_A = np.nanmax(abs_diff_A)
rel_diff_A = np.where(np.abs(A_ref) > 0, abs_diff_A / np.abs(A_ref), abs_diff_A)
max_rel_A = np.nanmax(rel_diff_A)

print('A comparison: max_abs_diff =', max_abs_A, 'max_rel_diff =', max_rel_A)
indices_A = np.argwhere((abs_diff_A > ATOL) & (rel_diff_A > RTOL))
print('Number of differing A entries:', len(indices_A))
for i, j in indices_A[:20]:
    print(f'A[{i},{j}]: ref={A_ref[i,j]}, pypath={A_p[i,j]}, absdiff={abs_diff_A[i,j]}, reldiff={rel_diff_A[i,j]}')

# Compare b_vec
abs_diff_b = np.abs(b_vec_ref - b_p)
max_abs_b = np.nanmax(abs_diff_b)
rel_diff_b = abs_diff_b / np.where(np.abs(b_vec_ref) > 0, np.abs(b_vec_ref), 1.0)
max_rel_b = np.nanmax(rel_diff_b)
print('\nb_vec comparison: max_abs_diff =', max_abs_b, 'max_rel_diff =', max_rel_b)
indices_b = np.argwhere(abs_diff_b > ATOL)
print('b_vec differing indices (count):', len(indices_b))
for (i,) in indices_b[:20]:
    print(f'b[{i}]: ref={b_vec_ref[i]}, pypath={b_p[i]}, absdiff={abs_diff_b[i]}, reldiff={rel_diff_b[i]}')

# Save comparison summary
summary = {
    'A_max_abs_diff': float(max_abs_A),
    'A_max_rel_diff': float(max_rel_A),
    'A_n_diffs': int(len(indices_A)),
    'b_max_abs_diff': float(max_abs_b),
    'b_max_rel_diff': float(max_rel_b),
    'b_n_diffs': int(len(indices_b))
}
json.dump(summary, open(ECOPATH_DIR / 'intermediate_comparison_summary.json', 'w'), indent=2)
print('\nSaved comparison summary to', ECOPATH_DIR / 'intermediate_comparison_summary.json')
