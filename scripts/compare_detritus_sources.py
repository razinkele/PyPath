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
living_idx = [i for i, t in enumerate(types) if t < 2]
dead_idx = [i for i, t in enumerate(types) if t == 2]

# get detfate matrix same as in ecopath
ndead = len(dead_idx)
detfate = np.zeros((ngroups, ndead))
for d_local, det_global in enumerate(dead_idx):
    det_name = groups[det_global]
    if det_name in model_df.columns:
        detfate[:, d_local] = model_df[det_name].values.astype(float)

# PyPath run
params = create_rpath_params(groups, types.tolist())
params.model = model_df.copy()
params.diet = diet_df
model_py, diag = rpath(params, debug=True)

B_py = np.array(model_py.Biomass, dtype=float)
PB_py = np.array(model_py.PB, dtype=float)
QB_py = np.array(model_py.QB, dtype=float)
EE_py = np.array(model_py.EE, dtype=float)
unassim = np.nan_to_num(model_df['Unassim'].values.astype(float), nan=0.0)

# compute per-source loss for PyPath
m0_py = np.zeros(ngroups)
for i in living_idx:
    m0_py[i] = PB_py[i] * (1 - EE_py[i])
loss_py = np.zeros(ngroups)
qb_val_py = np.nan_to_num(QB_py, nan=0.0)
for idx in living_idx:
    m0pos = max(0.0, m0_py[idx])
    loss_py[idx] = (m0pos * B_py[idx]) + (B_py[idx] * qb_val_py[idx] * unassim[idx])

# add fleet discards (as in ecopath code)
fleet_idx = [i for i,t in enumerate(types) if t == 3]
if fleet_idx:
    # build land/discard as in other scripts
    landmat = np.zeros((ngroups, len(fleet_idx)))
    discardmat = np.zeros((ngroups, len(fleet_idx)))
    fleet_names = [groups[i] for i in fleet_idx]
    for g, name in enumerate(fleet_names):
        if name in model_df.columns:
            landmat[:, g] = model_df[name].values.astype(float)
        disc = f"{name}.disc"
        if disc in model_df.columns:
            discardmat[:, g] = model_df[disc].values.astype(float)
    discardmat = np.nan_to_num(discardmat, nan=0.0)
    for f_i, fg in enumerate(fleet_idx):
        loss_py[fg] = np.sum(discardmat[living_idx, f_i])

# contributions to detritus
contribs_py = (loss_py[:, np.newaxis] * detfate)
detinputs_py = np.sum(contribs_py, axis=0)
detcons_py = diag['detcons'] if 'detcons' in diag else np.sum(contribs_py, axis=0)

# Rref computed loss
B_r = np.array(rref['Biomass'], dtype=float)
PB_r = np.array(rref['PB'], dtype=float)
QB_r = np.array(rref['QB'], dtype=float)
EE_r = np.array(rref['EE'], dtype=float)

m0_r = np.zeros(ngroups)
for i in living_idx:
    m0_r[i] = PB_r[i] * (1 - EE_r[i])
loss_r = np.zeros(ngroups)
qb_val_r = np.nan_to_num(QB_r, nan=0.0)
for idx in living_idx:
    m0pos = max(0.0, m0_r[idx])
    loss_r[idx] = (m0pos * B_r[idx]) + (B_r[idx] * qb_val_r[idx] * unassim[idx])
# fleets have zero loss in reference unless discards exist (we compute from model_df)
# compute discards from model_df's fleet columns as above
if fleet_idx:
    landmat_r = np.zeros((ngroups, len(fleet_idx)))
    discardmat_r = np.zeros((ngroups, len(fleet_idx)))
    fleet_names = [groups[i] for i in fleet_idx]
    for g, name in enumerate(fleet_names):
        if name in model_df.columns:
            landmat_r[:, g] = model_df[name].values.astype(float)
        disc = f"{name}.disc"
        if disc in model_df.columns:
            discardmat_r[:, g] = model_df[disc].values.astype(float)
    discardmat_r = np.nan_to_num(discardmat_r, nan=0.0)
    for f_i, fg in enumerate(fleet_idx):
        loss_r[fg] = np.sum(discardmat_r[living_idx, f_i])

contribs_r = (loss_r[:, np.newaxis] * detfate)
detinputs_r = np.sum(contribs_r, axis=0)

delta_inputs = detinputs_py - detinputs_r

# focus on det index 0 (Detritus)
d0_py = contribs_py[:, 0]
d0_r = contribs_r[:, 0]
d0_diff = d0_py - d0_r

# prepare report: top contributions whose abs diff is largest
rows = []
for i in range(ngroups):
    rows.append({
        'idx': int(i), 'group': groups[i], 'loss_py': float(loss_py[i]), 'loss_r': float(loss_r[i]),
        'd0_py': float(d0_py[i]), 'd0_r': float(d0_r[i]), 'd0_diff': float(d0_diff[i])
    })
rows_sorted = sorted(rows, key=lambda r: abs(r['d0_diff']), reverse=True)

out = {
    'detinputs_py': detinputs_py.tolist(),
    'detcons_py': detcons_py.tolist() if hasattr(detcons_py, 'tolist') else list(detcons_py),
    'detinputs_r': detinputs_r.tolist(),
    'delta_inputs': delta_inputs.tolist(),
    'per_source': rows_sorted[:50]
}
json.dump(out, open(ECOPATH_DIR / 'detritus_loss_comparison.json', 'w'), indent=2)
print('Saved detritus_loss_comparison.json')
print('Detritus: py=', detinputs_py[0], 'r=', detinputs_r[0], 'delta=', detinputs_py[0]-detinputs_r[0])
