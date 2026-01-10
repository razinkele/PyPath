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

params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df.copy()
params.diet = diet_df
model, diag = rpath(params, debug=True)

types = model_df['Type'].values.astype(float)
living_idx = [i for i,t in enumerate(types) if t < 2]
dead_idx = [i for i,t in enumerate(types) if t == 2]

# recalc losses per source
biomass = model.Biomass
pb = model.PB
qb = model.QB
unassim = model.Unassim

m0 = np.zeros(len(biomass))
for i, idx in enumerate(living_idx):
    m0[idx] = pb[idx] * (1 - model.EE[idx])
loss = np.zeros(len(biomass))
qb_loss = np.where(np.isnan(qb), 0.0, qb)
for idx in living_idx:
    m0_pos = max(0.0, m0[idx])
    loss[idx] = (m0_pos * biomass[idx]) + (biomass[idx] * qb_loss[idx] * unassim[idx])

# detfate matrix
det_groups = [model_df.loc[i,'Group'] for i in dead_idx]
# build detfate
detfate = []
for d_idx, det in enumerate(det_groups):
    if det in model_df.columns:
        detfate.append(model_df[det].values.astype(float))
    else:
        detfate.append([0]*len(biomass))
# transpose to get all_source_detfate
import numpy as np
all_source_detfate = np.array(detfate).T

# compute contributions to det 0
det0_contribs = []
for src_idx in range(len(biomass)):
    contrib = loss[src_idx] * all_source_detfate[src_idx, 0]
    if contrib > 0:
        det0_contribs.append((src_idx, model_df.loc[src_idx,'Group'], float(loss[src_idx]), float(all_source_detfate[src_idx,0]), float(contrib)))

det0_contribs_sorted = sorted(det0_contribs, key=lambda x: x[3], reverse=True)
print('Top det0 contributors by fraction:')
for c in det0_contribs_sorted[:20]:
    print(c)

print('Sum contribs:', sum([c[3] for c in det0_contribs_sorted]))
print('Det inputs:', diag['detinputs'][0], 'detcons', diag['detcons'][0])
json.dump(det0_contribs_sorted, open(ECOPATH_DIR / 'det0_contribs.json', 'w'), indent=2)
print('Saved det0_contribs.json')
