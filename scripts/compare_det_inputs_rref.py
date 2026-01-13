import json
from pathlib import Path
import numpy as np
import pandas as pd
ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')

rref = json.load(open(ECOPATH_DIR / 'balanced_model.json'))
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')

groups = model_df['Group'].tolist()
types = model_df['Type'].values.astype(float)
living_idx = [i for i,t in enumerate(types) if t < 2]
dead_idx = [i for i,t in enumerate(types) if t == 2]

B_r = np.array(rref['Biomass'], dtype=float)
PB_r = np.array(rref['PB'], dtype=float)
QB_r = np.array(rref['QB'], dtype=float)
EE_r = np.array(rref['EE'], dtype=float)

unassim = model_df['Unassim'].values.astype(float)
unassim = np.nan_to_num(unassim, nan=0.0)

m0 = np.zeros(len(B_r))
for i in living_idx:
    m0[i] = PB_r[i] * (1 - EE_r[i])
loss = np.zeros(len(B_r))
for i in living_idx:
    m0_pos = max(0.0, m0[i])
    qb_val = QB_r[i] if not np.isnan(QB_r[i]) else 0.0
    loss[i] = m0_pos * B_r[i] + (B_r[i] * qb_val * unassim[i])

# detfate
ndead = len(dead_idx)
detfate = np.zeros((len(B_r), ndead))
for d_idx, det_name in enumerate([model_df.loc[i,'Group'] for i in dead_idx]):
    if det_name in model_df.columns:
        detfate[:, d_idx] = model_df[det_name].values.astype(float)

all_source_detfate = detfate

detinputs_r = np.sum(loss[:, np.newaxis] * all_source_detfate, axis=0) + np.zeros(ndead)
print('detinputs_r', detinputs_r)
print('detcons_r', np.sum(loss[:, np.newaxis] * all_source_detfate, axis=0))
print('rref EE for det', EE_r[dead_idx])
json.dump({'detinputs_r': detinputs_r.tolist()}, open(ECOPATH_DIR / 'detinputs_r.json','w'), indent=2)
print('Saved detinputs_r.json')
