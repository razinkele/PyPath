from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

BASE = Path('tests/data/rpath_reference')
ECOPATH_DIR = BASE / 'ecopath'
ECOSIM_DIR = BASE / 'ecosim'

model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')

groups = model_df['Group'].tolist()
params = create_rpath_params(groups, model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df

# Instrument Macrobenthos
params.INSTRUMENT_GROUPS = ['Macrobenthos']
instrumented = []

def cb(payload):
    instrumented.append(payload)

params.instrument_callback = cb

pypath_model = rpath(params)

scenario = rsim_scenario(pypath_model, params, years=range(1, 3))
print('Scenario params has INSTRUMENT_GROUPS:', hasattr(scenario.params, 'INSTRUMENT_GROUPS') and scenario.params.INSTRUMENT_GROUPS)
print('Scenario params has instrument_callback:', hasattr(scenario.params, 'instrument_callback'))
out_ab = rsim_run(scenario, method='AB', years=range(1, 3))

rpath_traj_ab = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_ab.csv')
macrob_idx = groups.index('Macrobenthos')
macrob_out_idx = macrob_idx + 1
rvals = rpath_traj_ab['Macrobenthos'].values
pvals = out_ab.out_Biomass[:len(rvals), macrob_out_idx]

L = min(len(rvals), len(pvals), 12)
print('Rpath Macrobenthos first values:', rvals[:L])
print('PyPath Macrobenthos first values:', pvals[:L])
print('Diffs:', (pvals[:L] - rvals[:L]))
print('Rel diffs:', np.where(rvals[:L]!=0, (pvals[:L]-rvals[:L])/rvals[:L], np.inf))

print('\nInstrumented payloads count =', len(instrumented))
if instrumented:
    print('First payload keys:', instrumented[0].keys())
    grp_idxs = instrumented[0].get('groups', [])
    print('Groups in payload (0-based indices):', grp_idxs)
    # Map 0-based indices back to group names for readability
    try:
        print('Group names in payload:', [groups[i] for i in grp_idxs])
    except Exception:
        pass
    print('deriv_current sample:', instrumented[0].get('deriv_current'))
    print('derivs_history lengths:', [len(h) for h in instrumented[0].get('derivs_history', [])])

print('\nDone')
