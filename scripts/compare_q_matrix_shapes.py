import numpy as np
import pandas as pd
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _compute_Q_matrix

REPO = Path(__file__).parent.parent
ECOPATH_DIR = REPO / 'tests' / 'data' / 'rpath_reference' / 'ecopath'
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df

r = rpath(params)
scenario = rsim_scenario(r, params, years=range(1,101))
out = rsim_run(scenario, method='rk4', years=range(1,101))
py_biom = out.out_Biomass

sp = scenario.params
NUM_GROUPS = sp.NUM_GROUPS

# Helper to ensure a matrix is global (NUM_GROUPS+1, NUM_GROUPS+1)
def ensure_mat_global(mat):
    mat = np.asarray(mat)
    if mat.ndim == 1:
        try:
            return mat.reshape((NUM_GROUPS + 1, NUM_GROUPS + 1))
        except Exception:
            return np.zeros((NUM_GROUPS + 1, NUM_GROUPS + 1))
    return mat

# Build matrices
Active = _build_active = None
from pypath.core.ecosim import _build_active_link_matrix, _build_link_matrix
Active = _build_active_link_matrix(sp)
VV = ensure_mat_global(sp.VV)
DD = ensure_mat_global(sp.DD)
QQbase = ensure_mat_global(sp.QQ)

print('VV shape:', VV.shape, 'DD shape:', DD.shape, 'QQbase shape:', QQbase.shape)

# Compute QQ for month 1
month = 1
state = out.out_Biomass[month].copy()
forcing = {'Ftime': scenario.start_state.Ftime.copy(), 'ForcedBio': np.where(scenario.forcing.ForcedBio[month] > 0, scenario.forcing.ForcedBio[month], 0), 'ForcedMigrate': scenario.forcing.ForcedMigrate[month], 'ForcedEffort': (scenario.fishing.ForcedEffort[month] if month < len(scenario.fishing.ForcedEffort) else np.ones(sp.NUM_GEARS+1))}

QQ = _compute_Q_matrix({'NUM_GROUPS': sp.NUM_GROUPS, 'NUM_LIVING': sp.NUM_LIVING, 'NUM_DEAD': sp.NUM_DEAD, 'NUM_GEARS': sp.NUM_GEARS, 'PB': sp.PBopt, 'QB': sp.FtimeQBOpt, 'M0': sp.MzeroMort, 'Unassim': sp.UnassimRespFrac, 'ActiveLink': Active, 'VV': VV, 'DD': DD, 'QQbase': QQbase, 'Bbase': sp.B_BaseRef, 'PP_type': sp.PP_type}, state, forcing)

groups = model_df['Group'].tolist()
seab_idx = groups.index('Seabirds') + 1
print('\nNon-zero QQ (month 1) for Seabirds predator:')
for i in range(1, NUM_GROUPS+1):
    if QQ[i, seab_idx] != 0.0:
        print(i, groups[i-1], QQ[i, seab_idx])
print('\nQQbase columns for Seabirds (non-zero):')
for i in range(1, NUM_GROUPS+1):
    if QQbase[i, seab_idx] != 0.0:
        print(i, groups[i-1], QQbase[i, seab_idx])
print('done')