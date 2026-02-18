from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.params import create_rpath_params

REFERENCE_DIR = Path("tests/data/rpath_reference")
ECOPATH_DIR = REFERENCE_DIR / "ecopath"
ECOSIM_DIR = REFERENCE_DIR / "ecosim"

model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

groups = model_df["Group"].tolist()
params = create_rpath_params(groups, [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = diet_df

pypath_model = rpath(params)
scenario = rsim_scenario(pypath_model, params, years=range(1, 101))
out = rsim_run(scenario, method='rk4', years=range(1, 101))
py_biom = out.out_Biomass
rpath_df = pd.read_csv(ECOSIM_DIR / "biomass_trajectory_rk4.csv")

worst_month = 195
sidx = scenario.params.spname.index('Seabirds')
didx = scenario.params.spname.index('Discards') if 'Discards' in scenario.params.spname else None
print('month', worst_month, 'sidx', sidx, 'didx', didx)
state_py = py_biom[worst_month].copy()
state_r = state_py.copy()
# replace with rpath state
rrow = rpath_df.iloc[worst_month]
for i, g in enumerate(groups, start=1):
    if g in rrow:
        state_r[i] = float(rrow[g])

print('\nSample biomass values (group index, name, B_py, B_r, Bbase):')
for idx in [didx, sidx, 13,14,15,12]:
    name = scenario.params.spname[idx]
    print(idx, name, state_py[idx], state_r[idx], scenario.params.B_BaseRef[idx])

# prepare params dict for deriv_vector
from pypath.core.ecosim import (
    _build_active_link_matrix,
    _build_link_matrix,
    _compute_Q_matrix,
)

params_dict = {
    'NUM_GROUPS': scenario.params.NUM_GROUPS,
    'NUM_LIVING': scenario.params.NUM_LIVING,
    'NUM_DEAD': scenario.params.NUM_DEAD,
    'NUM_GEARS': scenario.params.NUM_GEARS,
    'PB': scenario.params.PBopt,
    'QB': scenario.params.FtimeQBOpt,
    'M0': scenario.params.MzeroMort.copy(),
    'Unassim': scenario.params.UnassimRespFrac,
    'ActiveLink': _build_active_link_matrix(scenario.params),
    'VV': _build_link_matrix(scenario.params, scenario.params.VV),
    'DD': _build_link_matrix(scenario.params, scenario.params.DD),
    'QQbase': _build_link_matrix(scenario.params, scenario.params.QQ),
    'Bbase': scenario.params.B_BaseRef,
    'PP_type': scenario.params.PP_type,
}

forcing = {
    'Ftime': scenario.start_state.Ftime.copy(),
    'ForcedBio': np.where(scenario.forcing.ForcedBio[worst_month] > 0, scenario.forcing.ForcedBio[worst_month], 0),
    'ForcedMigrate': scenario.forcing.ForcedMigrate[worst_month],
    'ForcedEffort': (scenario.fishing.ForcedEffort[worst_month] if worst_month < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS+1)),
}

# call deriv_vector to get detritus diagnostics and full deriv
# deriv_vector returns (deriv, diag) perhaps; inspect signature
from inspect import signature

print('deriv_vector signature:', signature(deriv_vector))

# run for PyPath state
try:
    deriv_py = deriv_vector(state_py, params_dict, forcing, {}, t=worst_month)
    print('deriv_py type:', type(deriv_py))
except Exception as e:
    print('deriv_py call failed:', e)

try:
    deriv_r = deriv_vector(state_r, params_dict, forcing, {}, t=worst_month)
    print('deriv_r type:', type(deriv_r))
except Exception as e:
    print('deriv_r call failed:', e)

# deriv_vector outputs: check type
print('deriv_py type:', type(deriv_py))

# try to print detritus diagnostics in deriv_py (if present)
if isinstance(deriv_py, dict):
    print('deriv_py keys:', deriv_py.keys())

# fallback: compute Q matrices and values printed earlier
QQ_py = _compute_Q_matrix(params_dict, state_py, forcing)
QQ_r = _compute_Q_matrix(params_dict, state_r, forcing)

col_py = QQ_py[:, sidx]
col_r = QQ_r[:, sidx]
print('Seabirds column differences top 10:')
order = np.argsort(np.abs(col_py - col_r))[::-1]
for idx in order[:10]:
    print(idx, scenario.params.spname[idx], col_py[idx], col_r[idx], abs(col_py[idx]-col_r[idx]))

print('\nDiscards B_py, B_r, Bbase:')
print(state_py[didx], state_r[didx], params_dict['Bbase'][didx])

# print detailed detritus inputs for Discards if available
# if deriv_vector prints DETRITUS entries when DEBUG=True, they go to stdout; this script also prints the Q contributions above.
print('\nDone')
