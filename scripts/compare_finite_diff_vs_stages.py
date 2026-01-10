"""Compare Rpath finite-difference derivative to PyPath RK4 stage derivatives for a group

Usage: python scripts/compare_finite_diff_vs_stages.py --month 0 --group Discards --pred Seabirds
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _build_link_matrix, _build_active_link_matrix
from pypath.core.ecosim_deriv import deriv_vector

RE = Path('tests/data/rpath_reference')
ECOPATH_DIR = RE / 'ecopath'
ECOSIM_DIR = RE / 'ecosim'

model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = diet_df
pypath_model = rpath(params)
scenario = rsim_scenario(pypath_model, params, years=range(1, 101))
# official PyPath run (for comparison)
out = rsim_run(scenario, method='rk4', years=range(1, 101))
py_biom = out.out_Biomass

rpath_df = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_rk4.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--month', type=int, default=0)
parser.add_argument('--group', type=str, default='Discards')
parser.add_argument('--pred', type=str, default='Seabirds')
args = parser.parse_args()
month = args.month
group_name = args.group
pred_name = args.pred

NUM_GROUPS = scenario.params.NUM_GROUPS

# build pdict and instrument groups
pdict = {
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
    'spname': scenario.params.spname,
}
# include NoIntegrate flags from scenario so deriv_vector uses algebraic groups
pdict['NoIntegrate'] = scenario.params.NoIntegrate
pdict['INSTRUMENT_GROUPS'] = [group_name, pred_name]

# fishing
FishMort = np.zeros(scenario.params.NUM_GROUPS + 1)
for i in range(1, len(scenario.params.FishFrom)):
    grp = int(scenario.params.FishFrom[i])
    FishMort[grp] += scenario.params.FishQ[i]

fishing = {'FishingMort': FishMort, 'FishFrom': scenario.params.FishFrom, 'FishThrough': getattr(scenario.params, 'FishThrough', None), 'FishQ': getattr(scenario.params, 'FishQ', None)}

# helper compute_Q
from pypath.core.ecosim import _compute_Q_matrix as compute_Q

# prepare state from Rpath row
if month < 0 or (month + 1) >= len(rpath_df):
    raise SystemExit(f"Month {month} out of bounds for finite-diff (need month and month+1 available)")

row = rpath_df.iloc[month]
state_r = np.zeros(NUM_GROUPS + 1)
for i, g in enumerate(model_df['Group'].tolist(), start=1):
    if g in row:
        state_r[i] = float(row[g])

rown = rpath_df.iloc[month + 1]
rstate_next = np.zeros(NUM_GROUPS + 1)
for i, g in enumerate(model_df['Group'].tolist(), start=1):
    if g in rown:
        rstate_next[i] = float(rown[g])

# forcings
forcing = {
    'Ftime': scenario.start_state.Ftime.copy(),
    'ForcedBio': (scenario.forcing.ForcedBio[month] if hasattr(scenario.forcing, 'ForcedBio') else np.zeros(scenario.params.NUM_GROUPS + 1)),
    'ForcedMigrate': (scenario.forcing.ForcedMigrate[month] if hasattr(scenario.forcing, 'ForcedMigrate') else np.zeros(scenario.params.NUM_GROUPS + 1)),
    'ForcedEffort': (scenario.fishing.ForcedEffort[month] if month < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS + 1)),
    'PP_forcing': (scenario.forcing.PP_forcing[month] if hasattr(scenario.forcing, 'PP_forcing') else np.ones(scenario.params.NUM_GROUPS + 1)),
    'ForcedPrey': (getattr(scenario.forcing, 'ForcedPrey', np.ones(scenario.params.NUM_GROUPS + 1))[month] if hasattr(scenario.forcing, 'ForcedPrey') else np.ones(scenario.params.NUM_GROUPS + 1)),
}

# Derivatives (PyPath) at Rpath state and intermediate RK4 stages
k1 = deriv_vector(state_r.copy(), pdict, forcing, fishing, t=month)
s2 = state_r + 0.5 * (1.0 / 12.0) * k1
k2 = deriv_vector(s2, pdict, forcing, fishing, t=month)
s3 = state_r + 0.5 * (1.0 / 12.0) * k2
k3 = deriv_vector(s3, pdict, forcing, fishing, t=month)
s4 = state_r + (1.0 / 12.0) * k3
k4 = deriv_vector(s4, pdict, forcing, fishing, t=month)
s_next = state_r + (1.0 / 12.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# indices
if group_name not in pdict['spname'] or pred_name not in pdict['spname']:
    raise SystemExit('Groups not in model')
d_idx = pdict['spname'].index(group_name)
s_idx = pdict['spname'].index(pred_name)

# finite-diff derivative (annualized)
B_here = state_r[d_idx]
B_next = rstate_next[d_idx]
fd_month = (B_next - B_here)
fd_annual = fd_month * 12.0

# PyPath derivatives
k1d = k1[d_idx]
k2d = k2[d_idx]
k3d = k3[d_idx]
k4d = k4[d_idx]
rk4_annual = (s_next[d_idx] - state_r[d_idx]) * 12.0

# QQ comparisons
QQ_start = compute_Q(pdict, state_r, forcing)
QQ_r_next = compute_Q(pdict, rstate_next, forcing)
QQ_py_next = compute_Q(pdict, s_next, forcing)
q_start = QQ_start[d_idx, s_idx]
q_r_next = QQ_r_next[d_idx, s_idx]
q_py_next = QQ_py_next[d_idx, s_idx]
qbase = pdict['QQbase'][d_idx, s_idx]

# breakdown calc Q formula
Bbase = pdict['Bbase']
PYY = (state_r[d_idx] / Bbase[d_idx]) if Bbase[d_idx] > 0 else 0.0
PDY = (scenario.start_state.Ftime[s_idx] * state_r[s_idx] / Bbase[s_idx]) if Bbase[s_idx] > 0 else 0.0
dd = pdict['DD'][d_idx, s_idx]
vv = pdict['VV'][d_idx, s_idx]

dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
calc_Q = qbase * PDY * PYY * dd_term * vv_term

# print results
print('\n=== Finite-diff vs PyPath derivatives (group: {0}) ==='.format(group_name))
print(f'Month {month} Rpath: B_here={B_here:.12e} B_next={B_next:.12e} delta_month={fd_month:.12e} finite-diff annual={fd_annual:.12e}')
print('\nPyPath derivatives at Rpath start state (per-year units):')
print(f'  k1[{group_name}] = {k1d:.12e}')
print(f'  k2[{group_name}] = {k2d:.12e}')
print(f'  k3[{group_name}] = {k3d:.12e}')
print(f'  k4[{group_name}] = {k4d:.12e}')
print(f'  RK4 integrated annual delta = {rk4_annual:.12e} (s_next - state)*12')
print('\nDifferences:')
print(f'  k1 - fd_annual = {k1d - fd_annual:.12e}  (relative {(k1d - fd_annual) / (fd_annual if fd_annual!=0 else 1):.6e})')
print(f'  RK4_annual - fd_annual = {rk4_annual - fd_annual:.12e}  (relative {(rk4_annual - fd_annual) / (fd_annual if fd_annual!=0 else 1):.6e})')

print('\n=== Q link comparisons (Seabirds<-Discards) ===')
print(f'  Q start = {q_start:.12e}')
print(f'  Q Rpath next = {q_r_next:.12e}')
print(f'  Q PyPath next = {q_py_next:.12e}')
print(f'  qbase = {qbase:.12e}  calc_Q(start) = {calc_Q:.12e}')

print('\nCandidate code locations to inspect:')
print('  - src/pypath/core/ecosim_deriv.py  (deriv_vector: detritus handling, NoIntegrate enforcement, per-term contributions)')
print('  - src/pypath/core/ecosim.py        (_compute_Q_matrix: Q formation, dd/vv normalization)')
print('  - src/pypath/core/params.py        (DetFrac and fish-discard mapping, fish-derived DetFrac assembly)')
print('\nDone')
