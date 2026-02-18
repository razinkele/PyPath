"""Compute RK4-like advance but keeping consumption matrix QQ fixed at its start value.

Usage: python scripts/compare_with_constant_Q.py --month 0 --group Discards --pred Seabirds
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import (
    _build_active_link_matrix,
    _build_link_matrix,
    _compute_Q_matrix,
    rsim_scenario,
)
from pypath.core.params import create_rpath_params

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

parser = argparse.ArgumentParser()
parser.add_argument('--month', type=int, default=0)
parser.add_argument('--group', type=str, default='Discards')
parser.add_argument('--pred', type=str, default='Seabirds')
args = parser.parse_args()
month = args.month
group_name = args.group
pred_name = args.pred

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

# prepare state from Rpath row
rpath_df = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_rk4.csv')
row = rpath_df.iloc[month]
state_r = np.zeros(scenario.params.NUM_GROUPS + 1)
for i, g in enumerate(model_df['Group'].tolist(), start=1):
    if g in row:
        state_r[i] = float(row[g])

# forcings
forcing = {
    'Ftime': scenario.start_state.Ftime.copy(),
    'ForcedBio': (scenario.forcing.ForcedBio[month] if hasattr(scenario.forcing, 'ForcedBio') else np.zeros(scenario.params.NUM_GROUPS + 1)),
    'ForcedMigrate': (scenario.forcing.ForcedMigrate[month] if hasattr(scenario.forcing, 'ForcedMigrate') else np.zeros(scenario.params.NUM_GROUPS + 1)),
    'ForcedEffort': (scenario.fishing.ForcedEffort[month] if month < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS + 1)),
    'PP_forcing': (scenario.forcing.PP_forcing[month] if hasattr(scenario.forcing, 'PP_forcing') else np.ones(scenario.params.NUM_GROUPS + 1)),
    'ForcedPrey': (getattr(scenario.forcing, 'ForcedPrey', np.ones(scenario.params.NUM_GROUPS + 1))[month] if hasattr(scenario.forcing, 'ForcedPrey') else np.ones(scenario.params.NUM_GROUPS + 1)),
}

# QQ fixed at start
QQ_start = _compute_Q_matrix(pdict, state_r, forcing)

# helper to compute derivative using fixed QQ

def deriv_with_fixed_Q(state, QQ_fixed):
    # Copy of necessary logic from deriv_vector but using QQ_fixed
    NUM_GROUPS = pdict['NUM_GROUPS']
    NUM_LIVING = pdict['NUM_LIVING']
    deriv = np.zeros(NUM_GROUPS + 1)
    BB = state.copy()

    # primary production
    PB = pdict['PB']
    QB = pdict['QB']
    PP_type = pdict['PP_type']
    # Unassim available in pdict as pdict['Unassim'] (used below via pdict)

    # primary production (simplified): use PB * BB for producers
    # compute consumption and predation as per deriv_vector
    for i in range(1, NUM_LIVING + 1):
        consumption = np.sum(QQ_fixed[1:, i])
        predation_loss = np.sum(QQ_fixed[i, 1: NUM_LIVING + 1])
        m0 = pdict['M0'][i] if isinstance(pdict['M0'], np.ndarray) else 0.0
        if PP_type[i] > 0:
            production = PB[i] * BB[i]
        elif QB[i] > 0:
            GE = PB[i] / QB[i]
            production = GE * consumption
        else:
            production = PB[i] * BB[i]
        deriv[i] = production - predation_loss - 0.0 - m0 * BB[i]

    # detritus
    DetFrac = np.asarray(pdict.get('DetFrac', np.zeros((NUM_GROUPS + 1, 1))))
    for d in range(pdict['NUM_LIVING'] + 1, pdict['NUM_LIVING'] + pdict['NUM_DEAD'] + 1):
        det_idx = d - pdict['NUM_LIVING']
        unas_input = 0.0
        for pred in range(1, pdict['NUM_LIVING'] + 1):
            total_consump = np.sum(QQ_fixed[1:, pred])
            unas_input += total_consump * pdict['Unassim'][pred] * (DetFrac[pred, det_idx] if DetFrac.shape[1] > det_idx else 0)
        mort_input = 0.0
        for grp in range(1, pdict['NUM_LIVING'] + 1):
            mort_input += pdict.get('M0', np.zeros(NUM_GROUPS + 1))[grp] * state[grp] * (DetFrac[grp, det_idx] if DetFrac.shape[1] > det_idx else 0)
        det_consumed = np.sum(QQ_fixed[d, 1: pdict['NUM_LIVING'] + 1])
        decay_rate = pdict.get('DetDecay', np.zeros(pdict['NUM_DEAD'] + 1))
        decay = decay_rate[det_idx] * state[d] if len(decay_rate) > det_idx else 0
        deriv[d] = unas_input + mort_input - det_consumed - decay
    return deriv

# compute k1..k4 with fixed QQ
k1 = deriv_with_fixed_Q(state_r, QQ_start)
s2 = state_r + 0.5 * (1.0 / 12.0) * k1
k2 = deriv_with_fixed_Q(s2, QQ_start)
s3 = state_r + 0.5 * (1.0 / 12.0) * k2
k3 = deriv_with_fixed_Q(s3, QQ_start)
s4 = state_r + (1.0 / 12.0) * k3
k4 = deriv_with_fixed_Q(s4, QQ_start)
s_next = state_r + (1.0 / 12.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# report for group
if group_name not in pdict['spname'] or pred_name not in pdict['spname']:
    raise SystemExit('Groups not in model')
d_idx = pdict['spname'].index(group_name)
s_idx = pdict['spname'].index(pred_name)

print('=== RK4 with fixed QQ at start ===')
print(f'  k1[{group_name}] = {k1[d_idx]:.12e}')
print(f'  k2[{group_name}] = {k2[d_idx]:.12e}')
print(f'  k3[{group_name}] = {k3[d_idx]:.12e}')
print(f'  k4[{group_name}] = {k4[d_idx]:.12e}')
print(f'  RK4 integrated annual delta = {(s_next[d_idx] - state_r[d_idx]) * 12.0:.12e}')

# Print QQ-based Q for link
q_start = QQ_start[d_idx, s_idx]
print('\nQQ start Q on link =', q_start)
print('QQ evaluated at Rpath next (for reference) =', _compute_Q_matrix(pdict, state_r, forcing)[d_idx, s_idx])
print('\nDone')
