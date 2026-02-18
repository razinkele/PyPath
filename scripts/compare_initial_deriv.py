from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import _build_link_matrix, rsim_scenario
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.params import create_rpath_params

REPO = Path('.')
model_df = pd.read_csv(REPO/'tests'/'data'/'rpath_reference'/'ecopath'/'model_params.csv')
diet_df = pd.read_csv(REPO/'tests'/'data'/'rpath_reference'/'ecopath'/'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df
r = rpath(params)
scenario = rsim_scenario(r, params)
sp = scenario.params
groups = model_df['Group'].tolist()
seabirds_idx = groups.index('Seabirds')
pred_idx = seabirds_idx + 1

# Build QQbase and other matrices
QQbase = _build_link_matrix(sp, sp.QQ)
VV = _build_link_matrix(sp, sp.VV)
DD = _build_link_matrix(sp, sp.DD)
Bbase = sp.B_BaseRef

# initial state
state = scenario.start_state.Biomass.copy()
forcing = {'Ftime': scenario.start_state.Ftime.copy(), 'ForcedPrey': np.ones(sp.NUM_GROUPS + 1)}

# Compute preyYY and predYY
NUM_GROUPS = sp.NUM_GROUPS
preyYY = np.zeros(NUM_GROUPS + 1)
for i in range(1, NUM_GROUPS + 1):
    if Bbase[i] > 0:
        preyYY[i] = state[i] / Bbase[i] * forcing['ForcedPrey'][i]
predYY = np.zeros(NUM_GROUPS + 1)
for i in range(1, sp.NUM_LIVING + 1):
    if Bbase[i] > 0:
        predYY[i] = forcing['Ftime'][i] * state[i] / Bbase[i]

# Compute QQ contributions to Seabirds predator
cons_by_pred = 0.0
per_prey = []
for prey in range(1, NUM_GROUPS + 1):
    qbase = QQbase[prey, pred_idx]
    if qbase <= 0:
        continue
    PYY = preyYY[prey]
    PDY = predYY[pred_idx]
    dd = DD[prey, pred_idx]
    vv = VV[prey, pred_idx]
    dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
    vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
    Q_calc = qbase * PDY * PYY * dd_term * vv_term
    per_prey.append((prey, groups[prey-1] if prey-1 < len(groups) else f'idx{prey}', qbase, PYY, PDY, dd_term, vv_term, Q_calc))
    cons_by_pred += Q_calc

# Production via GE * consumption
PB = sp.PBopt
QB = sp.FtimeQBOpt
if QB[pred_idx] > 0:
    GE = PB[pred_idx] / QB[pred_idx]
    prod_GE = GE * cons_by_pred
else:
    prod_GE = PB[pred_idx] * state[pred_idx]
prod_PB = PB[pred_idx] * state[pred_idx]
M0 = sp.MzeroMort[pred_idx]
M0_loss = M0 * state[pred_idx]

# deriv
deriv = deriv_vector(state, {'NUM_GROUPS': sp.NUM_GROUPS, 'NUM_LIVING': sp.NUM_LIVING, 'NUM_DEAD': sp.NUM_DEAD, 'NUM_GEARS': sp.NUM_GEARS, 'PB': sp.PBopt, 'QB': sp.FtimeQBOpt, 'M0': sp.MzeroMort, 'Unassim': sp.UnassimRespFrac, 'ActiveLink': _build_link_matrix(sp, sp.QQ), 'VV': _build_link_matrix(sp, sp.VV), 'DD': _build_link_matrix(sp, sp.DD), 'QQbase': QQbase, 'Bbase': sp.B_BaseRef, 'PP_type': sp.PP_type}, forcing, {'FishingMort': np.zeros(sp.NUM_GROUPS + 1)})

print('Seabirds initial state: biomass=', state[pred_idx])
print('Per-prey QQbase and Q_calc:')
for item in per_prey:
    print(' prey_idx', item[0], item[1], 'qbase', item[2], 'PYY', item[3], 'PDY', item[4], 'dd_term', item[5], 'vv_term', item[6], 'Q_calc', item[7])

print('\nsum consumption_by_pred (QQ col):', cons_by_pred)
print('GE (PB/QB):', GE)
print('prod (GE*cons):', prod_GE)
print('prod (PB*B):', prod_PB)
print('M0:', M0, 'M0_loss M0*B:', M0_loss)
print('derivative from deriv_vector:', deriv[pred_idx])
print('difference prod_PB - M0_loss:', prod_PB - M0_loss)
