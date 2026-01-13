"""One-step RK4 diagnostic for Discards starting from Rpath state using PyPath dynamics.

Usage: python scripts/one_step_discard_breakdown.py -m 195

Prints: start B, next-step B (RK4), delta, and labeled breakdown of derivative contributions
for Discards (predation sink, unas_input, mort_input, det_consumed, decay, DetFrac contributors,
and the Seabirds->Discards Q link terms).
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
out = rsim_run(scenario, method='rk4', years=range(1, 101))
py_biom = out.out_Biomass

rpath_df = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_rk4.csv')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--month', type=int, default=195)
args = parser.parse_args()
month = args.month

# prepare state from Rpath row
if month < 0 or month >= len(rpath_df):
    raise SystemExit(f"Month {month} out of bounds for rpath data (0..{len(rpath_df)-1})")

row = rpath_df.iloc[month]
# build state vector sized NUM_GROUPS + 1 (index 0 unused)
NUM_GROUPS = scenario.params.NUM_GROUPS
state_r = np.zeros(NUM_GROUPS + 1)
# use groups from the loaded ecopath model CSV (scenario.params may not carry a 'model' attribute)
groups = model_df['Group'].tolist()
for i, g in enumerate(groups, start=1):
    if g in row:
        state_r[i] = float(row[g])

# build params dict (same as deriv scripts)
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
# Request instrumentation for Discards and Seabirds
pdict['INSTRUMENT_GROUPS'] = ['Discards', 'Seabirds']

# forcing for that month
forcing = {
    'Ftime': scenario.start_state.Ftime.copy(),
    'ForcedBio': (scenario.forcing.ForcedBio[month] if hasattr(scenario.forcing, 'ForcedBio') else np.zeros(scenario.params.NUM_GROUPS + 1)),
    'ForcedMigrate': (scenario.forcing.ForcedMigrate[month] if hasattr(scenario.forcing, 'ForcedMigrate') else np.zeros(scenario.params.NUM_GROUPS + 1)),
    'ForcedEffort': (scenario.fishing.ForcedEffort[month] if month < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS + 1)),
    'PP_forcing': (scenario.forcing.PP_forcing[month] if hasattr(scenario.forcing, 'PP_forcing') else np.ones(scenario.params.NUM_GROUPS + 1)),
    'ForcedPrey': (getattr(scenario.forcing, 'ForcedPrey', np.ones(scenario.params.NUM_GROUPS + 1))[month] if hasattr(scenario.forcing, 'ForcedPrey') else np.ones(scenario.params.NUM_GROUPS + 1)),
}

# fishing dict (we keep base FishMort computed like other helpers)
FishMort = np.zeros(scenario.params.NUM_GROUPS + 1)
for i in range(1, len(scenario.params.FishFrom)):
    grp = int(scenario.params.FishFrom[i])
    FishMort[grp] += scenario.params.FishQ[i]

fishing = {'FishingMort': FishMort, 'FishFrom': scenario.params.FishFrom, 'FishThrough': getattr(scenario.params, 'FishThrough', None), 'FishQ': getattr(scenario.params, 'FishQ', None)}

# helper to compute QQ for a state
from pypath.core.ecosim import _compute_Q_matrix as compute_Q

# Derivative at Rpath state
deriv_r = deriv_vector(state_r.copy(), pdict, forcing, fishing, t=month)
# Note: deriv_vector returns a numpy vector; sometimes it prints extra DETRITUS traces

# perform RK4 one step (monthly dt)
dt = 1.0 / 12.0
k1 = deriv_vector(state_r.copy(), pdict, forcing, fishing, t=month)
s2 = state_r + 0.5 * dt * k1
k2 = deriv_vector(s2, pdict, forcing, fishing, t=month)
s3 = state_r + 0.5 * dt * k2
k3 = deriv_vector(s3, pdict, forcing, fishing, t=month)
s4 = state_r + dt * k3
k4 = deriv_vector(s4, pdict, forcing, fishing, t=month)

s_next = state_r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# identify Discards and Seabirds indices
if 'Discards' not in pdict['spname'] or 'Seabirds' not in pdict['spname']:
    raise SystemExit('Model does not contain Discards or Seabirds groups')

d_idx = pdict['spname'].index('Discards')
s_idx = pdict['spname'].index('Seabirds')

print(f"Month {month}: Discards start B = {state_r[d_idx]:.6f}")
print(f"Month {month}: Discards next-step B = {s_next[d_idx]:.6f} (delta = {s_next[d_idx]-state_r[d_idx]:.6e})")

# compute QQ at start and after step for diagnostic
QQ_start = compute_Q(pdict, state_r, forcing)
QQ_next = compute_Q(pdict, s_next, forcing)

# print Seabirds->Discards link (pred=Seabirds, prey=Discards): Q[prey,pred]
q_start = QQ_start[d_idx, s_idx]
q_next = QQ_next[d_idx, s_idx]
print(f"Seabirds->Discards Q start = {q_start:.6e}, Q next = {q_next:.6e}, diff = {q_next - q_start:.6e}")

# print preyYY, PDY, dd/vv terms for the Seabirds<-Discards link at start
Bbase = pdict['Bbase']
VV = pdict['VV']
DD = pdict['DD']
QQbase = pdict['QQbase']

PYY = (state_r[d_idx] / Bbase[d_idx]) if Bbase[d_idx] > 0 else 0.0
PDY = (scenario.start_state.Ftime[s_idx] * state_r[s_idx] / Bbase[s_idx]) if Bbase[s_idx] > 0 else 0.0
qbase = QQbase[d_idx, s_idx]
vv = VV[d_idx, s_idx]
dd = DD[d_idx, s_idx]

dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
calc_Q = qbase * PDY * PYY * dd_term * vv_term
print('\nSeabirds<-Discards term breakdown (start state):')
print(f"  preyYY (PYY) = {PYY:.6e}")
print(f"  predYY (PDY) = {PDY:.6e}")
print(f"  dd = {dd}, vv = {vv}")
print(f"  dd_term = {dd_term:.6e}, vv_term = {vv_term:.6e}")
print(f"  qbase = {qbase:.6e}, calc_Q = {calc_Q:.6e}")

# also compute decomposition using the RK4 next-step state (s_next)
PYY_next = (s_next[d_idx] / Bbase[d_idx]) if Bbase[d_idx] > 0 else 0.0
PDY_next = (scenario.start_state.Ftime[s_idx] * s_next[s_idx] / Bbase[s_idx]) if Bbase[s_idx] > 0 else 0.0
dd_term_next = dd / (dd - 1.0 + max(PYY_next, 1e-10)) if dd > 1.0 else 1.0
vv_term_next = vv / (vv - 1.0 + max(PDY_next, 1e-10)) if vv > 1.0 else 1.0
calc_Q_next = qbase * PDY_next * PYY_next * dd_term_next * vv_term_next
print('\nSeabirds<-Discards term breakdown (next state):')
print(f"  preyYY (PYY) = {PYY_next:.6e}")
print(f"  predYY (PDY) = {PDY_next:.6e}")
print(f"  dd_term = {dd_term_next:.6e}, vv_term = {vv_term_next:.6e}")
print(f"  qbase = {qbase:.6e}, calc_Q_next = {calc_Q_next:.6e}")
print(f"  calc diff = {calc_Q_next - calc_Q:.6e}")

# detritus detailed breakdown for Discards at start (reusing compute_deriv_discards logic)
NUM_GROUPS = pdict['NUM_GROUPS']
NUM_LIVING = pdict['NUM_LIVING']

# recompute QQ (already have QQ_start)
QQ = QQ_start

# Reconstruct DetFrac as deriv does
DetFrac_raw = getattr(scenario.params, 'DetFrac', None)
if DetFrac_raw is None:
    DetFrac = np.zeros((NUM_GROUPS + 1, pdict['NUM_DEAD'] + 1))
else:
    DetFrac = np.asarray(DetFrac_raw)
    if DetFrac.ndim == 2:
        if DetFrac.shape != (NUM_GROUPS + 1, pdict['NUM_DEAD'] + 1):
            mat = np.zeros((NUM_GROUPS + 1, pdict['NUM_DEAD'] + 1))
            r = min(mat.shape[0], DetFrac.shape[0])
            c = min(mat.shape[1], DetFrac.shape[1])
            mat[:r, :c] = DetFrac[:r, :c]
            DetFrac = mat
    elif DetFrac.ndim == 1:
        det_from = getattr(scenario.params, 'DetFrom', None)
        det_to = getattr(scenario.params, 'DetTo', None)
        if det_from is not None and det_to is not None:
            mat = np.zeros((NUM_GROUPS + 1, pdict['NUM_DEAD'] + 1))
            for k in range(len(DetFrac)):
                f = int(det_from[k])
                t = int(det_to[k])
                if t >= (NUM_LIVING + 1) and t <= (NUM_LIVING + pdict['NUM_DEAD']) and f >= 0 and f <= NUM_GROUPS:
                    det_col = t - NUM_LIVING
                    mat[f, det_col] += DetFrac[k]
            DetFrac = mat
        else:
            DetFrac = DetFrac.reshape((DetFrac.size, 1))

# detritus diagnostics for Discards
d = d_idx
det_idx = d - NUM_LIVING
unas_input = 0.0
for pred in range(1, NUM_LIVING + 1):
    total_consump = np.sum(QQ[1:, pred])
    unas_input += total_consump * pdict['Unassim'][pred] * (DetFrac[pred, det_idx] if DetFrac.shape[1] > det_idx else 0)
mort_input = 0.0
for grp in range(1, NUM_LIVING + 1):
    mort_input += pdict.get('M0', np.zeros(NUM_GROUPS + 1))[grp] * state_r[grp] * (DetFrac[grp, det_idx] if DetFrac.shape[1] > det_idx else 0)
det_consumed = np.sum(QQ[d, 1: NUM_LIVING + 1])
decay_rate = getattr(scenario.params, 'DetDecay', np.zeros(pdict['NUM_DEAD'] + 1))
decay = decay_rate[det_idx] * state_r[d] if len(decay_rate) > det_idx else 0

print('\nDetritus diagnostics for Discards (start state):')
print(f"  unas_input = {unas_input:.6e}")
print(f"  mort_input = {mort_input:.6e}")
print(f"  det_consumed = {det_consumed:.6e}")
print(f"  decay = {decay:.6e}")
print(f"  sum(inputs) - sinks = {unas_input + mort_input - det_consumed - decay:.6e}")

print('\nPer-group contributions to unas_input (predators->unas_input):')
for pred in range(1, NUM_LIVING + 1):
    total_consump = np.sum(QQ[1:, pred])
    contrib = total_consump * pdict['Unassim'][pred] * (DetFrac[pred, det_idx] if DetFrac.shape[1] > det_idx else 0)
    if contrib != 0 or total_consump != 0:
        name = pdict['spname'][pred]
        print(f"  pred={pred} name={name} total_consump={total_consump:.6e} unassim={pdict['Unassim'][pred]:.6e} DetFrac_to={DetFrac[pred,det_idx] if DetFrac.shape[1] > det_idx else 0:.6e} contrib={contrib:.6e}")

print('\nPer-group mortality contributions (mort_input):')
for grp in range(1, NUM_LIVING + 1):
    contrib = pdict.get('M0', np.zeros(NUM_GROUPS + 1))[grp] * state_r[grp] * (DetFrac[grp, det_idx] if DetFrac.shape[1] > det_idx else 0)
    if contrib != 0:
        name = pdict['spname'][grp]
        print(f"  grp={grp} name={name} M0={pdict.get('M0',np.zeros(NUM_GROUPS+1))[grp]:.6e} state={state_r[grp]:.6e} DetFrac_to={DetFrac[grp,det_idx] if DetFrac.shape[1] > det_idx else 0:.6e} contrib={contrib:.6e}")

# print fish-derived DetFrac candidates
fish_from = getattr(scenario.params, 'FishFrom', None)
fish_to = getattr(scenario.params, 'FishTo', None)
fish_q = getattr(scenario.params, 'FishQ', None)
if fish_from is not None and fish_to is not None and fish_q is not None:
    print('\nFish discard mappings (first five entries):')
    for k in range(min(5, len(fish_from))):
        print(f"  {k}: from={fish_from[k]} to={fish_to[k]} q={fish_q[k]}")

print('\nDone')
