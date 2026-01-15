from pathlib import Path

import numpy as np
import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import (
    _build_active_link_matrix,
    _build_link_matrix,
    rsim_run,
    rsim_scenario,
)
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.params import create_rpath_params

RE = Path('tests/data/rpath_reference')
model_df = pd.read_csv(RE/'ecopath'/'model_params.csv')
diet_df = pd.read_csv(RE/'ecopath'/'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = diet_df
pypath_model = rpath(params)
scenario = rsim_scenario(pypath_model, params, years=range(1,101))
out = rsim_run(scenario, method='rk4', years=range(1,101))
py_biom = out.out_Biomass

import argparse

parser = argparse.ArgumentParser(description='Compute deriv diagnostics for Discards')
parser.add_argument('-m','--month',type=int,default=1199,help='Month index to inspect')
args = parser.parse_args()
month = args.month
state = py_biom[month].copy()

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
    'TRACE_DEBUG_GROUPS': [scenario.params.spname.index('Discards')],
    'FishFrom': getattr(scenario.params, 'FishFrom', None),
    'FishTo': getattr(scenario.params, 'FishTo', None),
    'FishQ': getattr(scenario.params, 'FishQ', None),
}
forcing = {
    'Ftime': scenario.start_state.Ftime.copy(),
    'ForcedBio': scenario.forcing.ForcedBio[month] if hasattr(scenario.forcing,'ForcedBio') else np.zeros(scenario.params.NUM_GROUPS+1),
    'ForcedMigrate': scenario.forcing.ForcedMigrate[month] if hasattr(scenario.forcing,'ForcedMigrate') else np.zeros(scenario.params.NUM_GROUPS+1),
    'ForcedEffort': (scenario.fishing.ForcedEffort[month] if month < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS+1)),
    'PP_forcing': scenario.forcing.PP_forcing[month] if hasattr(scenario.forcing,'PP_forcing') else np.ones(scenario.params.NUM_GROUPS+1),
    'ForcedPrey': getattr(scenario.forcing,'ForcedPrey', np.ones(scenario.params.NUM_GROUPS+1))[month] if hasattr(scenario.forcing,'ForcedPrey') else np.ones(scenario.params.NUM_GROUPS+1),
}

# empty fishing dict for now
fishing = {
    'FishFrom': scenario.params.FishFrom,
    'FishThrough': scenario.params.FishThrough,
    'FishQ': scenario.params.FishQ,
}

deriv = deriv_vector(state, pdict, forcing, fishing)

idx = pdict['spname'].index('Discards')
print('Deriv for Discards:', deriv[idx])
# Also print related quantities
NUM_GROUPS = pdict['NUM_GROUPS']
NUM_LIVING = pdict['NUM_LIVING']
prey = idx
pred = pdict['spname'].index('Seabirds')
print('Discards state, Bbase, ForcedPrey:', state[prey], pdict['Bbase'][prey], forcing['ForcedPrey'][prey])
print('Seabirds state, Bbase, Ftime:', state[pred], pdict['Bbase'][pred], forcing['Ftime'][pred])

# Recompute QQ (consumption) using the same logic as deriv_vector for diagnostics
QQ = np.zeros((NUM_GROUPS+1, NUM_GROUPS+1))
BB = state.copy()
Bbase = pdict['Bbase']
Ftime = forcing.get('Ftime', np.ones(NUM_GROUPS+1))
ForcedPrey = forcing.get('ForcedPrey', np.ones(NUM_GROUPS+1))

# preyYY and predYY
preyYY = np.zeros(NUM_GROUPS+1)
for i in range(1, NUM_GROUPS+1):
    if Bbase[i] > 0:
        preyYY[i] = BB[i] / Bbase[i] * ForcedPrey[i]
predYY = np.zeros(NUM_GROUPS+1)
for i in range(1, NUM_LIVING+1):
    if Bbase[i] > 0:
        predYY[i] = Ftime[i] * BB[i] / Bbase[i]

QQbase = pdict['QQbase']
VV = pdict['VV']
DD = pdict['DD']
ActiveLink = pdict['ActiveLink']
for pred in range(1, NUM_LIVING+1):
    if BB[pred] <= 0:
        continue
    for prey in range(1, NUM_GROUPS+1):
        if not ActiveLink[prey, pred]:
            continue
        if BB[prey] <= 0:
            continue
        vv = VV[prey, pred]
        dd = DD[prey, pred]
        qbase = QQbase[prey, pred]
        if qbase <= 0:
            continue
        PYY = preyYY[prey]
        PDY = predYY[pred]
        dd_term = dd / (dd - 1.0 + max(PYY, 1e-10)) if dd > 1.0 else 1.0
        vv_term = vv / (vv - 1.0 + max(PDY, 1e-10)) if vv > 1.0 else 1.0
        Q_calc = qbase * PDY * PYY * dd_term * vv_term
        QQ[prey, pred] = max(Q_calc, 0.0)

# Reconstruct DetFrac for diagnostics (mirror deriv_vector logic)
DetFrac_raw = getattr(scenario.params, 'DetFrac', None)
if DetFrac_raw is None:
    DetFrac = np.zeros((NUM_GROUPS+1, pdict['NUM_DEAD']+1))
else:
    DetFrac = np.asarray(DetFrac_raw)
    if DetFrac.ndim == 2:
        if DetFrac.shape != (NUM_GROUPS+1, pdict['NUM_DEAD']+1):
            mat = np.zeros((NUM_GROUPS+1, pdict['NUM_DEAD']+1))
            r = min(mat.shape[0], DetFrac.shape[0])
            c = min(mat.shape[1], DetFrac.shape[1])
            mat[:r, :c] = DetFrac[:r, :c]
            DetFrac = mat
    elif DetFrac.ndim == 1:
        det_from = getattr(scenario.params, 'DetFrom', None)
        det_to = getattr(scenario.params, 'DetTo', None)
        if det_from is not None and det_to is not None:
            mat = np.zeros((NUM_GROUPS+1, pdict['NUM_DEAD']+1))
            for k in range(len(DetFrac)):
                f = int(det_from[k])
                t = int(det_to[k])
                if t >= (NUM_LIVING + 1) and t <= (NUM_LIVING + pdict['NUM_DEAD']) and f >= 0 and f <= NUM_GROUPS:
                    det_col = t - NUM_LIVING
                    mat[f, det_col] += DetFrac[k]
            # include fish-derived contributions if present
            DetFrac = mat
        else:
            DetFrac = DetFrac.reshape((DetFrac.size, 1))
# ensure DetFrac shape
if not isinstance(DetFrac, np.ndarray):
    DetFrac = np.zeros((NUM_GROUPS+1, pdict['NUM_DEAD']+1))

# detritus diagnostic breakdown for Discards
d = idx
det_idx = d - NUM_LIVING
# unas_input
unas_input = 0.0
for pred in range(1, NUM_LIVING+1):
    total_consump = np.sum(QQ[1:, pred])
    unas_input += total_consump * pdict['Unassim'][pred] * (DetFrac[pred, det_idx] if DetFrac.shape[1] > det_idx else 0)
# mort_input
mort_input = 0.0
for grp in range(1, NUM_LIVING+1):
    mort_input += pdict.get('M0', np.zeros(NUM_GROUPS+1))[grp] * state[grp] * (DetFrac[grp, det_idx] if DetFrac.shape[1] > det_idx else 0)
# det_consumed
det_consumed = np.sum(QQ[d, 1: NUM_LIVING+1])
# decay
decay_rate = getattr(scenario.params, 'DetDecay', np.zeros(pdict['NUM_DEAD']+1))
decay = decay_rate[det_idx] * state[d] if len(decay_rate) > det_idx else 0

print('\nDetritus diagnostics for Discards (group index', d, 'det_idx', det_idx, '):')
print('  unas_input =', unas_input)
print('  mort_input =', mort_input)
print('  det_consumed =', det_consumed)
print('  decay =', decay)
print('  sum(inputs) - sinks =', unas_input + mort_input - det_consumed - decay)
print('  state =', state[d])

# Detailed per-group sources for Discards
print('\nPer-group contributions to inputs (predators -> unas_input):')
for pred in range(1, NUM_LIVING+1):
    total_consump = np.sum(QQ[1:, pred])
    contrib = total_consump * pdict['Unassim'][pred] * (DetFrac[pred, det_idx] if DetFrac.shape[1] > det_idx else 0)
    if contrib != 0 or total_consump != 0:
        name = pdict['spname'][pred]
        print(f"  pred={pred} name={name} total_consump={total_consump:.6e} unassim={pdict['Unassim'][pred]:.6e} DetFrac_to={DetFrac[pred,det_idx] if DetFrac.shape[1] > det_idx else 0:.6e} contrib={contrib:.6e}")

print('\nPer-group mortality contributions (mort_input):')
for grp in range(1, NUM_LIVING+1):
    contrib = pdict.get('M0', np.zeros(NUM_GROUPS+1))[grp] * state[grp] * (DetFrac[grp, det_idx] if DetFrac.shape[1] > det_idx else 0)
    if contrib != 0:
        name = pdict['spname'][grp]
        print(f"  grp={grp} name={name} M0={pdict.get('M0',np.zeros(NUM_GROUPS+1))[grp]:.6e} state={state[grp]:.6e} DetFrac_to={DetFrac[grp,det_idx] if DetFrac.shape[1] > det_idx else 0:.6e} contrib={contrib:.6e}")

print('\nDetFrac column for Discards (non-zero entries):')
for i in range(1, NUM_GROUPS+1):
    v = DetFrac[i, det_idx] if DetFrac.shape[1] > det_idx else 0
    if v != 0:
        print('  ', i, pdict['spname'][i], v)

print('\nFish discard mappings (pdict vs scenario.params):')
print('  pdict FishFrom=', pdict.get('FishFrom'))
print('  pdict FishTo=', pdict.get('FishTo'))
print('  pdict FishQ=', pdict.get('FishQ'))
print('  scenario.params FishFrom=', getattr(scenario.params, 'FishFrom', None))
print('  scenario.params FishTo=', getattr(scenario.params, 'FishTo', None))
print('  scenario.params FishQ=', getattr(scenario.params, 'FishQ', None))

# Emulate fish-derived DetFrac additions using scenario.params (for diagnostic)
fish_from = getattr(scenario.params, 'FishFrom', None)
fish_to = getattr(scenario.params, 'FishTo', None)
fish_q = getattr(scenario.params, 'FishQ', None)
if fish_from is not None and fish_to is not None and fish_q is not None:
    print('\nApplying fish-derived DetFrac contributions (diagnostic only):')
    for k in range(len(fish_from)):
        f = int(fish_from[k])
        t = int(fish_to[k])
        if t >= (NUM_LIVING + 1) and t <= (NUM_LIVING + pdict['NUM_DEAD']) and f >= 0 and f <= NUM_GROUPS:
            det_col = t - NUM_LIVING
            src_idx = f
            fish_input = float(fish_q[k]) * float(state[src_idx])
            m0_arr = pdict.get('M0', np.zeros(NUM_GROUPS+1))
            qb_arr = pdict.get('QB', np.zeros(NUM_GROUPS+1))
            unassim_arr = pdict.get('Unassim', np.zeros(NUM_GROUPS+1))
            m0_pos = max(0.0, float(m0_arr[src_idx]))
            qb_loss = 0.0 if np.isnan(qb_arr[src_idx]) else float(qb_arr[src_idx])
            source_loss = m0_pos * float(state[src_idx]) + float(state[src_idx]) * qb_loss * float(unassim_arr[src_idx])
            frac = fish_input / (source_loss + 1e-30)
            if frac > 0:
                print(f"  would add DetFrac[{src_idx},{det_col}] += {frac:.3e} (fish_from {src_idx} fish_to {t})")
                # apply to local DetFrac copy for recomputing inputs
                DetFrac[src_idx, det_col] += frac

    # recompute unas_input and mort_input with the augmented DetFrac
    unas_input2 = 0.0
    for pred in range(1, NUM_LIVING+1):
        total_consump = np.sum(QQ[1:, pred])
        unas_input2 += total_consump * pdict['Unassim'][pred] * (DetFrac[pred, det_idx] if DetFrac.shape[1] > det_idx else 0)
    mort_input2 = 0.0
    for grp in range(1, NUM_LIVING+1):
        mort_input2 += pdict.get('M0', np.zeros(NUM_GROUPS+1))[grp] * state[grp] * (DetFrac[grp, det_idx] if DetFrac.shape[1] > det_idx else 0)
    print('\nAfter applying fish-derived contributions:')
    print('  unas_input (new) =', unas_input2)
    print('  mort_input (new) =', mort_input2)
else:
    print('\nNo fish_from/fish_to/fish_q found in scenario.params; cannot add fish-derived DetFrac')

# Also find the month where Discards experienced the largest relative drop in biomass
print('\nScanning months 2..', month, 'for largest relative drop for Discards...')
min_ratio = 1.0
min_month = None
max_abs_drop = 0.0
max_drop_month = None
threshold = 1e-6
for m in range(2, month+1):
    prev = py_biom[m-1][idx]
    cur = py_biom[m][idx]
    if prev > 0:
        ratio = cur / prev
        if ratio < min_ratio:
            min_ratio = ratio
            min_month = m
    abs_drop = prev - cur
    if abs_drop > max_abs_drop:
        max_abs_drop = abs_drop
        max_drop_month = m
    # early exit if we detect collapse below threshold
    if cur < threshold:
        print('  Detected state below threshold', threshold, 'at month', m)
        break
print('  smallest ratio =', min_ratio, 'at month', min_month)
print('  largest absolute drop =', max_abs_drop, 'at month', max_drop_month)
# compute derivative at the month with largest absolute drop (if found)
if max_drop_month is not None:
    m = max_drop_month
    s = py_biom[m]
    dvec = deriv_vector(s, pdict, forcing, fishing)
    print('  deriv at largest drop month', m, ' =', dvec[idx])
else:
    print('  no drop months detected')

