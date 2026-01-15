"""Scan months advancing from an Rpath start with PyPath dynamics and detect the first
month where a per-term difference exceeds thresholds between Rpath state derivatives and
PyPath-evolved-state derivatives.

Usage: python scripts/scan_term_divergence.py --start 0 --steps 31 --abs-thresh 1e-9 --rel-thresh 1e-6
"""
import argparse
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
ECOPATH_DIR = RE / 'ecopath'
ECOSIM_DIR = RE / 'ecosim'

model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = diet_df
pypath_model = rpath(params)
scenario = rsim_scenario(pypath_model, params, years=range(1, 101))
# official PyPath run (for comparison, not strictly needed here)
out = rsim_run(scenario, method='rk4', years=range(1, 101))
py_biom = out.out_Biomass

rpath_df = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_rk4.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--steps', type=int, default=31)
parser.add_argument('--abs-thresh', type=float, default=1e-9)
parser.add_argument('--rel-thresh', type=float, default=1e-6)
args = parser.parse_args()
start = args.start
steps = args.steps
ABS_THRESH = args.abs_thresh
REL_THRESH = args.rel_thresh

NUM_GROUPS = scenario.params.NUM_GROUPS

# build params dict and request instrumentation prints
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
# instrument prints for these groups
pdict['INSTRUMENT_GROUPS'] = ['Discards', 'Seabirds']

# helper for forcing
def make_forcing(m):
    return {
        'Ftime': scenario.start_state.Ftime.copy(),
        'ForcedBio': (scenario.forcing.ForcedBio[m] if hasattr(scenario.forcing, 'ForcedBio') else np.zeros(scenario.params.NUM_GROUPS + 1)),
        'ForcedMigrate': (scenario.forcing.ForcedMigrate[m] if hasattr(scenario.forcing, 'ForcedMigrate') else np.zeros(scenario.params.NUM_GROUPS + 1)),
        'ForcedEffort': (scenario.fishing.ForcedEffort[m] if m < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS + 1)),
        'PP_forcing': (scenario.forcing.PP_forcing[m] if hasattr(scenario.forcing, 'PP_forcing') else np.ones(scenario.params.NUM_GROUPS + 1)),
        'ForcedPrey': (getattr(scenario.forcing, 'ForcedPrey', np.ones(scenario.params.NUM_GROUPS + 1))[m] if hasattr(scenario.forcing, 'ForcedPrey') else np.ones(scenario.params.NUM_GROUPS + 1)),
    }

# fishing mort
FishMort = np.zeros(scenario.params.NUM_GROUPS + 1)
for i in range(1, len(scenario.params.FishFrom)):
    grp = int(scenario.params.FishFrom[i])
    FishMort[grp] += scenario.params.FishQ[i]

fishing = {'FishingMort': FishMort, 'FishFrom': scenario.params.FishFrom, 'FishThrough': getattr(scenario.params, 'FishThrough', None), 'FishQ': getattr(scenario.params, 'FishQ', None)}

# helper compute_Q
from pypath.core.ecosim import _compute_Q_matrix as compute_Q

# prepare initial state from Rpath at 'start'
if start < 0 or start >= len(rpath_df):
    raise SystemExit(f"start {start} out of bounds for rpath data (0..{len(rpath_df)-1})")

row = rpath_df.iloc[start]
state_py = np.zeros(NUM_GROUPS + 1)
for i, g in enumerate(model_df['Group'].tolist(), start=1):
    if g in row:
        state_py[i] = float(row[g])

print(f"Scanning months {start}..{start + steps - 1} advancing PyPath from Rpath month {start}")

# indices
if 'Discards' not in pdict['spname'] or 'Seabirds' not in pdict['spname']:
    raise SystemExit('Model missing Discards or Seabirds group')
d_idx = pdict['spname'].index('Discards')
s_idx = pdict['spname'].index('Seabirds')

found = False

for step in range(0, steps):
    m = start + step
    print(f"\n--- month {m} ---")
    forcing = make_forcing(m)

    # Rpath state at month m
    row_r = rpath_df.iloc[m]
    state_r = np.zeros(NUM_GROUPS + 1)
    for i, g in enumerate(model_df['Group'].tolist(), start=1):
        if g in row_r:
            state_r[i] = float(row_r[g])

    # compute QQ matrices
    QQ_r = compute_Q(pdict, state_r, forcing)
    QQ_py = compute_Q(pdict, state_py, forcing)

    # Seabirds->Discards Q
    q_r = QQ_r[d_idx, s_idx]
    q_py = QQ_py[d_idx, s_idx]
    q_abs_diff = abs(q_py - q_r)
    q_rel_diff = q_abs_diff / (abs(q_r) + 1e-30)

    # Detritus parts for Discards (index d)
    NUM_LIVING = pdict['NUM_LIVING']
    # DetFrac reconstruction similar to deriv
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

    det_idx = d_idx - NUM_LIVING

    def det_parts(QQmat, BBstate):
        unas_input = 0.0
        for pred in range(1, NUM_LIVING + 1):
            total_consump = np.sum(QQmat[1:, pred])
            unas_input += total_consump * pdict['Unassim'][pred] * (DetFrac[pred, det_idx] if DetFrac.shape[1] > det_idx else 0)
        mort_input = 0.0
        for grp in range(1, NUM_LIVING + 1):
            mort_input += pdict.get('M0', np.zeros(NUM_GROUPS + 1))[grp] * BBstate[grp] * (DetFrac[grp, det_idx] if DetFrac.shape[1] > det_idx else 0)
        det_consumed = np.sum(QQmat[d_idx, 1 : NUM_LIVING + 1])
        decay_rate = getattr(scenario.params, 'DetDecay', np.zeros(pdict['NUM_DEAD'] + 1))
        decay = decay_rate[det_idx] * BBstate[d_idx] if len(decay_rate) > det_idx else 0
        return unas_input, mort_input, det_consumed, decay

    unas_r, mort_r, det_cons_r, decay_r = det_parts(QQ_r, state_r)
    unas_p, mort_p, det_cons_p, decay_p = det_parts(QQ_py, state_py)

    # differences
    checks = []
    checks.append(('Q_Seabirds->Discards', q_r, q_py, q_abs_diff, q_rel_diff))
    checks.append(('unas_input', unas_r, unas_p, abs(unas_p - unas_r), abs(unas_p - unas_r) / (abs(unas_r) + 1e-30)))
    checks.append(('mort_input', mort_r, mort_p, abs(mort_p - mort_r), abs(mort_p - mort_r) / (abs(mort_r) + 1e-30)))
    checks.append(('det_consumed', det_cons_r, det_cons_p, abs(det_cons_p - det_cons_r), abs(det_cons_p - det_cons_r) / (abs(det_cons_r) + 1e-30)))
    checks.append(('decay', decay_r, decay_p, abs(decay_p - decay_r), abs(decay_p - decay_r) / (abs(decay_r) + 1e-30)))

    # report any check above thresholds
    for name, val_r, val_p, abs_diff, rel_diff in checks:
        if abs_diff > ABS_THRESH and (rel_diff > REL_THRESH or abs_diff > ABS_THRESH * 100):
            print(f"FOUND divergence at month {m}: term={name} R={val_r:.12e} PY={val_p:.12e} abs_diff={abs_diff:.12e} rel_diff={rel_diff:.12e}")
            found = True
            break

    if found:
        break

    # advance PyPath state one month using RK4 (with instrumented group prints enabled)
    k1 = deriv_vector(state_py.copy(), pdict, forcing, fishing, t=m)
    s2 = state_py + 0.5 * (1.0 / 12.0) * k1
    k2 = deriv_vector(s2, pdict, forcing, fishing, t=m)
    s3 = state_py + 0.5 * (1.0 / 12.0) * k2
    k3 = deriv_vector(s3, pdict, forcing, fishing, t=m)
    s4 = state_py + (1.0 / 12.0) * k3
    k4 = deriv_vector(s4, pdict, forcing, fishing, t=m)

    state_next = state_py + (1.0 / 12.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # ensure non-negative
    state_next = np.maximum(state_next, 0.0)
    # advance
    state_py = state_next

if not found:
    print('\nNo per-term divergence above thresholds detected in scanned months')

print('\nDone')
