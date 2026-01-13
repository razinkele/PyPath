"""Advance RK4 from an Rpath monthly state with PyPath dynamics for multiple months and
report per-month Discards B, Q(Seabirds<-Discards) and deltas vs Rpath and PyPath official runs.

Usage: python scripts/multi_step_discard_reproducer.py --start 180 --steps 20
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
parser.add_argument('--start', type=int, default=180)
parser.add_argument('--steps', type=int, default=20)
args = parser.parse_args()
start = args.start
steps = args.steps

NUM_GROUPS = scenario.params.NUM_GROUPS

# build params dict
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
# instrument Discards and Seabirds
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
state = np.zeros(NUM_GROUPS + 1)
for i, g in enumerate(model_df['Group'].tolist(), start=1):
    if g in row:
        state[i] = float(row[g])

print(f"Advancing from Rpath month {start} for {steps} steps (to month {start + steps})")
print("month | B_rpath_next | B_rpath_here | B_from_rpath_next | B_py_official_next | Q_start | preyYY | PDY | delta_from_rpath")

for step in range(0, steps):
    m = start + step
    forcing = make_forcing(m)

    # quick diagnostics: max/min biomass in state
    max_b = np.max(state)
    imin = np.argmin(state)
    imax = np.argmax(state)
    print(f"DEBUG month={m} max_state idx={imax} name={pdict['spname'][imax]} value={max_b:.6e} min_state idx={imin} name={pdict['spname'][imin]} value={np.min(state):.6e}")
    if np.isnan(state).any():
        print(f"NaN detected in state at month {m}; indices: {np.where(np.isnan(state))}")
        break

    k1 = deriv_vector(state.copy(), pdict, forcing, fishing, t=m)
    if np.isnan(k1).any():
        print(f"NaN in k1 at month {m}; NaN indices: {np.where(np.isnan(k1))}")
        # print some context
        print(f"  state[d_idx]={state[d_idx]:.6e}, state[s_idx]={state[s_idx]:.6e}")
        break

    s2 = state + 0.5 * (1.0 / 12.0) * k1
    k2 = deriv_vector(s2, pdict, forcing, fishing, t=m)
    if np.isnan(k2).any():
        print(f"NaN in k2 at month {m}; NaN indices: {np.where(np.isnan(k2))}")
        break

    s3 = state + 0.5 * (1.0 / 12.0) * k2
    k3 = deriv_vector(s3, pdict, forcing, fishing, t=m)
    if np.isnan(k3).any():
        print(f"NaN in k3 at month {m}; NaN indices: {np.where(np.isnan(k3))}")
        break

    s4 = state + (1.0 / 12.0) * k3
    k4 = deriv_vector(s4, pdict, forcing, fishing, t=m)
    if np.isnan(k4).any():
        print(f"NaN in k4 at month {m}; NaN indices: {np.where(np.isnan(k4))}")
        break

    # print the per-stage derivatives for Discards to inspect where negative bias is introduced
    try:
        print(f"k1[d]={k1[d_idx]:.12e} k2[d]={k2[d_idx]:.12e} k3[d]={k3[d_idx]:.12e} k4[d]={k4[d_idx]:.12e}")
    except Exception:
        pass

    state_next = state + (1.0 / 12.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    if np.isnan(state_next).any():
        print(f"NaN in state_next computed at month {m}; NaN indices: {np.where(np.isnan(state_next))}")
        break

    # index lookups
    d_idx = pdict['spname'].index('Discards')
    s_idx = pdict['spname'].index('Seabirds')

    # print the per-stage derivatives for Discards to inspect where negative bias is introduced
    try:
        print(f"k1[d]={k1[d_idx]:.12e} k2[d]={k2[d_idx]:.12e} k3[d]={k3[d_idx]:.12e} k4[d]={k4[d_idx]:.12e}")
    except Exception:
        pass

    QQ_start = compute_Q(pdict, state, forcing)
    q_start = QQ_start[d_idx, s_idx]

    B_rpath_here = float(row[model_df['Group'].tolist()[d_idx - 1]]) if model_df['Group'].tolist()[d_idx - 1] in row else None
    # rpath next: try to read rpath_df at m+1
    if (m + 1) < len(rpath_df):
        row_next = rpath_df.iloc[m + 1]
        B_rpath_next = float(row_next[model_df['Group'].tolist()[d_idx - 1]]) if model_df['Group'].tolist()[d_idx - 1] in row_next else None
    else:
        B_rpath_next = None

    B_py_official_next = float(py_biom[m + 1][d_idx]) if (m + 1) < py_biom.shape[0] else None

    preyYY = (state[d_idx] / pdict['Bbase'][d_idx]) if pdict['Bbase'][d_idx] > 0 else 0.0
    PDY = (scenario.start_state.Ftime[s_idx] * state[s_idx] / pdict['Bbase'][s_idx]) if pdict['Bbase'][s_idx] > 0 else 0.0

    delta_from_rpath = (state_next[d_idx] - (B_rpath_next if B_rpath_next is not None else np.nan))

    print(f"{m:5d} | {B_rpath_next!s:14} | {B_rpath_here!s:13} | {state_next[d_idx]:18.6f} | {B_py_official_next!s:18} | {q_start:.6e} | {preyYY:.6e} | {PDY:.6e} | {delta_from_rpath:.6e}")

    # detailed comparison for month 1 (first step)
    if (m + 1) == 1 and (m + 1) < len(rpath_df):
        # build Rpath next state vector
        rstate_next = np.zeros(NUM_GROUPS + 1)
        rown = rpath_df.iloc[m + 1]
        for i, g in enumerate(model_df['Group'].tolist(), start=1):
            if g in rown:
                rstate_next[i] = float(rown[g])
        forcing_next = make_forcing(m + 1)
        QQ_r_next = compute_Q(pdict, rstate_next, forcing_next)
        QQ_py_next = compute_Q(pdict, state_next, forcing_next)
        q_r = QQ_r_next[d_idx, s_idx]
        q_py = QQ_py_next[d_idx, s_idx]
        qbase = pdict['QQbase'][d_idx, s_idx]
        dd = pdict['DD'][d_idx, s_idx]
        vv = pdict['VV'][d_idx, s_idx]
        PYY_r = (rstate_next[d_idx] / pdict['Bbase'][d_idx]) if pdict['Bbase'][d_idx] > 0 else 0.0
        PDY_r = (scenario.start_state.Ftime[s_idx] * rstate_next[s_idx] / pdict['Bbase'][s_idx]) if pdict['Bbase'][s_idx] > 0 else 0.0
        PYY_py = (state_next[d_idx] / pdict['Bbase'][d_idx]) if pdict['Bbase'][d_idx] > 0 else 0.0
        PDY_py = (scenario.start_state.Ftime[s_idx] * state_next[s_idx] / pdict['Bbase'][s_idx]) if pdict['Bbase'][s_idx] > 0 else 0.0
        dd_term_r = dd / (dd - 1.0 + max(PYY_r, 1e-10)) if dd > 1.0 else 1.0
        vv_term_r = vv / (vv - 1.0 + max(PDY_r, 1e-10)) if vv > 1.0 else 1.0
        dd_term_py = dd / (dd - 1.0 + max(PYY_py, 1e-10)) if dd > 1.0 else 1.0
        vv_term_py = vv / (vv - 1.0 + max(PDY_py, 1e-10)) if vv > 1.0 else 1.0
        calc_r = qbase * PDY_r * PYY_r * dd_term_r * vv_term_r
        calc_py = qbase * PDY_py * PYY_py * dd_term_py * vv_term_py
        print('\n*** Month 1 detailed link comparison: Seabirds<-Discards ***')
        print(f'  Rpath month1 Q = {q_r:.12e}  calc = {calc_r:.12e}  PYY={PYY_r:.6e} PDY={PDY_r:.6e} dd_term={dd_term_r:.6e} vv_term={vv_term_r:.6e}')
        print(f'  PyPath month1 Q = {q_py:.12e}  calc = {calc_py:.12e}  PYY={PYY_py:.6e} PDY={PDY_py:.6e} dd_term={dd_term_py:.6e} vv_term={vv_term_py:.6e}')

    # advance
    state = state_next
    # update row reference for rpath here
    if (m + 1) < len(rpath_df):
        row = rpath_df.iloc[m + 1]

print('\nDone')
