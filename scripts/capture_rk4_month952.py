"""Capture RK4 instrumentation over full run and report stage totals for month 952."""
import numpy as np
import pandas as pd
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _compute_Q_matrix, _build_link_matrix

ECOPATH_DIR = Path("tests/data/rpath_reference/ecopath")
ECOSIM_DIR = Path("tests/data/rpath_reference/ecosim")

model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

groups = model_df['Group'].tolist()
params = create_rpath_params(groups, [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = diet_df
pypath_model = rpath(params)
scenario = rsim_scenario(pypath_model, params, years=range(1, 101))

# Capture
captured = []

def cb(payload):
    captured.append(payload)

scenario.params.instrument_callback = cb
scenario.params.INSTRUMENT_GROUPS = ['Seabirds']

# Run all years (100 years -> 1200 months)
out = rsim_run(scenario, method='RK4', years=range(1, 101))
print('Captured payloads:', len(captured))

# select month 952 -> payload index 951
month_idx = 952
payload_idx = month_idx - 1
if payload_idx < 0 or payload_idx >= len(captured):
    print('Requested month payload not captured (index out of range)')
else:
    payload = captured[payload_idx]
    print('Payload for month', month_idx, 'keys =', payload.keys())
    stages = payload.get('stage_consumption_totals', [])
    print('Per-stage totals (each stage shows totals for requested groups):')
    for si,st in enumerate(stages):
        print(f' stage {si+1}:', st)
    # sum across stages for the Seabirds group (first entry)
    totals = [float(st[0]) for st in stages]
    stage_sum = sum(totals)
    print('\nSum across RK4 stages (Seabirds) =', stage_sum)

    # compute QQ total from state snapshot
    # build state from simulation's biomass trajectory
    py_biom = out.out_Biomass
    state_py = py_biom[month_idx].copy()
    # build params_dict for _compute_Q_matrix
    params_dict = {
        'NUM_GROUPS': scenario.params.NUM_GROUPS,
        'NUM_LIVING': scenario.params.NUM_LIVING,
        'NUM_DEAD': scenario.params.NUM_DEAD,
        'NUM_GEARS': scenario.params.NUM_GEARS,
        'PB': scenario.params.PBopt,
        'QB': scenario.params.FtimeQBOpt,
        'M0': scenario.params.MzeroMort.copy(),
        'Unassim': scenario.params.UnassimRespFrac,
        'ActiveLink': _build_link_matrix(scenario.params, scenario.params.QQ),
        'VV': _build_link_matrix(scenario.params, scenario.params.VV),
        'DD': _build_link_matrix(scenario.params, scenario.params.DD),
        'QQbase': _build_link_matrix(scenario.params, scenario.params.QQ),
        'Bbase': scenario.params.B_BaseRef,
        'PP_type': scenario.params.PP_type,
    }
    forcing = {
        'Ftime': scenario.start_state.Ftime.copy(),
        'ForcedBio': np.where(scenario.forcing.ForcedBio[month_idx] > 0, scenario.forcing.ForcedBio[month_idx], 0),
        'ForcedMigrate': scenario.forcing.ForcedMigrate[month_idx],
        'ForcedEffort': (scenario.fishing.ForcedEffort[month_idx] if month_idx < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS+1)),
    }
    QQ_py = _compute_Q_matrix(params_dict, state_py, forcing)
    sidx = scenario.params.spname.index('Seabirds')
    qq_total = float(np.nansum(QQ_py[:, sidx]))
    print('QQ computed total consumption_by_predator (Seabirds) =', qq_total)
    print('Difference stage_sum - qq_total =', stage_sum - qq_total)

    # save detailed stage totals
    outp = Path('build') / f'seabirds_rk4_stage_totals_month{month_idx}.csv'
    rows = [{'stage': i+1, 'total': float(st[0])} for i,st in enumerate(stages)]
    import pandas as pd
    pd.DataFrame(rows).to_csv(outp, index=False)
    print('Saved stage totals CSV to', outp)
