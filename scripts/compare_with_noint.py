"""Compare RK4 derivatives when marking Discards as NoIntegrate (algebraic)

Usage: python scripts/compare_with_noint.py --month 0 --group Discards --pred Seabirds
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, _build_link_matrix, _build_active_link_matrix, _compute_Q_matrix
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

parser = argparse.ArgumentParser()
parser.add_argument('--month', type=int, default=0)
parser.add_argument('--group', type=str, default='Discards')
parser.add_argument('--pred', type=str, default='Seabirds')
args = parser.parse_args()
month = args.month
group_name = args.group
pred_name = args.pred

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

# set NoIntegrate for Discards
d_idx = pdict['spname'].index(group_name)
no = np.zeros(scenario.params.NUM_GROUPS + 1, dtype=int)
no[d_idx] = 1
pdict['NoIntegrate'] = no

# prepare state and forcing (same as compare script)
rpath_df = pd.read_csv(ECOPATH_DIR.parent / 'ecosim' / 'biomass_trajectory_rk4.csv')
row = rpath_df.iloc[month]
state_r = np.zeros(scenario.params.NUM_GROUPS + 1)
for i, g in enumerate(model_df['Group'].tolist(), start=1):
    if g in row:
        state_r[i] = float(row[g])

forcing = {
    'Ftime': scenario.start_state.Ftime.copy(),
    'ForcedBio': (scenario.forcing.ForcedBio[month] if hasattr(scenario.forcing, 'ForcedBio') else np.zeros(scenario.params.NUM_GROUPS + 1)),
    'ForcedMigrate': (scenario.forcing.ForcedMigrate[month] if hasattr(scenario.forcing, 'ForcedMigrate') else np.zeros(scenario.params.NUM_GROUPS + 1)),
    'ForcedEffort': (scenario.fishing.ForcedEffort[month] if month < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS + 1)),
    'PP_forcing': (scenario.forcing.PP_forcing[month] if hasattr(scenario.forcing, 'PP_forcing') else np.ones(scenario.params.NUM_GROUPS + 1)),
    'ForcedPrey': (getattr(scenario.forcing, 'ForcedPrey', np.ones(scenario.params.NUM_GROUPS + 1))[month] if hasattr(scenario.forcing, 'ForcedPrey') else np.ones(scenario.params.NUM_GROUPS + 1)),
}

# compute derivatives
k1 = deriv_vector(state_r.copy(), pdict, forcing, {'FishQ': scenario.params.FishQ, 'FishFrom': scenario.params.FishFrom, 'FishThrough': getattr(scenario.params, 'FishThrough', None)})

s2 = state_r + 0.5 * (1.0 / 12.0) * k1
k2 = deriv_vector(s2, pdict, forcing, {'FishQ': scenario.params.FishQ, 'FishFrom': scenario.params.FishFrom, 'FishThrough': getattr(scenario.params, 'FishThrough', None)})

s3 = state_r + 0.5 * (1.0 / 12.0) * k2
k3 = deriv_vector(s3, pdict, forcing, {'FishQ': scenario.params.FishQ, 'FishFrom': scenario.params.FishFrom, 'FishThrough': getattr(scenario.params, 'FishThrough', None)})

s4 = state_r + (1.0 / 12.0) * k3
k4 = deriv_vector(s4, pdict, forcing, {'FishQ': scenario.params.FishQ, 'FishFrom': scenario.params.FishFrom, 'FishThrough': getattr(scenario.params, 'FishThrough', None)})

s_next = state_r + (1.0 / 12.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

print('With Discards marked NoIntegrate:')
print(f'  s_next - state for Discards: {(s_next[d_idx] - state_r[d_idx]):.12e} (annual {(s_next[d_idx]-state_r[d_idx])*12:.12e})')
print('Done')
