"""Compute per-prey Q for each RK4 stage and compare to QQ final for month 952."""
import numpy as np
import pandas as pd
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _compute_Q_matrix, _build_link_matrix, deriv_vector

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
out = rsim_run(scenario, method='RK4', years=range(1, 101))
py_biom = out.out_Biomass

sidx = scenario.params.spname.index('Seabirds')
month = 952
state = py_biom[month].copy()

# params_dict for deriv and Q computation
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
    'ForcedBio': np.where(scenario.forcing.ForcedBio[month] > 0, scenario.forcing.ForcedBio[month], 0),
    'ForcedMigrate': scenario.forcing.ForcedMigrate[month],
    'ForcedEffort': (scenario.fishing.ForcedEffort[month] if month < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS+1)),
}

fishing_dict = {'FishingMort': np.zeros(scenario.params.NUM_GROUPS + 1)}

dt = 1.0 / 12.0
# compute k1, k2, k3
k1 = deriv_vector(state, params_dict, forcing, fishing_dict, 0.0)
k2 = deriv_vector(state + 0.5 * dt * k1, params_dict, forcing, fishing_dict, 0.0)
k3 = deriv_vector(state + 0.5 * dt * k2, params_dict, forcing, fishing_dict, 0.0)

stages = [state, state + 0.5 * dt * k1, state + 0.5 * dt * k2, state + dt * k3]

rows = []
QQ_stage_cols = []
for si, st in enumerate(stages):
    QQs = _compute_Q_matrix(params_dict, st, forcing)
    col = QQs[:, sidx]
    QQ_stage_cols.append(col)

QQ_final = _compute_Q_matrix(params_dict, state, forcing)[:, sidx]

for prey_idx in range(len(QQ_final)):
    name = scenario.params.spname[prey_idx] if prey_idx < len(scenario.params.spname) else str(prey_idx)
    qbase = params_dict['QQbase'][prey_idx, sidx]
    Bprey = state[prey_idx]
    vals = [float(QQ_stage_cols[si][prey_idx]) for si in range(4)]
    row = {
        'prey_idx': prey_idx,
        'prey_name': name,
        'qbase': qbase,
        'B_pre': float(Bprey),
        'Q_stage1': vals[0],
        'Q_stage2': vals[1],
        'Q_stage3': vals[2],
        'Q_stage4': vals[3],
        'Q_final': float(QQ_final[prey_idx]),
        'diff_stage4_final': float(vals[3] - QQ_final[prey_idx]),
    }
    rows.append(row)

out_df = pd.DataFrame(rows)
out_path = Path('build') / f'seabirds_stage_per_prey_month{month}.csv'
out_path.parent.mkdir(exist_ok=True)
out_df.to_csv(out_path, index=False)
print('Saved per-prey per-stage CSV to', out_path)
# print top non-zero qbase rows sorted by abs(diff)
out_df['abs_diff'] = out_df['diff_stage4_final'].abs()
print('\nTop prey where stage4 differs from Q_final:')
print(out_df.sort_values('abs_diff', ascending=False).head(10).to_string(index=False, float_format='{:0.9e}'.format))
