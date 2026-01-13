import numpy as np
import pandas as pd
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _compute_Q_matrix, _build_link_matrix

REFERENCE_DIR = Path("tests/data/rpath_reference")
ECOPATH_DIR = REFERENCE_DIR / "ecopath"
ECOSIM_DIR = REFERENCE_DIR / "ecosim"

model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

groups = model_df["Group"].tolist()
params = create_rpath_params(groups, [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = diet_df

pypath_model = rpath(params)
scenario = rsim_scenario(pypath_model, params, years=range(1, 101))
out = rsim_run(scenario, method='rk4', years=range(1, 101))
py_biom = out.out_Biomass

rpath_df = pd.read_csv(ECOSIM_DIR / "biomass_trajectory_rk4.csv")

params_dict = {
    'NUM_GROUPS': scenario.params.NUM_GROUPS,
    'NUM_LIVING': scenario.params.NUM_LIVING,
    'NUM_DEAD': scenario.params.NUM_DEAD,
    'NUM_GEARS': scenario.params.NUM_GEARS,
    'PB': scenario.params.PBopt,
    'QB': scenario.params.FtimeQBOpt,
    'M0': scenario.params.MzeroMort.copy(),
    'Unassim': scenario.params.UnassimRespFrac,
    'ActiveLink': scenario.params.ActiveLink if hasattr(scenario.params, 'ActiveLink') else _build_link_matrix(scenario.params, scenario.params.QQ),
    'VV': _build_link_matrix(scenario.params, scenario.params.VV),
    'DD': _build_link_matrix(scenario.params, scenario.params.DD),
    'QQbase': _build_link_matrix(scenario.params, scenario.params.QQ),
    'Bbase': scenario.params.B_BaseRef,
    'PP_type': scenario.params.PP_type,
}

sidx = scenario.params.spname.index('Seabirds')
month = 952
state_py = py_biom[month].copy()

rrow = rpath_df.iloc[month]
state_r = np.zeros(scenario.params.NUM_GROUPS + 1)
state_r[0] = 1.0
for i, g in enumerate(groups, start=1):
    if g in rrow:
        state_r[i] = float(rrow[g])
    else:
        state_r[i] = float(state_py[i])

prey_indices = [22, 13, 14, 15, 12]
print(f"Month {month} — Seabirds index: {sidx} — predator biomass Py={state_py[sidx]:.6e}, R={state_r[sidx]:.6e}")
for p in prey_indices:
    name = scenario.params.spname[p]
    print(f"Prey idx {p} {name}: B_py={state_py[p]:.6e}, B_r={state_r[p]:.6e}, qbase={params_dict['QQbase'][p, sidx]:.6f}")

forcing = {
    'Ftime': scenario.start_state.Ftime.copy(),
    'ForcedBio': np.where(scenario.forcing.ForcedBio[month] > 0, scenario.forcing.ForcedBio[month], 0),
    'ForcedMigrate': scenario.forcing.ForcedMigrate[month],
    'ForcedEffort': (scenario.fishing.ForcedEffort[month] if month < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS+1)),
}

QQ_py = _compute_Q_matrix(params_dict, state_py, forcing)
QQ_r = _compute_Q_matrix(params_dict, state_r, forcing)
col_py = QQ_py[:, sidx]
col_r = QQ_r[:, sidx]

print('\nTop diffs and PDY details:')
for p in prey_indices:
    diff = col_py[p] - col_r[p]
    # compute PDY
    Bbase = params_dict['Bbase']
    preyYY_py = (state_py[p] / Bbase[p]) if Bbase[p] > 0 else 0.0
    preyYY_r = (state_r[p] / Bbase[p]) if Bbase[p] > 0 else 0.0
    PDY_py = forcing['Ftime'][sidx] * state_py[sidx] / Bbase[sidx] if Bbase[sidx] > 0 else 0.0
    PDY_r = forcing['Ftime'][sidx] * state_r[sidx] / Bbase[sidx] if Bbase[sidx] > 0 else 0.0
    print(f"{p} {scenario.params.spname[p]} Q_py={col_py[p]:.9f} Q_r={col_r[p]:.9f} diff={diff:.9f} preyYY_py={preyYY_py:.6e} preyYY_r={preyYY_r:.6e} PDY_py={PDY_py:.6e} PDY_r={PDY_r:.6e}")
