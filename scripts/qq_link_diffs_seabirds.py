"""Inspect per-link differences for Seabirds column for a selected month."""
import numpy as np
import pandas as pd
from pathlib import Path
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run, _compute_Q_matrix, _build_active_link_matrix, _build_link_matrix

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
    'ActiveLink': _build_active_link_matrix(scenario.params),
    'VV': _build_link_matrix(scenario.params, scenario.params.VV),
    'DD': _build_link_matrix(scenario.params, scenario.params.DD),
    'QQbase': _build_link_matrix(scenario.params, scenario.params.QQ),
    'Bbase': scenario.params.B_BaseRef,
    'PP_type': scenario.params.PP_type,
}

sidx = scenario.params.spname.index('Seabirds')

# load comparison summary
csv = Path('build/qq_comparison_monthly.csv')
df = pd.read_csv(csv)
worst = int(df.sort_values('seab_col_max', ascending=False).iloc[0]['month'])
print('worst month:', worst)

month = worst
state_py = py_biom[month].copy()
forcing = {
    'Ftime': scenario.start_state.Ftime.copy(),
    'ForcedBio': np.where(scenario.forcing.ForcedBio[month] > 0, scenario.forcing.ForcedBio[month], 0),
    'ForcedMigrate': scenario.forcing.ForcedMigrate[month],
    'ForcedEffort': (scenario.fishing.ForcedEffort[month] if month < len(scenario.fishing.ForcedEffort) else np.ones(scenario.params.NUM_GEARS+1)),
}
QQ_py = _compute_Q_matrix(params_dict, state_py, forcing)

rrow = rpath_df.iloc[month]
state_r = np.zeros(scenario.params.NUM_GROUPS + 1)
state_r[0] = 1.0
for i, g in enumerate(groups, start=1):
    if g in rrow:
        state_r[i] = float(rrow[g])
    else:
        state_r[i] = float(state_py[i])
QQ_r = _compute_Q_matrix(params_dict, state_r, forcing)

col_py = QQ_py[:, sidx]
col_r = QQ_r[:, sidx]
diff = np.abs(col_py - col_r)
# report top links
order = np.argsort(diff)[::-1]
print('Top differences in Seabirds column (prey index, prey name, py, r, diff):')
for idx in order[:10]:
    name = scenario.params.spname[idx] if idx < len(scenario.params.spname) else str(idx)
    print(idx, name, col_py[idx], col_r[idx], diff[idx])

# also print QQbase values for Seabirds column
QQbase = params_dict['QQbase']
print('\nQQbase column for Seabirds:')
for idx in order[:10]:
    print(idx, scenario.params.spname[idx], QQbase[idx, sidx])

# Detailed per-link breakdown for the worst month (prey, qbase, preyYY_py, preyYY_r, PDY_py, PDY_r, dd, vv, Q_py, Q_r):
print('\nDetailed link breakdown:')
for idx in order[:10]:
    prey = idx
    qbase = QQbase[prey, sidx]
    Bprey_py = state_py[prey]
    Bprey_r = state_r[prey]
    Bbase = params_dict['Bbase']
    preyYY_py = (Bprey_py / Bbase[prey]) if Bbase[prey] > 0 else 0.0
    preyYY_r = (Bprey_r / Bbase[prey]) if Bbase[prey] > 0 else 0.0
    PDY_py = forcing['Ftime'][sidx] * state_py[sidx] / Bbase[sidx] if Bbase[sidx] > 0 else 0.0
    PDY_r = forcing['Ftime'][sidx] * state_r[sidx] / Bbase[sidx] if Bbase[sidx] > 0 else 0.0
    dd = params_dict['DD'][prey, sidx] if 'DD' in params_dict else 1.0
    vv = params_dict['VV'][prey, sidx] if 'VV' in params_dict else 1.0
    dd_term_py = dd / (dd - 1.0 + max(preyYY_py, 1e-10)) if dd > 1.0 else 1.0
    dd_term_r = dd / (dd - 1.0 + max(preyYY_r, 1e-10)) if dd > 1.0 else 1.0
    vv_term_py = vv / (vv - 1.0 + max(PDY_py, 1e-10)) if vv > 1.0 else 1.0
    vv_term_r = vv / (vv - 1.0 + max(PDY_r, 1e-10)) if vv > 1.0 else 1.0
    Q_py = max(qbase * PDY_py * preyYY_py * dd_term_py * vv_term_py, 0.0)
    Q_r = max(qbase * PDY_r * preyYY_r * dd_term_r * vv_term_r, 0.0)
    print(prey, scenario.params.spname[prey], f"qbase={qbase:.6f}", f"preyYY_py={preyYY_py:.6e}", f"preyYY_r={preyYY_r:.6e}", f"PDY_py={PDY_py:.6e}", f"PDY_r={PDY_r:.6e}", f"dd={dd:.3f}", f"vv={vv:.3f}", f"Q_py={Q_py:.9f}", f"Q_r={Q_r:.9f}")
