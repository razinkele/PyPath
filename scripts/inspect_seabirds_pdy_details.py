"""Print Ftime/PDY sampling details for Seabirds in worst month and neighbors."""
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

sidx = scenario.params.spname.index('Seabirds')

# pick worst month from comparison
csv = Path('build/qq_comparison_monthly.csv')
df = pd.read_csv(csv)
worst = int(df.sort_values('seab_col_max', ascending=False).iloc[0]['month'])
print('worst month:', worst)

for m in (worst-1, worst, worst+1):
    if m < 0:
        continue
    print('\n--- Month', m, '---')
    state_py = py_biom[m].copy()
    rrow = rpath_df.iloc[m]
    state_r = np.zeros(scenario.params.NUM_GROUPS + 1)
    state_r[0] = 1.0
    for i, g in enumerate(groups, start=1):
        if g in rrow:
            state_r[i] = float(rrow[g])
        else:
            state_r[i] = float(state_py[i])

    Ftime_start = scenario.start_state.Ftime[sidx]
    Bbase_s = scenario.params.B_BaseRef[sidx]
    Bpy = state_py[sidx]
    Br = state_r[sidx]
    PDY_py = Ftime_start * Bpy / Bbase_s if Bbase_s > 0 else 0.0
    PDY_r = Ftime_start * Br / Bbase_s if Bbase_s > 0 else 0.0
    print('Ftime_start[Seabirds]=', Ftime_start)
    print('Bbase[Seabirds]=', Bbase_s)
    print('B_py[Seabirds]=', Bpy)
    print('B_r[Seabirds]=', Br)
    print('B_diff (py - r)=', Bpy - Br)
    print('PDY_py=', PDY_py)
    print('PDY_r=', PDY_r)
    print('PDY_diff=', PDY_py - PDY_r)

# Also show Ftime vector sample and any forcing fields for visibility
print('\nFtime sample (first 10):', scenario.start_state.Ftime[:10])
# if scenario.forcing has any time-varying fields print keys
print('\nScenario.forcing has attributes:', [a for a in dir(scenario.forcing) if not a.startswith('_')][:20])
try:
    print('scenario.forcing.ForcedBio shape:', np.array(scenario.forcing.ForcedBio).shape)
except Exception:
    pass
