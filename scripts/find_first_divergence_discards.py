"""Find earliest month where PyPath Discards diverges noticeably from Rpath."""
from pathlib import Path
import pandas as pd

from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run

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

d_name = 'Discards'
if d_name not in scenario.params.spname:
    raise SystemExit('No Discards in model')

d_idx = scenario.params.spname.index(d_name)

n = min(len(py_biom), len(rpath_df))

first = None
for m in range(n):
    b_py = float(py_biom[m][d_idx])
    row = rpath_df.iloc[m]
    b_r = float(row[d_name]) if d_name in row else float('nan')
    if b_r == 0:
        rel = float('inf') if b_py != 0 else 0
    else:
        rel = (b_py - b_r) / b_r
    if rel > 0.05 or (b_py - b_r) > 1e-3:
        first = m
        print(f"First divergence at month {m}: b_py={b_py:.6f}, b_r={b_r:.6f}, rel={rel:.6f}")
        break

if first is None:
    print('No divergence found under thresholds')
else:
    print('\nSamples around divergence (m-3..m+3):')
    for mm in range(max(0, first - 3), min(n, first + 4)):
        b_py = float(py_biom[mm][d_idx])
        b_r = float(rpath_df.iloc[mm][d_name])
        print(f"  month {mm}: py={b_py:.6f}, r={b_r:.6f}, diff={b_py - b_r:.6e}")
