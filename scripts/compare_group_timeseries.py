"""Compare PyPath and Rpath group time series for key groups and save CSV."""
from pathlib import Path

import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

REFERENCE_DIR = Path("tests/data/rpath_reference")
ECOPATH_DIR = REFERENCE_DIR / "ecopath"
ECOSIM_DIR = REFERENCE_DIR / "ecosim"

model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

groups = model_df['Group'].tolist()
params = create_rpath_params(groups, [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
params.diet = diet_df

model = rpath(params)
scenario = rsim_scenario(model, params, years=range(1, 101))
out = rsim_run(scenario, method='rk4', years=range(1, 101))
py_biom = out.out_Biomass  # monthly snapshots

rpath_df = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_rk4.csv')

keys = ['Discards','Seabirds','Foragefish1','Foragefish2','OtherForagefish','OtherGroundfish']

rows = []
max_months = min(py_biom.shape[0]-1, rpath_df.shape[0])
for m in range(max_months):
    row = {'month': m}
    rrow = rpath_df.iloc[m]
    for name in keys:
        idx = scenario.params.spname.index(name)
        row[f'py_{name}'] = float(py_biom[m, idx])
        row[f'r_{name}'] = float(rrow[name]) if name in rrow else None
    rows.append(row)

out = Path('build')
out.mkdir(exist_ok=True)
df = pd.DataFrame(rows)
out_csv = out / 'group_timeseries_compare.csv'
df.to_csv(out_csv, index=False)
print(f'Wrote {out_csv} (months={max_months})')
print(df.head())
