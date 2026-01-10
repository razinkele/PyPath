from pathlib import Path
import pandas as pd
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run

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

month = 1199
print('Month:', month)
print('\nSeabirds index and name:')
print(scenario.params.spname.index('Seabirds'), 'Seabirds')

print('\nPyPath biomass for selected groups:')
for name in ['Discards','Foragefish1','Foragefish2','OtherForagefish','OtherGroundfish','Seabirds']:
    idx = scenario.params.spname.index(name)
    print(name, 'idx', idx, 'py', py_biom[month, idx])

print('\nRpath reference biomass for same groups:')
rrow = rpath_df.iloc[month]
for name in ['Discards','Foragefish1','Foragefish2','OtherForagefish','OtherGroundfish','Seabirds']:
    val = rrow[name] if name in rrow else 'MISSING'
    print(name, 'r', val)
