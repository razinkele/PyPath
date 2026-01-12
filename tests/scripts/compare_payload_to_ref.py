from pathlib import Path
import numpy as np
import os
os.environ['PYPATH_SILENCE_DEBUG'] = '1'
import pandas as pd
from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params
from pypath.core.ecosim import rsim_scenario, rsim_run

ECOPATH_DIR = Path("tests/data/rpath_reference/ecopath")
DIAG_DIR = Path("tests/data/rpath_reference/ecosim/diagnostics")

model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

groups = model_df['Group'].tolist()
types = model_df['Type'].tolist()
params = create_rpath_params(groups, types)
params.model = model_df
params.diet = diet_df
pypath_model = rpath(params)
scenario = rsim_scenario(pypath_model, params, years=range(1, 101))

# Instrument callback to capture RK4 stage totals
captured = []

def cb(payload):
    captured.append(payload)

scenario.params.instrument_callback = cb
scenario.params.INSTRUMENT_GROUPS = ['Seabirds']

# Run one year (12 months)
out = rsim_run(scenario, method='RK4', years=range(1, 2))

# Load reference
ref = pd.read_csv(DIAG_DIR / 'seabirds_components_rk4.csv')
row1 = ref[ref['month'] == 1].iloc[0]
ref_consumption = float(row1['consumption_by_predator'])
print('Reference consumption_by_predator month=1:', ref_consumption)

if not captured:
    print('No instrument payloads captured')
else:
    first = captured[0]
    stages = first.get('stage_consumption_totals', [])
    print('Captured stages:', stages)
    for i,st in enumerate(stages):
        val = float(st[0])
        print(f'stage {i+1} consumption total = {val:.12e} diff vs ref = {val - ref_consumption:.12e}')

print('Done')
