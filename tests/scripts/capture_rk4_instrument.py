from pathlib import Path
import numpy as np
import os
# silence debug
os.environ['PYPATH_SILENCE_DEBUG'] = '1'
import pandas as pd
from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params
from pypath.core.ecosim import rsim_scenario, rsim_run

ECOPATH_DIR = Path("tests/data/rpath_reference/ecopath")

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
    print('Instrument callback payload:', payload)
    captured.append(payload)

scenario.params.instrument_callback = cb
scenario.params.INSTRUMENT_GROUPS = ['Seabirds']

out = rsim_run(scenario, method='RK4', years=range(1, 2))

print('Captured callbacks count =', len(captured))
for c in captured:
    print(c)
