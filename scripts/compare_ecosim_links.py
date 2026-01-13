"""
Compare pypath PreyFrom/PreyTo/QQ arrays vs R reference and print first mismatches.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario

ECOPATH_DIR = Path('tests/data/rpath_reference/ecopath')
ECOSIM_DIR = Path('tests/data/rpath_reference/ecosim')

model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df.copy()
params.diet = diet_df
rpath_model, diag = rpath(params, debug=True)

scenario = rsim_scenario(rpath_model, params, years=range(1, 101))
pp = scenario.params

rref = json.load(open(ECOSIM_DIR / 'ecosim_params.json'))
# normalize scalars if needed
for k in ("NUM_GROUPS", "NUM_LIVING", "NUM_DEAD", "NUM_GEARS"):
    if k in rref and isinstance(rref[k], list) and len(rref[k]) == 1:
        rref[k] = rref[k][0]

print('Lengths: pypath PreyFrom', len(pp.PreyFrom), 'rref PreyFrom', len(rref['PreyFrom']))
print('First 30 entries (index, p_from->p_to -> QQ, pypath name, rref name):')
for i in range(min(len(pp.PreyFrom), len(rref['PreyFrom']))):
    pf = int(pp.PreyFrom[i])
    pt = int(pp.PreyTo[i])
    q = float(pp.QQ[i])
    rpf = int(rref['PreyFrom'][i])
    rpt = int(rref['PreyTo'][i])
    rq = float(rref['QQ'][i])
    pf_name = pp.spname[pf] if pf < len(pp.spname) else f'IDX{pf}'
    pt_name = pp.spname[pt] if pt < len(pp.spname) else f'IDX{pt}'
    print(i, f'{pf}->{pt} ({pf_name}->{pt_name})', f'pypath_Q={q:.6g}', f'rref={rq:.6g}', f'rref_pair={rpf}->{rpt}')

# Print mismatches count
mismatches = [(i, int(pp.PreyFrom[i]), int(rref['PreyFrom'][i]), float(pp.QQ[i]), float(rref['QQ'][i])) for i in range(min(len(pp.PreyFrom), len(rref['PreyFrom']))) if (int(pp.PreyFrom[i]) != int(rref['PreyFrom'][i]) or abs(float(pp.QQ[i]) - float(rref['QQ'][i])) > 1e-6)]
print('\nTotal mismatches in first N:', len(mismatches))
print('Example mismatches (first 20):')
for ex in mismatches[:20]:
    print(ex)
