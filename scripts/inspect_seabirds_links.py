import json
from pathlib import Path

import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params

REPO = Path(__file__).parent.parent
ECOPATH_DIR = REPO / 'tests' / 'data' / 'rpath_reference' / 'ecopath'
ECO_DIR = REPO / 'tests' / 'data' / 'rpath_reference' / 'ecosim'

model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df

r = rpath(params)
scenario = rsim_scenario(r, params)
sp = scenario.params

# Seabirds index (1-based in params arrays)
groups = model_df['Group'].tolist()
seabirds_idx = groups.index('Seabirds')
print('Seabirds 0-based index', seabirds_idx)

# Find links where PreyTo == seabirds_idx (1-based in PreyTo)
prey_from = sp.PreyFrom
prey_to = sp.PreyTo
qq = sp.QQ

seab_links = [(i, prey_from[i], prey_to[i], qq[i]) for i in range(len(prey_from)) if prey_to[i] == seabirds_idx]
print('\nPyPath links to Seabirds (index, PreyFrom, PreyTo, QQ):')
for item in seab_links:
    print(item)

# Load R reference params
eco_json = json.load(open(ECO_DIR / 'ecosim_params.json'))
r_prey_from = eco_json['PreyFrom']
r_prey_to = eco_json['PreyTo']
r_qq = eco_json['QQ']

r_seab = [(i, r_prey_from[i], r_prey_to[i], r_qq[i]) for i in range(len(r_prey_from)) if r_prey_to[i] == seabirds_idx]
print('\nR reference links to Seabirds:')
for item in r_seab:
    print(item)

print('\nCounts: PyPath->', len(seab_links), 'R->', len(r_seab))

# Also show the QQbase matrix entries for prey->Seabirds (one snapshot)
from pypath.core.ecosim import _build_link_matrix

QQbase_mat = _build_link_matrix(sp, sp.QQ)
print('\nNon-zero QQbase entries for Seabirds predator (col seabirds_idx+1):')
col = QQbase_mat[:, seabirds_idx + 1]  # matrix includes Outside at index 0
for i, val in enumerate(col):
    if val != 0.0:
        print(' prey', i, 'val', val)

# Also show initial biomass and Bbase for Seabirds
print('\nSeabirds biomass initial state', scenario.start_state.Biomass[seabirds_idx + 1])
print('Bbase for Seabirds', sp.B_BaseRef[seabirds_idx + 1])
print('Ftime initial', scenario.start_state.Ftime[seabirds_idx + 1])
print('PP_type for Seabirds', sp.PP_type[seabirds_idx + 1])
