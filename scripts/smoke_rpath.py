from pathlib import Path
import pandas as pd
from pypath.core.params import create_rpath_params
from pypath.core.ecopath import rpath
BASE = Path('tests/data/rpath_reference')
ECOPATH_DIR = BASE / 'ecopath'
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df
m = rpath(params)
print('Groups len', len(m.Group), 'sample', m.Group[:5])
