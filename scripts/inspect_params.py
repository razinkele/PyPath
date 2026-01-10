from pathlib import Path
import pandas as pd
RE = Path('tests/data/rpath_reference')
model_df = pd.read_csv(RE/'ecopath'/'model_params.csv')
from pypath.core.params import create_rpath_params
params = create_rpath_params(model_df['Group'].tolist(), [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
print('FishFrom', getattr(params,'FishFrom', None))
print('FishTo', getattr(params,'FishTo', None))
print('FishQ', getattr(params,'FishQ', None))
print('DetFrac (shape)', None if getattr(params,'DetFrac', None) is None else (getattr(params,'DetFrac').shape, type(getattr(params,'DetFrac'))))
print('DetFrom', getattr(params,'DetFrom', None))
print('DetTo', getattr(params,'DetTo', None))
