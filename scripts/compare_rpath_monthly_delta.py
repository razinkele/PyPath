from pathlib import Path

import pandas as pd

ECOSIM_DIR = Path('tests/data/rpath_reference/ecosim')
rt = pd.read_csv(ECOSIM_DIR / 'biomass_trajectory_rk4.csv')
seab = rt['Seabirds'].values
monthly_deltas = seab[1:] - seab[:-1]
print('Rpath mean monthly delta:', monthly_deltas.mean())
print('Rpath total change:', seab[-1]-seab[0])
# load PyPath diagnostics
import pandas as pd

df = pd.read_csv('build/seabirds_diagnostics.csv')
print('PyPath mean monthly delta:', df['delta'].mean())
print('PyPath total change (observed):', df['next_biomass'].iloc[-1] - df['biomass'].iloc[0])
