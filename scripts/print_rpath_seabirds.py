import pandas as pd
import numpy as np
r = pd.read_csv('tests/data/rpath_reference/ecosim/biomass_trajectory_rk4.csv')
print('Rpath Seabirds first 20:', r['Seabirds'].values[:20])
print('Mean, std:', np.mean(r['Seabirds'].values), np.std(r['Seabirds'].values))
