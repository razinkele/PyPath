import pandas as pd

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params

model_df = pd.read_csv('tests/data/rpath_reference/ecopath/model_params.csv')
groups = model_df['Group'].tolist()
params = create_rpath_params(groups, [int(t) for t in model_df['Type'].tolist()])
params.model = model_df
model = rpath(params)
scenario = rsim_scenario(model, params, years=range(1, 101))
sp = scenario.params.spname
idx = sp.index('Discards')
print('Discards idx', idx)
print('start_state.Biomass[Discards]=', scenario.start_state.Biomass[idx])
for name in ['Seabirds','Foragefish1','Foragefish2','OtherForagefish','OtherGroundfish']:
    print(name, scenario.start_state.Biomass[sp.index(name)])
