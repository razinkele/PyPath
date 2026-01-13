from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params
from pypath.core.ecosim import rsim_scenario, rsim_run
import numpy as np

# Build small model
groups = ["Phyto", "Zoo", "Fish", "Det", "Fleet"]
types = [1,0,0,2,3]
params = create_rpath_params(groups, types)
params.model.loc[0, "Biomass"] = 100.0
params.model.loc[0, "PB"] = 10.0
params.model.loc[0, "EE"] = 0.8
params.model.loc[1, "Biomass"] = 20.0
params.model.loc[1, "PB"] = 1.0
params.model.loc[1, "QB"] = 50.0
params.model.loc[1, "EE"] = 0.5
params.model.loc[2, "Biomass"] = 50.0
params.model.loc[2, "PB"] = 1.0
params.model.loc[2, "QB"] = 2.0
params.model.loc[2, "EE"] = 0.8
params.model.loc[3, "Biomass"] = 10.0
params.model.loc[4, "Biomass"] = 0.0
params.model["BioAcc"] = 0.0
params.model["Unassim"] = 0.2
params.model.loc[0, "Unassim"] = 0.0
params.model.loc[3, "Unassim"] = 0.0
params.model["Det"] = 1.0

model = rpath(params)
scenario = rsim_scenario(model, params, years=range(1,4))

# Ensure simple fishing link
if len(scenario.params.FishFrom) <= 1:
    scenario.params.FishFrom = np.array([0,3])
    scenario.params.FishThrough = np.array([0, int(scenario.params.NUM_LIVING + scenario.params.NUM_DEAD + 1)])
    scenario.params.FishQ = np.array([0.0, 0.5])
    scenario.params.NumFishingLinks = len(scenario.params.FishFrom) - 1

scenario.params.NoIntegrate = np.zeros(scenario.params.NUM_GROUPS + 1, dtype=int)

print('Manual estimate:')
state0 = scenario.start_state.Biomass.copy()
forcing0 = scenario.fishing.ForcedEffort[0]
man_catch = 0.0
for i in range(1,len(scenario.params.FishFrom)):
    grp = scenario.params.FishFrom[i]
    gear_group_idx = scenario.params.FishThrough[i]
    gear_idx = int(gear_group_idx - scenario.params.NUM_LIVING - scenario.params.NUM_DEAD)
    eff = forcing0[gear_idx] if 0 < gear_idx < len(forcing0) else 1.0
    man_catch += scenario.params.FishQ[i] * state0[grp] * eff / 12.0
    print('link', i, 'grp', grp, 'gear_idx', gear_idx, 'FishQ', scenario.params.FishQ[i], 'eff', eff, 'state', state0[grp])
print('manual catch', man_catch)

res = rsim_run(scenario, years=range(1,4))
print('Rs-run total catch:', np.nansum(res.out_Catch))
print('out_Catch monthly sums:', np.sum(res.out_Catch, axis=1))
print('out_Gear_Catch monthly sums:', np.sum(res.out_Gear_Catch, axis=1))
print('end biomass:', res.end_state.Biomass)
print('out_gear_catch first months values:', res.out_gear_catch[:5])
