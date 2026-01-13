"""Quick check: construct a balanced 5-group model and run rsim_run to verify output shape."""
import numpy as np
from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params
from pypath.core.ecosim import rsim_run, rsim_scenario

# Build params similar to balanced_5group_model fixture
groups = ["Phyto", "Zoo", "Fish", "Det", "Fleet"]
types = [1, 0, 0, 2, 3]
params = create_rpath_params(groups, types)
# Phytoplankton
params.model.loc[0, "Biomass"] = 100.0
params.model.loc[0, "PB"] = 10.0
params.model.loc[0, "EE"] = 0.8
# Zooplankton
params.model.loc[1, "Biomass"] = 20.0
params.model.loc[1, "PB"] = 1.0
params.model.loc[1, "QB"] = 50.0
params.model.loc[1, "EE"] = 0.5
# Fish
params.model.loc[2, "Biomass"] = 50.0
params.model.loc[2, "PB"] = 1.0
params.model.loc[2, "QB"] = 2.0
params.model.loc[2, "EE"] = 0.8
# Detritus
params.model.loc[3, "Biomass"] = 10.0
# Fleet (no biomass, gear)
params.model.loc[4, "Biomass"] = 0.0
# Set defaults
params.model["BioAcc"] = 0.0
params.model["Unassim"] = 0.2
params.model.loc[0, "Unassim"] = 0.0
params.model.loc[3, "Unassim"] = 0.0
params.model["Det"] = 1.0
# Diet
params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0]
params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0]

model = rpath(params)
scenario = rsim_scenario(model, params, years=range(1, 6))
output = rsim_run(scenario, method="RK4")
print('out_Biomass shape:', output.out_Biomass.shape)
print('expected rows:', 5*12 + 1)
assert output.out_Biomass.shape[0] == 5*12 + 1
print('OK: shape matches expected (includes initial snapshot)')
