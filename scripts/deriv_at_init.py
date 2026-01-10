import pandas as pd
import numpy as np
from pypath.core.ecosim import rsim_scenario
from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params
from pypath.core.ecosim_deriv import deriv_vector
from pathlib import Path

REPO=Path(__file__).parent.parent
ECOPATH_DIR = REPO / 'tests' / 'data' / 'rpath_reference' / 'ecopath'
model_df = pd.read_csv(ECOPATH_DIR / 'model_params.csv')
diet_df = pd.read_csv(ECOPATH_DIR / 'diet_matrix.csv')
params = create_rpath_params(model_df['Group'].tolist(), model_df['Type'].tolist())
params.model = model_df
params.diet = diet_df
r = rpath(params)
from pypath.core.ecosim import rsim_scenario, rsim_params, rsim_state
scenario = rsim_scenario(r, params)
params_dict = {"NUM_GROUPS": scenario.params.NUM_GROUPS, "NUM_LIVING": scenario.params.NUM_LIVING, "NUM_DEAD": scenario.params.NUM_DEAD, "NUM_GEARS": scenario.params.NUM_GEARS, "PB": scenario.params.PBopt, "QB": scenario.params.FtimeQBOpt, "M0": scenario.params.MzeroMort, "Unassim": scenario.params.UnassimRespFrac, "ActiveLink": None, "VV": scenario.params.VV, "DD": scenario.params.DD, "QQbase": scenario.params.QQ, "Bbase": scenario.params.B_BaseRef, "PP_type": scenario.params.PP_type }
state = scenario.start_state.Biomass.copy()
forcing_dict = {"Ftime": scenario.start_state.Ftime.copy(), "ForcedBio": np.ones(len(state)), "ForcedMigrate": np.zeros(len(state)), "ForcedEffort": scenario.fishing.ForcedEffort[0]}
# build ActiveLink
from pypath.core.ecosim import _build_active_link_matrix
params_dict['ActiveLink'] = _build_active_link_matrix(scenario.params)

d = deriv_vector(state, params_dict, forcing_dict, {"FishingMort": np.zeros_like(state)})
print('Deriv sample (first 10):', d[:10])
print('Deriv Seabirds:', d[1])
print('Any large magnitude:', np.max(np.abs(d)))
