from pypath.core.ecopath import rpath
from pypath.core.params import create_rpath_params
import pandas as pd
# Load params
model_df = pd.read_csv(r"tests/data/rpath_reference/ecopath/model_params.csv")
diet_df = pd.read_csv(r"tests/data/rpath_reference/ecopath/diet_matrix.csv")
groups = model_df["Group"].tolist()
types = model_df["Type"].tolist()
params = create_rpath_params(groups, types)
params.model = model_df
params.diet = diet_df
pypath_model = rpath(params)
from pypath.core.ecosim import rsim_scenario, rsim_run
scenario = rsim_scenario(pypath_model, params, years=range(1, 101))
out = rsim_run(scenario, method="RK4", years=range(1, 2))
print('done')
