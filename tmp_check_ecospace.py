import pandas as pd
from pypath.io.ewemdb import _construct_ecospace_params

habitat_df = pd.DataFrame({"Group": ["Fish", "Fish"], "Patch": [1, 2], "Value": [0.8, 0.6]})
grid_df = pd.DataFrame({"PatchID": [1, 2], "Area": [10.0, 5.0], "Lon": [0.0, 0.1], "Lat": [50.0, 50.1]})
dispersal_df = pd.DataFrame({"Group": ["Fish"], "Dispersal": [0.1]})

eospace_tables = {"EcospaceGrid": grid_df, "EcospaceHabitat": habitat_df, "EcospaceDispersal": dispersal_df}

from pypath.io.ewemdb import _construct_ecospace_params
from pypath.core.params import create_rpath_params

# Assume group_names like model['Group'] from earlier
params = create_rpath_params(['Fish','Detritus'], [0,2])

group_names = params.model['Group'].tolist()

print('group_names:', group_names)
try:
    eco = _construct_ecospace_params(eospace_tables, group_names)
    print('eco:', eco)
    if eco is not None:
        print('n_patches:', eco.grid.n_patches)
        print('habitat_pref shape:', eco.habitat_preference.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
