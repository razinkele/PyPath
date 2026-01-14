from pypath.io.ewemdb import _parse_ecosim_forcing
import pandas as pd
forcing_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Parameter": ["ForcedPrey"], "Fish": [None], "Jan": [1.0], "Feb": [1.1], "Mar": [1.2], "Apr": [1.3], "May": [1.4], "Jun": [1.5], "Jul": [1.6], "Aug": [1.7], "Sep": [1.8], "Oct": [1.9], "Nov": [2.0], "Dec": [2.1]})
res = _parse_ecosim_forcing(forcing_df)
print('parsed keys:', list(res.keys()))
for k, v in res.items():
    print(k, type(v))
print(res.get('ForcedPrey'))
