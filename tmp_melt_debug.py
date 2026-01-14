import pandas as pd
from pypath.io.ewemdb import _parse_ecosim_forcing
forcing_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Parameter": ["ForcedPrey"], "Fish": [None], "Jan": [1.0], "Feb": [1.1], "Mar": [1.2], "Apr": [1.3], "May": [1.4], "Jun": [1.5], "Jul": [1.6], "Aug": [1.7], "Sep": [1.8], "Oct": [1.9], "Nov": [2.0], "Dec": [2.1]})
from pypath.io.ewemdb import _parse_ecosim_forcing
# replicate the month_cols branch logic
print('original df columns:', list(forcing_df.columns))
# Reuse internal functions by re-importing module
import pypath.io.ewemdb as e
res = e._parse_ecosim_forcing(forcing_df)
# Show internal melt step by invoking small helper logic
# We'll re-run the melt here
month_name_map = {k: v for k,v in [('jan',1),('feb',2),('mar',3),('apr',4),('may',5),('jun',6),('jul',7),('aug',8),('sep',9),('oct',10),('nov',11),('dec',12)]}
month_cols = []
for c in forcing_df.columns:
    cl = c.lower()
    if cl in month_name_map:
        month_cols.append((c, month_name_map[cl]))
value_vars = [c for c,_ in month_cols]
print('value_vars', value_vars)
time_col = next((c for c in ['Year','Time'] if c in forcing_df.columns), None)
other_cols = [c for c in forcing_df.columns if c not in value_vars and c != time_col]
id_vars = [time_col] + other_cols if time_col is not None else other_cols
print('id_vars', id_vars)
melted = forcing_df.melt(id_vars=id_vars, value_vars=value_vars, var_name='MonthCol', value_name='Value')
print(melted.head(20))
