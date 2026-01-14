import pandas as pd
import pypath.io.ewemdb as e
forcing_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Parameter": ["ForcedPrey"], "M1": [4.0], "M2": [5.0], "M3": [6.0], "M4": [7.0], "M5": [8.0], "M6": [9.0], "M7": [10.0], "M8": [11.0], "M9": [12.0], "M10": [13.0], "M11": [14.0], "M12": [15.0]})
res = e._parse_ecosim_forcing(forcing_df, start_month=4, month_label_relative=True)
print('parsed times:', res['_times'])
print('ForcedPrey df:')
print(res.get('ForcedPrey'))
# Show melted intermediate steps by reusing module logic
import pandas as pd
month_name_map = {k: v for k,v in [('jan',1),('feb',2),('mar',3),('apr',4),('may',5),('jun',6),('jul',7),('aug',8),('sep',9),('oct',10),('nov',11),('dec',12)]}
month_cols = []
for c in forcing_df.columns:
    cl = c.lower()
    if cl in month_name_map:
        month_cols.append((c, month_name_map[cl]))
    elif cl.startswith('m') and cl[1:].isdigit() and 1 <= int(cl[1:]) <= 12:
        month_cols.append((c, int(cl[1:])))
value_vars = [c for c,_ in month_cols]
time_col = next((c for c in ['Year','Time'] if c in forcing_df.columns), None)
other_cols = [c for c in forcing_df.columns if c not in value_vars and c != time_col]
id_vars = [time_col] + other_cols if time_col is not None else other_cols
melted = forcing_df.melt(id_vars=id_vars, value_vars=value_vars, var_name='MonthCol', value_name='Value')
print('Melted:')
print(melted)
def month_index_from_label(lbl):
    l = str(lbl).lower()
    if l.startswith('m') and l[1:].isdigit():
        return int(l[1:])
    if l.startswith('month') and l[5:].isdigit():
        return int(l[5:])
    return None
melted['MonthIdx'] = melted['MonthCol'].apply(month_index_from_label)
print('With MonthIdx:')
print(melted)
# compute Month using rel_to_actual
start=4
melted['Month'] = melted.apply(lambda r: ((int(r['MonthIdx']) - 1 + (start - 1)) % 12) + 1 if pd.notna(r['MonthIdx']) else r['MonthCol'], axis=1)
print('With Month mapped:')
print(melted[['Year','MonthIdx','Month','Value']])
# compute fractional years

def to_frac_year(r):
    y = float(r['Year'])
    idx = int(r['MonthIdx'])
    mnum = ((idx - 1 + (start - 1)) % 12) + 1
    year_offset = (idx - 1 + (start - 1)) // 12
    return (y + year_offset) + (float(mnum) - 1.0) / 12.0
melted['_TimeFrac'] = melted.apply(to_frac_year, axis=1)
print('With _TimeFrac:')
print(melted[['MonthCol','MonthIdx','_TimeFrac','Value']])
# pivot
pivot = melted.pivot_table(index='_TimeFrac', columns='Parameter', values='Value', aggfunc='mean')
print('Pivot:')
print(pivot)
from pypath.io.ewemdb import _resample_to_monthly
r = _resample_to_monthly(res, 2000, 1, start_month=4, use_actual_month_lengths=False)
print('Resampled monthly times:', r.get('_monthly_times'))
print('Resampled ForcedPrey:')
print(r.get('ForcedPrey'))
