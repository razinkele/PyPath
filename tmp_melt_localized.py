import pandas as pd
forcing_df = pd.DataFrame({'ScenarioID':[1], 'Year':[2000], 'Parameter':['ForcedPrey'], 'Fish':[None], 'Janv':[1.0], 'Fev':[1.1], 'Mar':[1.2], 'Avr':[1.3], 'Mai':[1.4], 'Juin':[1.5], 'Juil':[1.6], 'Aou':[1.7], 'Sep':[1.8], 'Oct':[1.9], 'Nov':[2.0], 'Dec':[2.1]})
import pypath.io.ewemdb as e
res = e._parse_ecosim_forcing(forcing_df)
print('parsed keys:', list(res.keys()))
print('ForcedPrey type:', type(res.get('ForcedPrey')))
print('ForcedPrey (if df):')
print(res.get('ForcedPrey'))
print('forcing_monthly (via _resample_to_monthly):')
# try to resample
r = e._resample_to_monthly(res, 2000, 1, start_month=1, use_actual_month_lengths=False)
print(r['_monthly_times'])
print(r['Value'] if 'Value' in r else r)
