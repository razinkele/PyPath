import pandas as pd
import pypath.io.ewemdb as e

groups_df = pd.DataFrame({"GroupID": [1], "GroupName": ["Fish"], "Type": [0], "Biomass": [2.0], "PB": [1.5], "QB": [5.0], "EE": [0.80]})
ecosim_df = pd.DataFrame({"ScenarioID": [1], "ScenarioName": ["FishingMonthly"], "StartYear": [2000], "EndYear": [2000], "NumYears": [1]})
fishing_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Gear": [1], "Jan": [0.5], "Feb": [0.6], "Mar": [0.7], "Apr": [0.8], "May": [0.9], "Jun": [1.0], "Jul": [1.1], "Aug": [1.2], "Sep": [1.3], "Oct": [1.4], "Nov": [1.5], "Dec": [1.6]})

orig = e.read_ewemdb_table

def mock_reader(filepath, table):
    if table == 'EcopathGroup':
        return groups_df
    if table in ['EcosimScenario', 'EcosimScenarios']:
        return ecosim_df
    if table in ['EcosimForcing', 'EcosimForcings']:
        return pd.DataFrame()
    if table in ['EcosimFishing', 'EcosimEffort', 'EcosimEffortTable']:
        return fishing_df
    return pd.DataFrame()

try:
    e.read_ewemdb_table = mock_reader
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ewemdb', delete=False) as f:
        tp = f.name
    params = e.read_ewemdb(tp, include_ecosim=True)
    import os
    os.unlink(tp)
    sc = params.ecosim['scenarios'][0]
    print('fishing_ts keys:', sc.get('fishing_ts').keys())
    fm = sc.get('fishing_monthly').get('Effort')
    print('fishing_monthly keys:', sc.get('fishing_monthly').keys())
    print('Effort df columns:', fm.columns)
    print(fm.head(5))
finally:
    e.read_ewemdb_table = orig
