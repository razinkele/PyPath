import pandas as pd
import pypath.io.ewemdb as e

groups_df = pd.DataFrame({"GroupID": [1], "GroupName": ["Fish"], "Type": [0], "Biomass": [2.0], "PB": [1.5], "QB": [5.0], "EE": [0.80]})
ecosim_df = pd.DataFrame({"ScenarioID": [1], "ScenarioName": ["StartMonthRel"], "StartYear": [2000], "EndYear": [2000], "NumYears": [1], "StartMonth": [4]})
forcing_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Parameter": ["ForcedPrey"], "M1": [4.0], "M2": [5.0], "M3": [6.0], "M4": [7.0], "M5": [8.0], "M6": [9.0], "M7": [10.0], "M8": [11.0], "M9": [12.0], "M10": [13.0], "M11": [14.0], "M12": [15.0]})

original = e.read_ewemdb_table

def mock_reader(filepath, table):
    if table == 'EcopathGroup':
        return groups_df
    if table in ['EcosimScenario', 'EcosimScenarios']:
        return ecosim_df
    if table in ['EcosimForcing', 'EcosimForcings', 'EcosimForcingTable']:
        return forcing_df
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
    fm = sc.get('forcing_monthly').get('ForcedPrey')
    print('monthly times:', sc.get('forcing_monthly').get('_monthly_times'))
    print('ForcedPrey df (head):')
    print(fm.head(20))
    print('first month value:', float(fm.iloc[0]['Fish']))
    print('last month value:', float(fm.iloc[-1]['Fish']))
finally:
    e.read_ewemdb_table = original
