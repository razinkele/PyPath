import pandas as pd
import pypath.io.ewemdb as e

groups_df = pd.DataFrame({"GroupID": [1,2], "GroupName": ["Fish","Detritus"], "Type": [0,2], "Biomass": [2.0,100.0], "PB": [1.5,0.0], "QB": [5.0,0.0], "EE": [0.8,0.0]})
ecosim_df = pd.DataFrame({"ScenarioID": [1], "ScenarioName": ["TestScenario"], "StartYear": [2000], "EndYear": [2005], "NumYears": [6], "Description": ["Test Ecosim scenario"]})
forcing_df = pd.DataFrame({"ScenarioID": [1,1,1,1], "Time": [0,0,1,1], "Parameter": ["ForcedPrey","ForcedMort","ForcedPrey","ForcedMort"], "Group":["Fish","Fish","Fish","Fish"], "Value": [1.0,1.0,0.9,1.0]})
fishing_df = pd.DataFrame({"ScenarioID": [1], "Time": [0], "Gear": [1], "Effort": [0.5]})
frate_df = pd.DataFrame({"ScenarioID": [1,1], "Year": [2000,2001], "Group": ["Fish","Fish"], "FRate": [0.1,0.2]})
catch_yr_df = pd.DataFrame({"ScenarioID": [1,1], "Year": [2000,2001], "Group": ["Fish","Fish"], "Catch": [5.0,6.0]})
habitat_df = pd.DataFrame({"Group": ["Fish","Fish"], "Patch": [1,2], "Value": [0.8,0.6]})
grid_df = pd.DataFrame({"PatchID": [1,2], "Area": [10.0,5.0], "Lon": [0.0,0.1], "Lat": [50.0,50.1]})
dispersal_df = pd.DataFrame({"Group": ["Fish"], "Dispersal": [0.1]})

orig = e.read_ewemdb_table

def mock_reader(filepath, table):
    if table == 'EcopathGroup':
        return groups_df
    if table in ['EcosimScenario','EcosimScenarios']:
        return ecosim_df
    if table in ['EcosimForcing','EcosimForcings']:
        return forcing_df
    if table in ['EcosimFishing','EcosimEffort']:
        return fishing_df
    if table in ['EcosimFRate','EcosimFRateTable']:
        return frate_df
    if table in ['EcosimCatch','EcosimAnnualCatch']:
        return catch_yr_df
    if table in ['EcospaceHabitat','EcospaceLayer']:
        return habitat_df
    if table == 'EcospaceGrid':
        return grid_df
    if table == 'EcospaceDispersal':
        return dispersal_df
    return pd.DataFrame()

try:
    e.read_ewemdb_table = mock_reader
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ewemdb', delete=False) as f:
        tp = f.name
    scen = e.ecosim_scenario_from_ewemdb(tp, scenario=1)
    print('scen type:', type(scen))
    print('has ecospace attribute:', hasattr(scen, 'ecospace'))
    print('ecospace:', scen.ecospace)
finally:
    e.read_ewemdb_table = orig
