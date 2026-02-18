import pandas as pd

md = pd.read_csv('tests/data/rpath_reference/ecopath/model_params.csv')
print(md[md['Group']=='Discards'][['Group','Type','Biomass']])
print('index', list(md['Group']).index('Discards'))
