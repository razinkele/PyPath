import pandas as pd

df = pd.read_csv('build/seabirds_diagnostics.csv')
print('Column means:')
print(df[['production','consumption','predation_loss','fish_loss','m0_loss','deriv']].mean())
print('\nFirst 10 rows:')
print(df.head(10).to_string(index=False))
