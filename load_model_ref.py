from pathlib import Path
from pypath.io.ewemdb import read_ewemdb
import pandas as pd
import pickle

# Load the simple 20-group model for reference
db_path = Path('C:/Users/DELL/OneDrive - ku.lt/ecopath workshop/curonian pelagic.eweaccdb')
params = read_ewemdb(db_path)
groups = params.model.copy()

print("Simple Curonian Pelagic Model (20 groups):")
print(groups[['Group', 'Type']].to_string())
print()

# Save for reference
groups.to_csv('_curonian_20group_reference.csv', index=False)
print("Saved to _curonian_20group_reference.csv")
