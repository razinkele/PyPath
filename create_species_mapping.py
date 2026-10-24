import pandas as pd
import pickle
from pathlib import Path

print("="*70)
print("TASK 2.3: CREATE SPECIES MAPPING TABLE")
print("="*70)
print()

# Load the 35-group model
model_groups = pd.read_csv('curonian_model_groups_35.csv')

# Load timeseries collection to understand species
with open('curonian_timeseries_collection.pkl', 'rb') as f:
    collection = pickle.load(f)

# Create the species-to-group mapping
species_mapping_data = []

# CPUE species from timeseries (indices 1-5)
cpue_species = [
    {'code': 'ROAC', 'common': 'Roach', 'scientific': 'Rutilus rutilus', 'group_idx': 0, 'group_name': 'Roach', 'data_type': 'CPUE/Landings', 'time_period': '1947-2018', 'notes': 'Primary commercial species in Curonian Lagoon'},
    {'code': 'BRAM', 'common': 'Bream', 'scientific': 'Abramis brama', 'group_idx': 1, 'group_name': 'Bream', 'data_type': 'CPUE/Landings', 'time_period': '1947-2018', 'notes': 'Primary commercial species in Curonian Lagoon'},
    {'code': 'ZAND', 'common': 'Zander', 'scientific': 'Sander lucioperca', 'group_idx': 2, 'group_name': 'Pike-Perch (Zander)', 'data_type': 'CPUE/Landings', 'time_period': '1947-2018', 'notes': 'Predatory species, recreational & commercial fishing'},
    {'code': 'PERC', 'common': 'Perch', 'scientific': 'Perca fluviatilis', 'group_idx': 3, 'group_name': 'Perch', 'data_type': 'CPUE/Landings', 'time_period': '1947-2018', 'notes': 'Commercial species, important food resource'},
    {'code': 'VIMB', 'common': 'Vimba', 'scientific': 'Vimba vimba', 'group_idx': 6, 'group_name': 'Vimba', 'data_type': 'CPUE/Landings', 'time_period': '1947-2018', 'notes': 'Commercial species, limited habitat in Curonian'},
]

species_mapping_data.extend(cpue_species)

# Other potential landings species (estimated from ecosystem knowledge)
other_species = [
    {'code': 'SILB', 'common': 'Silver Bream', 'scientific': 'Blicca bjoerkna', 'group_idx': 5, 'group_name': 'Silver Bream', 'data_type': 'Landings', 'time_period': '1947-2011', 'notes': 'Secondary commercial species'},
    {'code': 'RUFE', 'common': 'Ruffe', 'scientific': 'Gymnocephalus cernuus', 'group_idx': 4, 'group_name': 'Ruffe', 'data_type': 'Landings', 'time_period': '1947-2011', 'notes': 'Minor commercial species'},
    {'code': 'EEL', 'common': 'Eel', 'scientific': 'Anguilla anguilla', 'group_idx': 9, 'group_name': 'Eel', 'data_type': 'Landings', 'time_period': '1947-2011', 'notes': 'Declining species, diadromous'},
]

species_mapping_data.extend(other_species)

# Monitoring/non-commercial species
monitoring_species = [
    {'code': 'PLKA', 'common': 'Planktiv. fish', 'scientific': 'Alburnus spp.', 'group_idx': 7, 'group_name': 'Planktivorous fish', 'data_type': 'Monitoring', 'time_period': '2001-2018', 'notes': 'Small pelagic fish, not targeted'},
]

species_mapping_data.extend(monitoring_species)

# Create DataFrame
mapping_df = pd.DataFrame(species_mapping_data)
mapping_df = mapping_df[[col for col in mapping_df.columns]]

print("Species to Group Mapping (7 mapped species):")
print(mapping_df[['code', 'common', 'scientific', 'group_idx', 'group_name']].to_string(index=False))
print()

# Save the mapping
mapping_df.to_csv('curonian_species_to_group_mapping.csv', index=False)
print("Saved mapping to: curonian_species_to_group_mapping.csv")
print()

print(f"Mapping summary:")
print(f"  Total species mapped: {len(mapping_df)}")
print(f"  CPUE species: {len([s for s in species_mapping_data if s['data_type'] == 'CPUE/Landings'])}")
print(f"  Landings only: {len([s for s in species_mapping_data if s['data_type'] == 'Landings'])}")
print(f"  Monitoring: {len([s for s in species_mapping_data if s['data_type'] == 'Monitoring'])}")
