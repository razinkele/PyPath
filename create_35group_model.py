from pathlib import Path
from pypath.io.ewemdb import read_ewemdb
import pandas as pd
import pickle

# Load timeseries collection to see CPUE species
with open('curonian_timeseries_collection.pkl', 'rb') as f:
    collection = pickle.load(f)

print("="*70)
print("PHASE 2 CHECKPOINT 2: SPECIES MAPPING")
print("="*70)
print()

print("Timeseries in collection:")
for i, s in enumerate(collection.series):
    print(f"  {i}: {s.name} (dat_type={s.dat_type})")
print()

# Define the 35-group Ecopath model structure
# Based on simplifying the 60-group multistanza down to key ecological groups

groups_data = [
    # Fish - commercial species (aggregated from multistanza)
    {'index': 0, 'name': 'Roach', 'scientific_name': 'Rutilus rutilus', 'type': 0, 'tl': 3.1},
    {'index': 1, 'name': 'Bream', 'scientific_name': 'Abramis brama', 'type': 0, 'tl': 3.0},
    {'index': 2, 'name': 'Pike-Perch (Zander)', 'scientific_name': 'Sander lucioperca', 'type': 0, 'tl': 4.2},
    {'index': 3, 'name': 'Perch', 'scientific_name': 'Perca fluviatilis', 'type': 0, 'tl': 3.8},
    {'index': 4, 'name': 'Ruffe', 'scientific_name': 'Gymnocephalus cernuus', 'type': 0, 'tl': 3.5},
    {'index': 5, 'name': 'Silver Bream', 'scientific_name': 'Blicca bjoerkna', 'type': 0, 'tl': 3.2},
    {'index': 6, 'name': 'Vimba', 'scientific_name': 'Vimba vimba', 'type': 0, 'tl': 3.3},
    
    # Other fish
    {'index': 7, 'name': 'Planktivorous fish', 'scientific_name': 'Cyprinidae spp.', 'type': 0, 'tl': 2.8},
    {'index': 8, 'name': 'Demersal fish', 'scientific_name': 'Benthic fish', 'type': 0, 'tl': 3.2},
    {'index': 9, 'name': 'Eel', 'scientific_name': 'Anguilla anguilla', 'type': 0, 'tl': 3.9},
    
    # Birds (piscivores)
    {'index': 10, 'name': 'Grey Heron', 'scientific_name': 'Ardea cinerea', 'type': 0, 'tl': 4.1},
    {'index': 11, 'name': 'Larus gull', 'scientific_name': 'Larus spp.', 'type': 0, 'tl': 4.0},
    {'index': 12, 'name': 'Goosander', 'scientific_name': 'Mergellus albellus', 'type': 0, 'tl': 3.9},
    {'index': 13, 'name': 'Great Crested Grebe', 'scientific_name': 'Podiceps cristatus', 'type': 0, 'tl': 3.8},
    {'index': 14, 'name': 'Cormorant', 'scientific_name': 'Phalacrocorax carbo', 'type': 0, 'tl': 4.2},
    
    # Zooplankton
    {'index': 15, 'name': 'Daphnia & Cladocera', 'scientific_name': 'Cladocera', 'type': 0, 'tl': 2.2},
    {'index': 16, 'name': 'Bosmina', 'scientific_name': 'Bosmina spp.', 'type': 0, 'tl': 2.1},
    {'index': 17, 'name': 'Grazing zooplankton', 'scientific_name': 'Copepoda', 'type': 0, 'tl': 2.3},
    {'index': 18, 'name': 'Carnivorous zooplankton', 'scientific_name': 'Hydra, Leptodora', 'type': 0, 'tl': 2.8},
    {'index': 19, 'name': 'Rotifers', 'scientific_name': 'Rotifera', 'type': 0, 'tl': 2.2},
    {'index': 20, 'name': 'Copepod nauplii', 'scientific_name': 'Copepoda nauplii', 'type': 0, 'tl': 2.0},
    
    # Benthic invertebrates
    {'index': 21, 'name': 'Chironomids', 'scientific_name': 'Diptera larvae', 'type': 0, 'tl': 2.5},
    {'index': 22, 'name': 'Oligochaetes', 'scientific_name': 'Oligochaeta', 'type': 0, 'tl': 2.4},
    {'index': 23, 'name': 'Dreissena (mussel)', 'scientific_name': 'Dreissena polymorpha', 'type': 0, 'tl': 2.0},
    {'index': 24, 'name': 'Nematodes', 'scientific_name': 'Nematoda', 'type': 0, 'tl': 2.3},
    {'index': 25, 'name': 'Mysids', 'scientific_name': 'Mysidacea', 'type': 0, 'tl': 2.6},
    {'index': 26, 'name': 'Deposit feeder gastropods', 'scientific_name': 'Gastropoda', 'type': 0, 'tl': 2.2},
    {'index': 27, 'name': 'Filtrator bivalves', 'scientific_name': 'Bivalvia', 'type': 0, 'tl': 2.1},
    {'index': 28, 'name': 'Meiobenthos', 'scientific_name': 'Meiofauna', 'type': 0, 'tl': 2.0},
    
    # Primary producers
    {'index': 29, 'name': 'Phytoplankton', 'scientific_name': 'Autotrophic protists', 'type': 1, 'tl': 1.0},
    {'index': 30, 'name': 'Bacteria', 'scientific_name': 'Prokaryota', 'type': 0, 'tl': 2.0},
    
    # Detritus and fleet
    {'index': 31, 'name': 'Detritus', 'scientific_name': 'Dead organic matter', 'type': 2, 'tl': 0.0},
    {'index': 32, 'name': 'Fishing Fleet - Commercial', 'scientific_name': 'Anthropogenic', 'type': 3, 'tl': 0.0},
]

# Create DataFrame
model_groups_df = pd.DataFrame(groups_data)

print("Created 35-group Ecopath model structure:")
print(model_groups_df.to_string(index=False))
print()

# Save the 35-group model definition
model_groups_df.to_csv('curonian_model_groups_35.csv', index=False)
print(f"Saved 35-group model to curonian_model_groups_35.csv")
print()
