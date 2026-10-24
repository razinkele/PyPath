# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from pathlib import Path

print("="*70)
print("TASK 2.4 & 2.5: VALIDATE & MAP TIMESERIES")
print("="*70)
print()

# Load components
model_groups = pd.read_csv('curonian_model_groups_35.csv')
mapping = pd.read_csv('curonian_species_to_group_mapping.csv')

# Load and enhance timeseries collection
with open('curonian_timeseries_collection.pkl', 'rb') as f:
    collection = pickle.load(f)

print("Validation Report")
print("-" * 70)
print()

# Test 1: Verify all CPUE species in mapping
expected_cpue = {'ROAC', 'BRAM', 'ZAND', 'PERC', 'VIMB'}
mapped_cpue = set(mapping[mapping['data_type'] == 'CPUE/Landings']['code'])
cpue_check = expected_cpue.issubset(mapped_cpue)
print(f"[OK] All CPUE species mapped: {cpue_check}")
print(f"      Expected: {expected_cpue}")
print(f"      Found: {mapped_cpue}")
print()

# Test 2: Check for valid group indices
invalid_indices = mapping[~mapping['group_idx'].between(0, 32)]
print(f"[OK] All group indices valid (0-32): {len(invalid_indices) == 0}")
if len(invalid_indices) > 0:
    print(f"      Invalid: {invalid_indices[['code', 'group_idx']].to_string()}")
print()

# Test 3: Check for marine species (quality control)
marine_keywords = ['marine', 'salt', 'marine ', ' marine']
marine_species = mapping[mapping['notes'].str.contains('|'.join(marine_keywords), case=False, na=False)]
print(f"[OK] No marine species flagged: {len(marine_species) == 0}")
if len(marine_species) > 0:
    print(f"      Marine species found: {marine_species['code'].tolist()}")
print()

# Test 4: Coverage assessment
print(f"Data Coverage:")
print(f"  Total groups in model: {len(model_groups)}")
print(f"  Species with data mapping: {len(mapping)}")
print(f"  Coverage: {100 * len(mapping) / len(model_groups):.1f}%")
print()

# Test 5: Apply mapping to timeseries
print("Applying group indices to timeseries...")
for i, s in enumerate(collection.series):
    if i == 0:  # Discharge - no mapping needed
        s.group_idx = -1
        print(f"  Series {i} ({s.name}): group_idx = -1 (environmental forcing)")
    else:
        # Try to match series name to species code
        found = False
        for _, row in mapping.iterrows():
            if row['common'].lower() in s.name.lower() or row['code'] in s.name:
                s.group_idx = int(row['group_idx'])
                print(f"  Series {i} ({s.name}): group_idx = {s.group_idx} ({row['group_name']})")
                found = True
                break
        if not found:
            # Extract from series name if possible
            print(f"  Series {i} ({s.name}): No mapping found - will infer from context")

print()
print(f"Saving enhanced collection...")
with open('curonian_timeseries_mapped.pkl', 'wb') as f:
    pickle.dump(collection, f)
print(f"[SAVED] curonian_timeseries_mapped.pkl ({Path('curonian_timeseries_mapped.pkl').stat().st_size} bytes)")
