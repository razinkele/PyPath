#!/usr/bin/env python
"""
Test script for biodiversity data workflow in Shiny app.

This script tests the complete workflow that the Shiny app uses:
1. Fetch species info from WoRMS/OBIS/FishBase
2. Create Ecopath model from biodiversity data
3. Verify model structure
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from pypath.io.biodata import (
    get_species_info,
    batch_get_species_info,
    biodata_to_rpath,
    _fetch_worms_vernacular,
)

print("=" * 70)
print("Biodiversity Data Workflow Test")
print("=" * 70)

# Test 1: Individual WoRMS lookup
print("\n1. Testing individual WoRMS vernacular search...")
print("-" * 70)

test_species = [
    "Atlantic cod",
    "cod",
    "Atlantic herring",
    "herring",
    "European sprat",
    "sprat"
]

for species in test_species:
    try:
        print(f"\nSearching for: '{species}'")
        results = _fetch_worms_vernacular(species, cache=False, timeout=30)
        if results:
            print(f"  [OK] Found {len(results)} result(s)")
            for i, r in enumerate(results[:3]):  # Show first 3
                print(f"    [{i+1}] {r.get('scientificname')} (AphiaID: {r.get('AphiaID')})")
        else:
            print(f"  [FAIL] No results found")
    except Exception as e:
        print(f"  [ERROR] {e}")

# Test 2: Single species workflow
print("\n\n2. Testing single species workflow...")
print("-" * 70)

try:
    print("\nFetching info for 'cod'...")
    info = get_species_info("cod", strict=False, timeout=30)
    print(f"[OK] Success!")
    print(f"  Common name: {info.common_name}")
    print(f"  Scientific name: {info.scientific_name}")
    print(f"  AphiaID: {info.aphia_id}")
    print(f"  Trophic level: {info.trophic_level}")
    print(f"  Max length: {info.max_length}")
    print(f"  OBIS occurrences: {info.occurrence_count}")
except Exception as e:
    print(f"[FAIL] Failed: {e}")

# Test 3: Batch workflow (as used in Shiny app)
print("\n\n3. Testing batch workflow (Shiny app scenario)...")
print("-" * 70)

species_list = [
    "cod",
    "herring",
    "sprat",
]

print(f"\nFetching data for {len(species_list)} species...")
print(f"Species: {', '.join(species_list)}")

try:
    df = batch_get_species_info(
        species_list,
        include_occurrences=True,
        include_traits=True,
        strict=False,
        max_workers=5,
        timeout=45
    )

    if df is not None and len(df) > 0:
        print(f"\n[OK] Retrieved data for {len(df)} species")
        print("\nResults:")
        for idx, row in df.iterrows():
            print(f"\n  {row['common_name']}:")
            print(f"    Scientific: {row['scientific_name']}")
            print(f"    TL: {row['trophic_level']}")
            print(f"    Max length: {row['max_length']} cm")
            print(f"    OBIS records: {row['occurrence_count']}")
    else:
        print("[FAIL] No species data retrieved")

except Exception as e:
    print(f"[FAIL] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Model creation (as used in Shiny app)
print("\n\n4. Testing model creation...")
print("-" * 70)

try:
    # Use simple species list
    simple_species = ["cod", "herring"]
    print(f"\nFetching data for: {', '.join(simple_species)}")

    df = batch_get_species_info(
        simple_species,
        include_occurrences=True,
        include_traits=True,
        strict=False,
        timeout=45
    )

    if df is not None and len(df) > 0:
        print(f"[OK] Retrieved {len(df)} species")

        # Create biomass estimates (as in Shiny app)
        biomass_estimates = {}
        for idx, row in df.iterrows():
            sp_name = row['common_name']
            biomass_estimates[sp_name] = 1.0  # Default biomass

        print(f"\nCreating Ecopath model...")
        params = biodata_to_rpath(
            df,
            biomass_estimates=biomass_estimates,
            area_km2=1000
        )

        print(f"[OK] Model created!")
        print(f"  Groups: {len(params.model)}")
        print(f"  Diet entries: {(params.diet.iloc[:, 1:] > 0).sum().sum()}")
        print(f"\nModel groups:")
        for idx, row in params.model.iterrows():
            print(f"  - {row['Group']} (Type: {int(row['Type'])}, TL: {row.get('TrophicLevel', 'N/A')})")
    else:
        print("[FAIL] No species data to create model")

except Exception as e:
    print(f"[FAIL] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: API connectivity check
print("\n\n5. Testing API connectivity...")
print("-" * 70)

try:
    import requests

    # Test WoRMS
    print("\nTesting WoRMS API...")
    response = requests.get(
        "https://www.marinespecies.org/rest/AphiaRecordsByVernacular/cod",
        params={"like": "false", "offset": 1},
        timeout=10
    )
    if response.status_code == 200:
        print(f"  [OK] WoRMS API accessible (status: {response.status_code})")
        data = response.json()
        print(f"  [OK] Found {len(data)} results for 'cod'")
    else:
        print(f"  [FAIL] WoRMS API error (status: {response.status_code})")

    # Test OBIS
    print("\nTesting OBIS API...")
    response = requests.get(
        "https://api.obis.org/v3/occurrence",
        params={"scientificname": "Gadus morhua", "size": 1},
        timeout=10
    )
    if response.status_code == 200:
        print(f"  [OK] OBIS API accessible (status: {response.status_code})")
    else:
        print(f"  [FAIL] OBIS API error (status: {response.status_code})")

    # Test FishBase
    print("\nTesting FishBase API...")
    response = requests.get(
        "https://fishbase.ropensci.org/species",
        params={"Genus": "Gadus", "Species": "morhua"},
        timeout=10
    )
    if response.status_code == 200:
        print(f"  [OK] FishBase API accessible (status: {response.status_code})")
    else:
        print(f"  [FAIL] FishBase API error (status: {response.status_code})")

except Exception as e:
    print(f"[FAIL] API connectivity test failed: {e}")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
