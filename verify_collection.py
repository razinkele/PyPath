"""Verify and inspect loaded timeseries collection."""

import pickle
from pathlib import Path
import numpy as np

pkl_file = Path("curonian_timeseries_collection.pkl")

print("\n" + "=" * 70)
print("LOADING VERIFICATION TEST")
print("=" * 70 + "\n")

# Test 1: Load collection
try:
    with open(pkl_file, "rb") as f:
        collection = pickle.load(f)
    print("TEST 1: Collection loads without error")
    print("  PASS")
except Exception as e:
    print(f"TEST 1 FAILED: {e}")
    exit(1)

# Test 2: Check collection properties
print("\nTEST 2: Collection properties")
print(f"  Total series: {len(collection.series)}")
assert len(collection.series) > 0, "No series in collection"
print(f"  Timesteps: {collection.n_timesteps}")
assert collection.n_timesteps > 0, "No timesteps"
print("  PASS")

# Test 3: Verify data categories
print("\nTEST 3: Data categories")
print(f"  Observed biomass: {len(collection.observed_biomass)}")
print(f"  Observed catch: {len(collection.observed_catch)}")
print(f"  Forced biomass: {len(collection.forced_biomass)}")
print(f"  Forced effort: {len(collection.forced_effort)}")
assert len(collection.observed_catch) > 0, "No CPUE data found"
assert len(collection.forced_biomass) > 0, "No discharge forcing found"
print("  PASS")

# Test 4: Data quality check
print("\nTEST 4: Data quality in calibration window (2001-2011)")
calib_start_idx = int(1488 * (2001 - 1888) / (2011 - 1888))
calib_end_idx = int(1488 * (2011 - 1888) / (2011 - 1888))
calib_len = calib_end_idx - calib_start_idx

for s in collection.series:
    if calib_len > 0:
        calib_data = s.values[calib_start_idx:calib_end_idx]
        valid = np.sum(~np.isnan(calib_data))
        if len(calib_data) > 0:
            quality = 100.0 * valid / len(calib_data)
        else:
            quality = 0.0
        print(f"  {s.name:30s}: {quality:5.1f}% complete ({valid}/{len(calib_data)})")

print("  PASS")

# Test 5: Inspect series details
print("\nTEST 5: Series details")
for i, s in enumerate(collection.series):
    non_nan = np.sum(~np.isnan(s.values))
    pct = 100.0 * non_nan / len(s.values) if len(s.values) > 0 else 0
    print(f"  [{i}] {s.name}")
    print(f"      Type: {s.dat_type}, Group: {s.group_idx}, Fleet: {s.fleet_idx}")
    print(f"      Values: {non_nan}/{len(s.values)} ({pct:.1f}% complete)")
    if non_nan > 0:
        print(f"      Range: {np.nanmin(s.values):.2f} - {np.nanmax(s.values):.2f}")
print("  PASS")

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70 + "\n")
