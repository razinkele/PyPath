"""Load Phase 1 CSV data and create PyPath timeseries collection."""

import pickle
from pathlib import Path
import sys

from curonian_data_loader import CurionianDataLoader

PHASE1_DATA_DIR = Path("curonian_data_phase1/raw_csv")
OUTPUT_FILE = Path("curonian_timeseries_collection.pkl")
LOG_FILE = Path("phase2_checkpoint1_loading.log")

def main():
    """Execute data loading pipeline."""
    print("\n" + "=" * 70)
    print("PHASE 2 - CHECKPOINT 1: DATA LOADING")
    print("=" * 70 + "\n")

    if not PHASE1_DATA_DIR.exists():
        print(f"ERROR: Phase 1 data directory not found: {PHASE1_DATA_DIR}")
        sys.exit(1)

    print(f"Data directory: {PHASE1_DATA_DIR.absolute()}")
    print(f"Output file: {OUTPUT_FILE.absolute()}")
    print()

    loader = CurionianDataLoader(PHASE1_DATA_DIR)
    collection, validation_results = loader.load_all()

    print("\n" + "=" * 70)
    print("LOADING SUMMARY")
    print("=" * 70)
    print(f"Total series loaded: {len(collection.series)}")
    print(f"Timesteps: {collection.n_timesteps}")
    print(f"Observed catch series: {len(collection.observed_catch)}")
    print(f"Forced biomass series: {len(collection.forced_biomass)}")
    print()

    print(f"Saving collection to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(collection, f)
        print(f"SUCCESS: Collection saved ({OUTPUT_FILE.stat().st_size} bytes)")
    except Exception as e:
        print(f"ERROR: Failed to save collection: {e}")
        sys.exit(1)

    log_text = loader.get_log_text()
    log_text += "\n\n" + "=" * 70 + "\n"
    log_text += "VALIDATION RESULTS\n"
    log_text += "=" * 70 + "\n"
    for key, val in validation_results.items():
        if isinstance(val, dict):
            log_text += f"{key}:\n"
            for k2, v2 in val.items():
                log_text += f"  {k2}: {v2}\n"
        else:
            log_text += f"{key}: {val}\n"

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(log_text)
    print(f"Log saved to {LOG_FILE}")

    print("\n" + "=" * 70)
    print("CHECKPOINT 1 COMPLETE")
    print("=" * 70)
    print(f"Ready for Checkpoint 2: Species Mapping")
    print()

if __name__ == "__main__":
    main()
