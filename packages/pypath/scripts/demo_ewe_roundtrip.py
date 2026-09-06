"""Demo: Load EwE model, calibrate Ecosim scenario, write back as new EwE file.

This script demonstrates the full round-trip workflow:
1. Load a real EwE database with Ecosim scenarios
2. Modify/calibrate scenario parameters (vulnerabilities, forcing)
3. Write the modified model back as a new EwE database
"""

import json
import zipfile
from pathlib import Path

import numpy as np

from pypath.core.ecosim import rsim_run
from pypath.io.ewe_writer import write_ewemdb
from pypath.io.ewemdb import (
    ecosim_scenario_from_ewemdb,
    get_ewemdb_metadata,
    read_ewemdb,
)

# ── Step 1: Explore what's in the database ──────────────────────────
db_path = str(Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb")
print(f"Database: {Path(db_path).name}\n")


meta = get_ewemdb_metadata(db_path)
print(f"Model: {meta['name']}")
print(f"Groups: {meta['num_groups']}, Fleets: {meta['num_fleets']}")
print(f"Has Ecosim: {meta['has_ecosim']}")
print(f"Scenarios: {meta['num_scenarios']}")
for s in meta.get("scenarios", []):
    if isinstance(s, dict):
        print(f"  #{s.get('id', '?')}: {s.get('name', '?')}")
    else:
        print(f"  {s}")

# ── Step 2: Load scenario 1 as a full RsimScenario ─────────────────
print("\n--- Loading Ecosim Scenario 1 ---")

scen = ecosim_scenario_from_ewemdb(db_path, scenario=1, years=range(1, 51))
print(f"Scenario loaded: {scen.eco_name}")
print(f"Groups: {len(scen.params.spname)} (including Outside)")
print(f"Start year: {scen.start_year}")
print(f"Simulation months: {scen.forcing.ForcedBio.shape[1]}")

# Show some VV (vulnerability) values
print("\nOriginal vulnerabilities (VV) for first 10 groups:")
for i in range(min(10, len(scen.params.VV))):
    print(f"  {scen.params.spname[i]:30s} VV={scen.params.VV[i]:.4f}")

# ── Step 3: Run baseline Ecosim ─────────────────────────────────────
print("\n--- Running Baseline Ecosim ---")

baseline_out = rsim_run(scen, method="RK4", years=range(1, 51))
print(f"Baseline run complete: {baseline_out.out_Biomass.shape[0]} months")

# Show final biomass for a few groups
print("\nBaseline final biomass (last year average):")
last_12 = baseline_out.out_Biomass[-12:, :]
for i in range(1, min(8, last_12.shape[1])):
    avg = np.mean(last_12[:, i])
    print(f"  {scen.params.spname[i]:30s} B={avg:.4f}")

# ── Step 4: Calibrate — modify vulnerabilities ─────────────────────
print("\n--- Calibrating: Adjusting Vulnerabilities ---")

# Example calibration: increase vulnerability for consumers (make them
# more top-down controlled) and slightly reduce for primary producers
n_groups = len(scen.params.VV)
original_vv = scen.params.VV.copy()

for i in range(n_groups):
    name = scen.params.spname[i] if i < len(scen.params.spname) else f"Group{i}"
    # Skip Outside (index 0) and fleet groups
    if i == 0:
        continue
    old_vv = scen.params.VV[i]
    if old_vv > 1.0:
        # Increase vulnerability by 25% for groups already > 1 (top-down)
        scen.params.VV[i] = min(old_vv * 1.25, 100.0)
    elif old_vv == 1.0:
        # Bump from wasp-waist to slightly top-down
        scen.params.VV[i] = 1.5

print("Calibrated vulnerabilities (first 10 groups):")
for i in range(min(10, n_groups)):
    old = original_vv[i]
    new = scen.params.VV[i]
    marker = " *" if abs(old - new) > 0.01 else ""
    print(f"  {scen.params.spname[i]:30s} VV: {old:.4f} -> {new:.4f}{marker}")

# ── Step 5: Run calibrated Ecosim ──────────────────────────────────
print("\n--- Running Calibrated Ecosim ---")
calibrated_out = rsim_run(scen, method="RK4", years=range(1, 51))
print(f"Calibrated run complete: {calibrated_out.out_Biomass.shape[0]} months")

print("\nCalibrated final biomass vs baseline:")
cal_last_12 = calibrated_out.out_Biomass[-12:, :]
for i in range(1, min(8, cal_last_12.shape[1])):
    base_avg = np.mean(last_12[:, i])
    cal_avg = np.mean(cal_last_12[:, i])
    pct = ((cal_avg - base_avg) / base_avg * 100) if base_avg > 0 else 0
    print(
        f"  {scen.params.spname[i]:30s} base={base_avg:.4f}  cal={cal_avg:.4f}  ({pct:+.1f}%)"
    )

# ── Step 6: Write back as new EwE database ──────────────────────────
print("\n--- Writing Calibrated Model ---")

# Re-read the original Ecopath params (for the model/diet structure)
params = read_ewemdb(db_path)

# Write as Access database (primary)
out_accdb = str(Path(db_path).parent / "LT2022_calibrated.eweaccdb")
write_ewemdb(params, out_accdb, scenarios=[scen], backend="access", source_db=db_path)
print(f"Written Access DB: {out_accdb}")
print(f"  Size: {Path(out_accdb).stat().st_size:,} bytes")

# Also write as CSV bundle (cross-platform fallback)
out_csv = str(Path(db_path).parent / "LT2022_calibrated.ewecsv.zip")
write_ewemdb(params, out_csv, scenarios=[scen], backend="csv")
print(f"Written CSV bundle: {out_csv}")
print(f"  Size: {Path(out_csv).stat().st_size:,} bytes")

# ── Step 7: Verify the round-trip ───────────────────────────────────
print("\n--- Verifying Round-Trip ---")
params_back = read_ewemdb(out_accdb)

orig_names = params.model[params.model["Type"] != 3]["Group"].tolist()
back_names = params_back.model[params_back.model["Type"] != 3]["Group"].tolist()
assert back_names == orig_names, "Group names mismatch!"
print(f"  Group names: OK ({len(orig_names)} groups)")

bio_orig = params.model[params.model["Type"] != 3]["Biomass"].values
bio_back = params_back.model[params_back.model["Type"] != 3]["Biomass"].values
mismatches = 0
for o, b in zip(bio_orig, bio_back):
    if np.isnan(o) and np.isnan(b):
        continue
    elif np.isnan(o) or np.isnan(b):
        mismatches += 1
    elif abs(o - b) / max(abs(o), 1e-10) > 1e-6:
        mismatches += 1
print(f"  Biomass values: {'OK' if mismatches == 0 else f'{mismatches} mismatches'}")

# Check Ecosim tables exist in the CSV bundle

with zipfile.ZipFile(out_csv) as zf:
    manifest = json.loads(zf.read("manifest.json"))
    tables = manifest["tables"]
    has_ecosim = any("Ecosim" in t for t in tables)
    print(f"  CSV bundle tables: {len(tables)}")
    print(f"  Has Ecosim tables: {has_ecosim}")
    for t in tables:
        if "Ecosim" in t:
            import pandas as pd

            df = pd.read_csv(zf.open(f"{t}.csv"))
            print(f"    {t}: {len(df)} rows")

print("\nDone! Calibrated model written successfully.")
