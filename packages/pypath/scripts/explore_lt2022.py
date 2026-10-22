"""Explore the LT2022 model structure and attempt calibration."""

from pathlib import Path
from pypath import read_ewemdb, read_ewemdb_table, rpath

DB = str(Path(__file__).resolve().parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb")

# Explore group structure
groups = read_ewemdb_table(DB, "EcopathGroup")
print(f"=== EcopathGroup: {len(groups)} rows, columns: ===")
print(groups.columns.tolist())
print()
for _, r in groups.iterrows():
    print(
        f"  #{int(r.get('GroupID', r.get('Sequence', '?'))):>3d}  {str(r.get('GroupName', ''))[:40]:<40s}  Type={r.get('Type', '?')}"
    )
print()

# Explore fleet structure
fleets = read_ewemdb_table(DB, "EcopathFleet")
print(f"=== EcopathFleet: {len(fleets)} rows ===")
print(fleets.columns.tolist())
for _, r in fleets.iterrows():
    print(
        f"  #{int(r.get('FleetID', r.get('Sequence', '?'))):>3d}  {str(r.get('FleetName', ''))[:40]}"
    )
print()

# Ecosim scenarios
ecosim = read_ewemdb_table(DB, "EcosimScenario")
print(f"=== EcosimScenario: {len(ecosim)} rows ===")
print(ecosim.columns.tolist())
for _, r in ecosim.iterrows():
    print(f"  #{r.get('ScenarioID', '?')}  {r.get('ScenarioName', r.get('Name', ''))}")
print()

# Time series datasets
try:
    ts = read_ewemdb_table(DB, "EcosimTimeSeriesDataset")
    print(f"=== EcosimTimeSeriesDataset: {len(ts)} rows ===")
    print(ts.columns.tolist())
    for _, r in ts.iterrows():
        print(
            f"  #{r.get('DatasetID', '?')}  {r.get('DatasetName', r.get('Name', ''))}"
        )
except Exception as e:
    print(f"EcosimTimeSeriesDataset: {e}")
print()

# Time series
try:
    tsg = read_ewemdb_table(DB, "EcosimTimeSeriesGroup")
    print(f"=== EcosimTimeSeriesGroup: {len(tsg)} rows ===")
    print(tsg.columns.tolist())
    print(tsg.head(10).to_string())
except Exception as e:
    print(f"EcosimTimeSeriesGroup: {e}")
print()

try:
    tsf = read_ewemdb_table(DB, "EcosimTimeSeriesFleet")
    print(f"=== EcosimTimeSeriesFleet: {len(tsf)} rows ===")
    print(tsf.columns.tolist())
    print(tsf.head(10).to_string())
except Exception as e:
    print(f"EcosimTimeSeriesFleet: {e}")
