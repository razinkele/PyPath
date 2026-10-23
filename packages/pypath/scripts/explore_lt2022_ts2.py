"""Get the latest time series dataset details and forcing shapes."""

from pathlib import Path

from pypath import read_ewemdb_table

DB = str(Path(__file__).resolve().parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb")

# Dataset #35 (latest with real estimates)
ts_all = read_ewemdb_table(DB, "EcosimTimeSeries")
ds35 = ts_all[ts_all["DatasetID"] == 35]
print(
    f"=== Dataset #35 'fittingdata2022 04 07with real estimates': {len(ds35)} series ==="
)
print(f"DatType codes: {sorted(ds35['DatType'].unique())}")
print()
for _, r in ds35.iterrows():
    vals = str(r["TimeValues"]).split()
    print(
        f"  TS#{r['TimeSeriesID']:>5d}  Type={r['DatType']:>3d}  {r['DatName']:<30s}  {len(vals)} points  [{vals[0]}..{vals[-1]}]"
    )

# Also check dataset #34
ds34 = ts_all[ts_all["DatasetID"] == 34]
print(f"\n=== Dataset #34 'fittingdata2022 04 07': {len(ds34)} series ===")
for _, r in ds34.iterrows():
    vals = str(r["TimeValues"]).split()
    print(
        f"  TS#{r['TimeSeriesID']:>5d}  Type={r['DatType']:>3d}  {r['DatName']:<30s}  {len(vals)} points  [{vals[0]}..{vals[-1]}]"
    )

# Get the group mapping for DatType interpretation
tsg = read_ewemdb_table(DB, "EcosimTimeSeriesGroup")
print("\n=== Time Series -> Group Mapping (first 30) ===")
groups = read_ewemdb_table(DB, "EcopathGroup")
gmap = dict(zip(groups["GroupID"], groups["GroupName"]))
for _, r in tsg.head(30).iterrows():
    gname = gmap.get(r["GroupID"], f"Group#{r['GroupID']}")
    print(
        f"  TS#{r['TimeSeriesID']:>5d}  -> {gname:<30s}  Var={r.get('VariableName', '')}"
    )

# Forcing matrix for scenario 16
fm = read_ewemdb_table(DB, "EcosimScenarioForcingMatrix")
fm16 = fm[fm["ScenarioID"] == 16] if "ScenarioID" in fm.columns else fm
print(f"\n=== Forcing Matrix (scenario 16): {len(fm16)} rows ===")
print(fm16.columns.tolist())
if len(fm16) > 0:
    print(fm16.head(20).to_string())
