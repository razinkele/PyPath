"""Explore LT2022 time series and Ecosim scenario details."""

from pathlib import Path
import pandas as pd
from pypath import read_ewemdb_table

DB = str(Path(__file__).resolve().parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb")

# Get the latest/best time series dataset
ts_datasets = read_ewemdb_table(DB, "EcosimTimeSeriesDataset")
print("=== Time Series Datasets (last 5) ===")
for _, r in ts_datasets.tail(5).iterrows():
    print(
        f"  #{r['DatasetID']}  {r['DatasetName']}  FirstYear={r.get('FirstYear', '')}  NumPoints={r.get('NumPoints', '')}"
    )

# Get time series definitions
ts = read_ewemdb_table(DB, "EcosimTimeSeries")
print(f"\n=== EcosimTimeSeries: {len(ts)} rows ===")
print(ts.columns.tolist())
print(ts.head(20).to_string())

# Get scenario #16 (test_st_final) details
print("\n=== Scenario 16 details ===")
sc = read_ewemdb_table(DB, "EcosimScenario")
sc16 = sc[sc["ScenarioID"] == 16]
print(sc16.to_string())

# Scenario group params (vulnerabilities)
sg = read_ewemdb_table(DB, "EcosimScenarioGroup")
print(f"\n=== EcosimScenarioGroup: {len(sg)} rows ===")
print(sg.columns.tolist())
# Filter to scenario 16
sg16 = (
    sg[sg.get("ScenarioID", sg.get("Scenario", pd.Series())) == 16]
    if "ScenarioID" in sg.columns
    else sg
)
print(f"Scenario 16 group params: {len(sg16)} rows")
if len(sg16) > 0:
    print(sg16.head(10).to_string())

# Scenario arena (vulnerabilities)
sa = read_ewemdb_table(DB, "EcosimScenarioArena")
print(f"\n=== EcosimScenarioArena: {len(sa)} rows ===")
print(sa.columns.tolist())
sa16 = (
    sa[sa.get("ScenarioID", sa.get("Scenario", pd.Series())) == 16]
    if "ScenarioID" in sa.columns
    else sa
)
print(f"Scenario 16 arena: {len(sa16)} rows")
if len(sa16) > 0:
    print(sa16.head(10).to_string())

# Forcing shapes
shapes = read_ewemdb_table(DB, "EcosimShape")
print(f"\n=== EcosimShape: {len(shapes)} rows ===")
print(shapes.columns.tolist())
print(shapes.head(10).to_string())
