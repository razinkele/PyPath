"""Test enhanced baselines step by step."""

import warnings

warnings.filterwarnings("ignore")
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = str(PROJECT_ROOT / "Data" / "LT2022_0.5ST_final7.eweaccdb")

from pypath import read_ewemdb, read_ewemdb_table, rpath, rsim_run, rsim_scenario

print("Loading model...")
params = read_ewemdb(DB_PATH)
model = rpath(params)
N = model.NUM_GROUPS

# Parse time series
ts_all = read_ewemdb_table(DB_PATH, "EcosimTimeSeries")
ds35 = ts_all[ts_all["DatasetID"] == 35].copy()
tsg = read_ewemdb_table(DB_PATH, "EcosimTimeSeriesGroup")
groups_db = read_ewemdb_table(DB_PATH, "EcopathGroup")
gid_to_name = dict(zip(groups_db["GroupID"], groups_db["GroupName"]))

ts_to_group = {}
for _, r in tsg.iterrows():
    db_gname = gid_to_name.get(r["GroupID"], "").strip()
    for g in range(N):
        if str(model.Group[g]).strip() == db_gname:
            ts_to_group[int(r["TimeSeriesID"])] = g
            break

observed_rel = {}
forced_bio = {}
effort_ts = None
for _, row in ds35.iterrows():
    vals = np.array([float(x) for x in str(row["TimeValues"]).split()])
    dtype, tsid = int(row["DatType"]), int(row["TimeSeriesID"])
    g = ts_to_group.get(tsid)
    if dtype == 0 and g is not None:
        observed_rel[g] = vals
    elif dtype == -1 and g is not None:
        forced_bio[g] = vals
    elif dtype == 3:
        effort_ts = vals

n_years = len(effort_ts)
YEARS = range(1, n_years + 1)

# Build VV overrides
fm = read_ewemdb_table(DB_PATH, "EcosimScenarioForcingMatrix")
fm16 = fm[fm["ScenarioID"] == 16]
esg = read_ewemdb_table(DB_PATH, "EcosimScenarioGroup")
esg16 = esg[esg["ScenarioID"] == 16]

egid_to_idx = {}
for _, row in groups_db.iterrows():
    gname = str(row["GroupName"]).strip()
    for g in range(N):
        if str(model.Group[g]).strip() == gname:
            egid_to_idx[int(row["GroupID"])] = g
            break

esim_to_idx = {}
for _, row in esg16.iterrows():
    esim_to_idx[int(row["GroupID"])] = egid_to_idx.get(int(row["EcopathGroupID"]), -1)

vv_overrides = {}
for _, row in fm16.iterrows():
    pg = esim_to_idx.get(int(row["PreyID"]), -1)
    dg = esim_to_idx.get(int(row["PredID"]), -1)
    if pg >= 0 and dg >= 0:
        vv_overrides[(pg, dg)] = float(row["vulnerability"])

print(
    f"VV overrides: {len(vv_overrides)}, non-default: {sum(1 for v in vv_overrides.values() if v != 2.0)}"
)
print(f"Forced bio groups: {list(forced_bio.keys())}")
for g, v in forced_bio.items():
    print(f"  g={g} {str(model.Group[g]).strip()}: {v}")


def compute_ss(output):
    total = 0.0
    per = {}
    for g, obs in observed_rel.items():
        col = g + 1
        sim = output.annual_Biomass[:, col] / model.Biomass[g]
        n = min(len(obs), len(sim))
        ss = np.sum((sim[:n] - obs[:n]) ** 2)
        per[g] = ss
        total += ss
    return total, per


# Test 1: Plain baseline
print("\n--- Test 1: Plain baseline ---")
t0 = time.time()
sc = rsim_scenario(model, params, years=YEARS)
for yr in range(n_years):
    for mo in range(12):
        t = yr * 12 + mo
        if t < sc.fishing.ForcedEffort.shape[0]:
            sc.fishing.ForcedEffort[t, :] = effort_ts[yr]
out = rsim_run(sc, method="AB")
ss, _ = compute_ss(out)
print(f"  SS={ss:.2f}  crashed={out.crashed_groups}  time={time.time() - t0:.1f}s")

# Test 2: + VV overrides
print("\n--- Test 2: + VV overrides ---")
t0 = time.time()
sc = rsim_scenario(model, params, years=YEARS)
for yr in range(n_years):
    for mo in range(12):
        t = yr * 12 + mo
        if t < sc.fishing.ForcedEffort.shape[0]:
            sc.fishing.ForcedEffort[t, :] = effort_ts[yr]
pf = sc.params.PreyFrom
pt = sc.params.PreyTo
applied = 0
for lk in range(len(pf)):
    pg = pf[lk] - 1
    dg = pt[lk] - 1
    if (pg, dg) in vv_overrides:
        sc.params.VV[lk] = vv_overrides[(pg, dg)]
        applied += 1
print(f"  Applied {applied} VV overrides")
out = rsim_run(sc, method="AB")
ss, _ = compute_ss(out)
print(f"  SS={ss:.2f}  crashed={out.crashed_groups}  time={time.time() - t0:.1f}s")

# Test 3: + Forced bio
print("\n--- Test 3: + VV overrides + forced bio ---")
t0 = time.time()
sc = rsim_scenario(model, params, years=YEARS)
for yr in range(n_years):
    for mo in range(12):
        t = yr * 12 + mo
        if t < sc.fishing.ForcedEffort.shape[0]:
            sc.fishing.ForcedEffort[t, :] = effort_ts[yr]
for lk in range(len(sc.params.PreyFrom)):
    pg = sc.params.PreyFrom[lk] - 1
    dg = sc.params.PreyTo[lk] - 1
    if (pg, dg) in vv_overrides:
        sc.params.VV[lk] = vv_overrides[(pg, dg)]

print(f"  ForcedBio shape: {sc.forcing.ForcedBio.shape}")
for g, bio_ts in forced_bio.items():
    col = g + 1
    print(f"  Applying forced bio for g={g} col={col}, values={bio_ts}")
    for yr in range(min(n_years, len(bio_ts))):
        for mo in range(12):
            t = yr * 12 + mo
            if (
                t < sc.forcing.ForcedBio.shape[0]
                and col < sc.forcing.ForcedBio.shape[1]
            ):
                sc.forcing.ForcedBio[t, col] = bio_ts[yr]
    print(f"  ForcedBio[:3, {col}] = {sc.forcing.ForcedBio[:3, col]}")

out = rsim_run(sc, method="AB")
ss, _ = compute_ss(out)
print(f"  SS={ss:.2f}  crashed={out.crashed_groups}  time={time.time() - t0:.1f}s")

print("\nAll tests done.")
