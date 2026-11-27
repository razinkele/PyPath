"""Compare PyPath's end-of-run biomass with EwE 6's (bench/ewe6_biomass.csv).

Usage:
    python scripts/compare_end_biomass.py --name LT2022 --database Data/LT2022_0.5ST_final7.eweaccdb --scenario 16
"""

import argparse
import csv
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from pypath.core.ecosim import rsim_run
from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--database", required=True)
ap.add_argument("--scenario", type=int, default=1)
ap.add_argument("--ewe-csv", default="bench/ewe6_biomass.csv")
args = ap.parse_args()

ewe = {}
years = None
with open(args.ewe_csv, newline="") as fh:
    ewe_rows = list(csv.DictReader(fh))
for r in ewe_rows:
    if r["model"] == args.name:
        ewe[r["group"].strip()] = float(r["biomass"])
        years = int(r["years"])
if not ewe:
    raise SystemExit(f"no rows for {args.name} in {args.ewe_csv}")

sc = ecosim_scenario_from_ewemdb(args.database, scenario=args.scenario)
first = int(sc.start_year) if hasattr(sc, "start_year") else 1
sc = ecosim_scenario_from_ewemdb(
    args.database, scenario=args.scenario, years=range(first, first + years)
)
end = np.asarray(rsim_run(sc, method="RK4").out_Biomass)[-1]
rows = []
for i, g in enumerate(sc.params.spname):
    g = str(g).strip()
    if g in ewe and ewe[g] > 0:
        rows.append((abs(end[i] / ewe[g] - 1), g, end[i], ewe[g]))
rows.sort(reverse=True)
rel = np.array([r[0] for r in rows])
print(
    f"{args.name}: {years} years, matched {len(rows)} groups; "
    f"median rel diff {np.median(rel):.2e}, max {rel.max():.2e}"
)
for d, g, p, e in rows[:5]:
    print(f"   {g:32s} pypath {p:.6g}  ewe6 {e:.6g}  diff {d:.2%}")
