"""Time PyPath's Ecospace (rsim_run_spatial) on an EwE database's Ecospace scenario.

Set up as tests/test_ecospace_ewe_calibration.py sets up the Black Sea run that
follows EwE 6's maps (EwE's habitat capacity and dispersal rates, from
read_ecospace). Prints seconds per run and per simulated year, the slope over the
run lengths, to compare with EweTimer --ecospace.

Usage:
    python scripts/benchmark_ecospace.py DATABASE ECOSIM_SCENARIO ECOSPACE_SCENARIO
        [--years 2 5 10] [--repeats 3] [--jobs 1 4]
"""

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from pypath.io.ewemdb import ecosim_scenario_from_ewemdb, read_ecospace
from pypath.spatial import EcospaceParams, rsim_run_spatial
from pypath.spatial.batched import batch_supported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("database")
    ap.add_argument("ecosim", type=int)
    ap.add_argument("ecospace", type=int)
    ap.add_argument("--years", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--jobs", type=int, nargs="+", default=[1])
    a = ap.parse_args()

    scenario = ecosim_scenario_from_ewemdb(a.database, scenario=a.ecosim)
    p = scenario.params
    e = read_ecospace(
        a.database, n_groups=p.NUM_LIVING + p.NUM_DEAD, scenario_id=a.ecospace
    ).ecospace
    n_rows, n_p = p.NUM_GROUPS, e.grid.n_patches
    hab = np.ones((n_rows, n_p))
    hab[: e.habitat_capacity.shape[0]] = e.habitat_capacity
    rates = np.zeros(n_rows)
    rates[: len(e.dispersal_rate)] = e.dispersal_rate
    eco = EcospaceParams(
        grid=e.grid,
        habitat_preference=np.ones((n_rows, n_p)),
        habitat_capacity=hab,
        dispersal_rate=rates,
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
    )
    first = int(scenario.start_year)
    from pypath.core.ecosim import _month_forcing, _run_setup

    setup = _run_setup(scenario, range(first, first + 1))
    batched = batch_supported(
        setup.params_dict, _month_forcing(scenario, setup, 1, setup.ftime)
    )
    print(
        f"{a.database}: Ecosim {a.ecosim}, Ecospace {a.ecospace}: {n_p} water cells, "
        f"{p.NUM_GROUPS} groups, monthly RK4, derivative "
        f"{'batched over cells' if batched else 'per cell (Python fallback)'}"
    )
    for jobs in a.jobs:
        # first call: numba compiles / caches warm up
        rsim_run_spatial(
            scenario, ecospace=eco, years=range(first, first + a.years[0]), n_jobs=jobs
        )
        best = {}
        for _ in range(a.repeats):
            for n in a.years:
                t = time.perf_counter()
                rsim_run_spatial(
                    scenario, ecospace=eco, years=range(first, first + n), n_jobs=jobs
                )
                dt = time.perf_counter() - t
                best[n] = min(best.get(n, np.inf), dt)
        ys = np.array(sorted(best))
        ts = np.array([best[n] for n in ys])
        slope, fixed = np.polyfit(ys, ts, 1)
        runs = ", ".join(f"{n} y {best[n]:.2f} s" for n in ys)
        print(
            f"  n_jobs {jobs}: {runs}; per simulated year {slope * 1000:.0f} ms (fixed {fixed * 1000:.0f} ms)"
        )


if __name__ == "__main__":  # Windows starts n_jobs workers by re-importing this file
    main()
