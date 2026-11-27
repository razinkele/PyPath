"""Write PyPath's example models as EwE 6 databases, for the engine benchmark
(``scripts/benchmark_engines.py database`` beside ``scripts/ewe6_timer``).

PyPath's example models (Curonian Lagoon, Central Baltic, Lithuanian Coast)
are EwE-F files, which EwE 6 does not read. This writes one as an EwE 6
database (``.eweaccdb``) from PyPath's blank EwE 6 template, then reads it back
and compares a PyPath run of the database with a PyPath run of the example, so
both engines are known to get the same model. Writing and reading ``.eweaccdb``
files needs pyodbc and the 64-bit Microsoft Access Database Engine (Windows).

Example::

    python scripts/convert_for_engines.py --example Curonian_Lagoon --out bench/Curonian_Lagoon.eweaccdb
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _run_biomass(scenario, years: int):
    from pypath.core.ecosim import rsim_run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rsim_run(scenario, method="RK4", years=range(1, years + 1))
    names = [str(s).strip() for s in scenario.params.spname]
    return names, np.asarray(out.out_Biomass)


def example_to_ewe6(example: str, out: str, years: int = 10) -> float:
    """Write ``example`` as an EwE 6 database at ``out``; return the largest
    relative biomass difference between PyPath's runs of the two."""
    from pypath.examples import example_ecosim_scenario, example_model
    from pypath.io.ewe_writer import write_ewemdb
    from pypath.io.ewef import read_ewef
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    params = read_ewef(example_model(example).ecopath_path)
    scenario = example_ecosim_scenario(example)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_ewemdb(params, out, scenarios=[scenario], backend="access")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = ecosim_scenario_from_ewemdb(out, scenario=1)
    n1, b1 = _run_biomass(example_ecosim_scenario(example), years)
    n2, b2 = _run_biomass(back, years)
    diffs = [
        float(
            np.max(
                np.abs(b1[:, n1.index(g)] - b2[:, j])
                / np.maximum(np.abs(b1[:, n1.index(g)]), 1e-12)
            )
        )
        for j, g in enumerate(n2)
        if j > 0 and g in n1
    ]
    return max(diffs) if diffs else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--example", required=True, help="e.g. Curonian_Lagoon")
    parser.add_argument("--out", required=True, help="output .eweaccdb")
    parser.add_argument("--years", type=int, default=10, help="length of the check run")
    args = parser.parse_args(argv)

    diff = example_to_ewe6(args.example, args.out, args.years)
    print(f"Wrote {args.out}")
    print(
        f"PyPath on the database beside PyPath on the example, {args.years} years: "
        f"largest relative biomass difference {diff:.3g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
