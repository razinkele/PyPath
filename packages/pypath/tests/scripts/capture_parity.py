"""Capture the parity paths' output, to check a refactor keeps it bit-identical.

Usage::

    capture_parity.py OUT.npz
    capture_parity.py --compare A.npz B.npz

The first form runs two years of each case below with ``rsim_run`` and saves
``out_Biomass``, ``out_Catch``, ``out_Gear_Catch``, ``out_Hden``,
``stanza_biomass`` and the end state (``Biomass``, ``N``, ``Ftime``) under
``<case>/<method>/<name>``; arrays a run returns as None are left out.

- REco (Rpath's reference model), effort 2: without stanzas under the
  ``"ewe"`` rule; with stanzas under the ``"ewe"`` and the ``"rpath"`` rule;
  without stanzas under the ``"ewe"`` rule with feeding time (FtimeAdj 0.5),
  RiskTime 0.3 on group 13, PaddP 0.5 and group 12 forced to 0.8 x B0 in
  months 1-6. RK4 and AB each.
- ``test_ewe_mediation._scenario`` with ``SHAPE_A`` mediating Zoo's
  availability to Fish (RK4).
- LT2022 scenario 16 (dataset 35) and the Black Sea scenario 1 (RK4), when
  ``Data/`` holds them and a database driver exists; otherwise they are
  skipped with a note.

``--compare`` prints ``identical: <n> arrays`` and exits 0 when both files
hold the same keys and every array is ``np.array_equal`` (NaN equal to NaN);
otherwise it prints the differing keys and exits 1.

It imports ``tests.*``, so run it with ``PYTHONPATH=<src>:packages/pypath``.
Compare captures taken on the same machine (machines differ by about 2e-6).
pytest never collects it: the name does not match ``test_*.py``.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

YEARS = range(1, 3)
DATA = Path(__file__).resolve().parents[4] / "Data"
LT2022 = DATA / "LT2022_0.5ST_final7.eweaccdb"
BLACK_SEA = DATA / "BlackSea_EwEF_testcase.eweaccdb"
OUTPUTS = ("out_Biomass", "out_Catch", "out_Gear_Catch", "out_Hden", "stanza_biomass")
END_STATE = ("Biomass", "N", "Ftime")


def _reco(stanzas, rule):
    from tests.continuous_support import reco_plain, reco_stanzas, scenario

    return scenario(reco_stanzas() if stanzas else reco_plain(), rule=rule)


def _reco_feeding_time():
    scen = _reco(False, "ewe")
    p = scen.params
    p.FtimeAdj[1 : p.NUM_LIVING + 1] = 0.5
    p.RiskTime[13] = 0.3
    p.PaddP[:] = 0.5
    scen.forcing.ForcedBio[0:6, 12] = 0.8 * p.B_BaseRef[12]
    return scen


def _mediated():
    from tests.test_ewe_mediation import (
        BENTHOS,
        FISH,
        SEARCH,
        SHAPE_A,
        ZOO,
        _mediation,
        _scenario,
    )

    scen = _scenario()
    weights = np.zeros(scen.params.NUM_GROUPS + 1)
    weights[BENTHOS] = 1.0
    scen.params.EweMediation = _mediation(
        SHAPE_A, 300, weights, [(ZOO, FISH, 0, SEARCH)], scen.params.B_BaseRef
    )
    return scen


def _ewe_db(path, **kwargs):
    """The scenario of an EwE database, or None (with the reason)."""
    from pypath.io.ewemdb import check_ewemdb_support, ecosim_scenario_from_ewemdb

    if not path.exists():
        return None, f"{path.name} not found"
    if not check_ewemdb_support().get("any_available"):
        return None, "no EwE database driver"
    return ecosim_scenario_from_ewemdb(str(path), **kwargs), None


# name -> (build, methods); a build returns a scenario, or (None, reason)
CASES = {
    "reco_plain_ewe": (lambda: _reco(False, "ewe"), ("RK4", "AB")),
    "reco_stanzas_ewe": (lambda: _reco(True, "ewe"), ("RK4", "AB")),
    "reco_stanzas_rpath": (lambda: _reco(True, "rpath"), ("RK4", "AB")),
    "reco_plain_ewe_ftime_risk_paddp_forced": (_reco_feeding_time, ("RK4", "AB")),
    "mediation_shape_a": (_mediated, ("RK4",)),
    "lt2022_s16": (
        lambda: _ewe_db(LT2022, scenario=16, timeseries_dataset=35),
        ("RK4",),
    ),
    "black_sea_s1": (lambda: _ewe_db(BLACK_SEA, scenario=1), ("RK4",)),
}


def capture(out_path):
    from pypath.core.ecosim import rsim_run

    arrays = {}
    for case, (build, methods) in CASES.items():
        for method in methods:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scen = build()  # a fresh scenario for every run
                if isinstance(scen, tuple):
                    scen, reason = scen
                    if scen is None:
                        print(f"skipped {case}: {reason}")
                        break
                out = rsim_run(scen, method=method, years=YEARS)
            for name in OUTPUTS:
                value = getattr(out, name)
                if value is not None:
                    arrays[f"{case}/{method}/{name}"] = np.asarray(value)
            for name in END_STATE:
                value = getattr(out.end_state, name)
                if value is not None:
                    arrays[f"{case}/{method}/end_state.{name}"] = np.asarray(value)
            print(f"captured {case} {method}")
    np.savez(out_path, **arrays)
    print(f"saved {len(arrays)} arrays to {out_path}")


def _equal(a, b):
    nan_ok = a.dtype.kind in "fc" and b.dtype.kind in "fc"
    return np.array_equal(a, b, equal_nan=nan_ok)


def compare(path_a, path_b):
    with np.load(path_a) as fa, np.load(path_b) as fb:
        keys_a, keys_b = set(fa.files), set(fb.files)
        differing = sorted(keys_a ^ keys_b)
        differing += sorted(k for k in keys_a & keys_b if not _equal(fa[k], fb[k]))
    if differing:
        print("differing:")
        for key in differing:
            where = "" if key in keys_a & keys_b else " (in one file only)"
            print(f"  {key}{where}")
        return 1
    print(f"identical: {len(keys_a)} arrays")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", help="OUT.npz, or A.npz B.npz")
    parser.add_argument("--compare", action="store_true", help="compare A and B")
    args = parser.parse_args(argv)
    if args.compare:
        if len(args.paths) != 2:
            parser.error("--compare takes two files")
        return compare(*args.paths)
    if len(args.paths) != 1:
        parser.error("capture takes one output file")
    capture(args.paths[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
