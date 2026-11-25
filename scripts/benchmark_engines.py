"""How fast PyPath runs Ecopath and Ecosim, beside EwE-F and EwE 6.

No one machine runs all three: EwE 6 needs Windows, and EwE-F is built
where its user builds it. So the three are measured as two pairs that
share a machine, with PyPath in both:

``ewef``
    PyPath and EwE-F, on EwE-F's own test cases. EwE-F's executables are
    run from a scratch copy of each case (they write beside their
    namelist, so never inside EwE-F's own folder). With ``--build`` its
    Ecosim is first built from its sources as they are now, with the
    Makefile's flags and with ``-O2``: an executable left in EwE-F's
    folder may be older than its sources, and then it is not the same
    model that is timed.
``database``
    PyPath alone, on an EwE database: the half of the second pair that
    runs wherever EwE 6 does. EwE 6's own times come from a program that
    drives its core and writes a CSV of the same shape as ``--ewe-csv``.
``report``
    The table, from one or more result files.

What is measured
----------------
Ecosim
    Seconds per simulated year: the slope of wall time over the run
    lengths of ``--years``. A slope leaves out start-up, reading and
    set-up, so it needs no clock inside EwE-F. The fit's R2 says whether
    the times lie on a line. The slope is taken through the fastest
    repeat of each length: on a machine that does other work as well,
    that work can only add time.
Ecopath
    The solve alone, in process, for PyPath (and EwE 6). EwE-F's solve
    cannot be seen apart from its executable, so the whole run of that
    executable stands in its own row, beside PyPath's whole run from the
    same files in a new process.

The first call is kept apart from the rest (numba compiles on it); the
others give a median and a range, shown for single timings. Every number is in the result file with
the configuration it was measured under. Agreement between the engines'
biomasses is recorded as a check that they ran the same case, not as a
parity result.

Nothing of EwE or EwE-F is in this repository: the script takes paths.

Examples
--------
::

    python scripts/benchmark_engines.py ewef --ewef-root /path/to/EwE-F --build \\
        --cases Black_Sea Central_Baltic --work /tmp/bench --out ewef.json
    python scripts/benchmark_engines.py database --database model.eweaccdb \\
        --scenario 1 --out database.json
    python scripts/benchmark_engines.py report ewef.json database.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# The variables of the namelist a non-spatial Ecosim-F reads (ecosim.F90).
# A case's own namelist may name spatial files too; those are left out.
EWEF_ECOSIM_KEYS = (
    "HDF5_fname",
    "GroupInfo_fname",
    "Vulnerability_fname",
    "Forcing_fname",
    "NutrientForcing_fname",
    "PrimaryProdForcing_fname",
    "tf",
    "StepsPerMonth",
    "NutBaseFreeProp",
    "NutPBmax",
    "relax",
    "ForagingTimeLowerLimit",
    "VulnerabilityCap",
    "RateForcing_fname",
    "Mediation_fname",
    "MediationOut_fname",
    "DiscardFlow_fname",
)
# Files with one row per month that this script knows how to lengthen
EWEF_MONTHLY_FILES = ("Forcing_fname", "NutrientForcing_fname")
# ... and those it does not: a case that names one is run only as long as it is
EWEF_OTHER_SERIES = (
    "PrimaryProdForcing_fname",
    "RateForcing_fname",
    "Mediation_fname",
    "DiscardFlow_fname",
)


# --- timing -------------------------------------------------------------------


def time_calls(call: Callable[[], object], repeats: int) -> Dict[str, object]:
    """Time ``call`` once (kept apart) and then ``repeats`` times.

    Returns the first time, the others, and their median, minimum and
    maximum, in seconds; ``result`` is what the last call returned.
    """
    times = []
    result = None
    for _ in range(repeats + 1):
        start = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - start)
    rest = times[1:]
    return {
        "first": times[0],
        "times": rest,
        "median": float(np.median(rest)),
        "min": float(min(rest)),
        "max": float(max(rest)),
        "result": result,
    }


def fit_line(years: Sequence[float], seconds: Sequence[float]) -> Dict[str, float]:
    """Seconds = intercept + slope x years, by least squares, with its R2."""
    x, y = np.asarray(years, dtype=float), np.asarray(seconds, dtype=float)
    if len(x) < 2:
        raise ValueError("a slope needs at least two run lengths")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else float("nan")
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}


def _plain(timing: Dict[str, object]) -> Dict[str, object]:
    """A timing without the object it returned, for the result file."""
    return {k: v for k, v in timing.items() if k != "result"}


def environment() -> Dict[str, object]:
    """The machine and the libraries, as far as they bear on speed."""
    import numba
    import pypath
    import scipy

    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    load = None
    if hasattr(os, "getloadavg"):
        load = [round(v, 2) for v in os.getloadavg()]
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "load_average": load,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "pypath": pypath.__version__,
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "numba_consumption": os.environ.get("PYPATH_NUMBA_CONSUMPTION", "(default)"),
    }


# --- PyPath -------------------------------------------------------------------


def pypath_ecopath(params, repeats: int) -> Dict[str, object]:
    """The mass-balance solve alone: ``rpath(params)``."""
    from pypath.core.ecopath import rpath

    timing = time_calls(lambda: rpath(params), repeats)
    model = timing["result"]
    out = _plain(timing)
    out["groups"] = int(model.NUM_GROUPS)
    out["living"] = int(model.NUM_LIVING)
    return out


def pypath_ecosim(
    build: Callable[[int], object],
    years: Sequence[int],
    repeats: int,
    methods: Sequence[str] = ("RK4", "AB"),
) -> Dict[str, object]:
    """Ecosim's time per simulated year, by method.

    ``build(n_years)`` makes the scenario. Each run starts from a copy of
    it, made outside the clock, so that no run inherits another's state.
    """
    from pypath.core.ecosim import rsim_run

    out: Dict[str, object] = {"years": list(years), "methods": {}}
    scenarios = {}
    for n in years:
        start = time.perf_counter()
        scenarios[n] = build(n)
        out.setdefault("scenario_seconds", {})[str(n)] = time.perf_counter() - start
    # not stored on a scenario: the EwE rule alone gives it a nutrient pool
    nutrient = getattr(scenarios[years[0]].params, "NutBaseFreeProp", None)
    out["fast_group_rule"] = "rpath" if nutrient is None else "ewe"
    end_biomass = {}
    for method in methods:
        runs = {}
        for n in years:
            fresh = [copy.deepcopy(scenarios[n]) for _ in range(repeats + 1)]
            # a database's scenario may begin in its own first year
            first = int(getattr(scenarios[n], "start_year", 1) or 1)
            span = range(first, first + n)
            timing = time_calls(
                lambda span=span: rsim_run(fresh.pop(), method=method, years=span),
                repeats,
            )
            biomass = np.asarray(timing["result"].out_Biomass)
            runs[str(n)] = _plain(timing)
            runs[str(n)]["finite"] = bool(np.isfinite(biomass).all())
            runs[str(n)]["months"] = int(len(biomass) - 1)
            end_biomass[(method, n)] = biomass[-1]
        fit = fit_line(years, [runs[str(n)]["min"] for n in years])
        out["methods"][method] = {"runs": runs, "fit": fit}
    out["_end_biomass"] = end_biomass
    out["_names"] = [str(s).strip() for s in scenarios[years[0]].params.spname]
    return out


_WHOLE_RUN = """
import sys, warnings, logging
warnings.simplefilter("ignore"); logging.disable(logging.CRITICAL)
from pypath.io.ewef import read_ewef, ecosim_scenario_from_ewef
from pypath.core.ecopath import rpath
mode, ecopath, ecosim, years, out = sys.argv[1:6]
params = read_ewef(ecopath)
model = rpath(params)
if mode == "ecopath":
    model.summary().to_csv(out)
else:
    from pypath.core.ecosim import rsim_run
    import numpy as np
    n = int(years)
    scenario = ecosim_scenario_from_ewef(ecopath, ecosim, years=range(1, n + 1))
    np.savetxt(out, rsim_run(scenario, method="RK4", years=range(1, n + 1)).out_Biomass)
"""


def pypath_whole_run(
    mode: str, ecopath_dir: Path, ecosim_dir: Path, years: int, work: Path, repeats: int
) -> Dict[str, object]:
    """A whole run from files in a new Python process: start, import, read,
    solve (and simulate), write. What stands beside EwE-F's executable."""
    out_file = work / f"pypath_{mode}.out"
    command = [
        sys.executable,
        "-c",
        _WHOLE_RUN,
        mode,
        str(ecopath_dir),
        str(ecosim_dir),
        str(years),
        str(out_file),
    ]

    def run():
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(
                f"PyPath's whole {mode} run failed: {done.stderr[-400:]}"
            )
        return out_file.stat().st_size

    return _plain(time_calls(run, repeats))


# --- EwE-F --------------------------------------------------------------------


def read_namelist(path: Path) -> Dict[str, str]:
    """The ``name = value`` pairs of an EwE-F namelist, values as written."""
    pairs = {}
    for line in path.read_text().splitlines():
        line = line.split("!")[0].strip()
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*,?$", line)
        if match:
            pairs[match.group(1)] = match.group(2)
    return pairs


def _unquoted(value: str) -> str:
    return value.strip().strip("\"'")


def lengthen_monthly_file(path: Path, months: int) -> int:
    """Give a monthly EwE-F file at least ``months`` rows, by repeating its own.

    Two layouts are known: the forcing table (``"Number of months",N`` and
    rows ``"Month k",...``) and the nutrient series (``nrows N`` and rows
    ``month_k value``). Returns the number of months the file had.
    """
    lines = path.read_text().splitlines()
    head = lines[0]
    if head.lower().startswith('"number of months"'):
        first = next(k for k, line in enumerate(lines) if line.startswith('"Month '))
        have = len(lines) - first
        if have >= months:
            return have
        rows = [line.split(",", 1)[1] for line in lines[first:]]
        tail = head.split(",", 2)[2:]  # trailing commas of a padded header row
        lines[0] = ",".join(['"Number of months"', str(months)] + tail)
        for k in range(have, months):
            lines.append(f'"Month {k + 1}",{rows[k % have]}')
    elif head.lower().startswith("nrows"):
        first = next(k for k, line in enumerate(lines) if line.startswith("month_"))
        have = len(lines) - first
        if have >= months:
            return have
        values = [line.split(None, 1)[1] for line in lines[first:]]
        lines[0] = f"nrows {months}"
        for k in range(have, months):
            lines.append(f"month_{k + 1} {values[k % have]}")
    else:
        raise ValueError(f"{path.name}: not a monthly file this script knows")
    path.write_text("\n".join(lines) + "\n")
    return have


def prepare_case(
    ewef_root: Path, case: str, work: Path, months: int
) -> Dict[str, object]:
    """A scratch copy of one EwE-F test case, its monthly files long enough.

    Returns the run folders and what was done. Both engines read the copy,
    so they see the same forcing however long the run.
    """
    info: Dict[str, object] = {"case": case, "lengthened": {}}
    for part, source in (("ecopath", "ecopath-f"), ("ecosim", "ecosim-f")):
        run_dir = work / case / part
        target = run_dir / "testcases" / case
        if run_dir.exists():
            shutil.rmtree(run_dir)
        target.parent.mkdir(parents=True)
        shutil.copytree(ewef_root / source / "testcases" / case, target)
        info[f"{part}_run"] = run_dir
        info[f"{part}_case"] = target
    ecosim_case = info["ecosim_case"]
    namelist = read_namelist(ecosim_case / f"{case}_filenames.nml")
    info["namelist"] = namelist
    info["capped_by"] = [k for k in EWEF_OTHER_SERIES if _unquoted(namelist.get(k, ""))]
    for key in EWEF_MONTHLY_FILES:
        name = _unquoted(namelist.get(key, ""))
        if name:
            had = lengthen_monthly_file(info["ecosim_run"] / name, months)
            info["lengthened"][key] = {"had": had, "now": max(had, months)}
    shutil.copy(
        info["ecopath_case"] / f"{case}_filenames.nml",
        info["ecopath_run"] / "filenames.nml",
    )
    return info


def write_ecosim_namelist(run_dir: Path, namelist: Dict[str, str], years: int) -> None:
    lines = ["&filenames"]
    for key in EWEF_ECOSIM_KEYS:
        if key == "tf":
            lines.append(f"tf = {years}")
        elif key in namelist:
            lines.append(f"{key} = {namelist[key]}")
    (run_dir / "filenames.nml").write_text("\n".join(lines + ["/"]) + "\n")


def run_executable(binary: Path, run_dir: Path) -> None:
    """Run an EwE-F executable in its folder; its console output is dropped."""
    done = subprocess.run(
        [str(binary)],
        cwd=run_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"{binary.name} ended with code {done.returncode}: {done.stderr[:300]}"
        )


def read_ewef_monthly(path: Path):
    """EwE-F Ecosim's monthly biomasses: group names and the table."""
    rows = path.read_text().splitlines()
    names = [cell.strip() for cell in rows[0].split(",")][1:]
    table = np.array(
        [
            [float(v) for v in row.split(",") if v.strip()]
            for row in rows[1:]
            if row.strip()
        ]
    )
    return names, table[:, 1:]


def ewef_ecosim(
    binary: Path, case: Dict[str, object], years: Sequence[int], repeats: int
) -> Dict[str, object]:
    """EwE-F Ecosim's whole-run time by run length, and the slope."""
    run_dir = case["ecosim_run"]
    out: Dict[str, object] = {"years": list(years), "runs": {}}
    end = {}
    for n in years:
        write_ecosim_namelist(run_dir, case["namelist"], n)
        timing = time_calls(lambda: run_executable(binary, run_dir), repeats)
        names, table = read_ewef_monthly(run_dir / "Ecosim_absResults_monthly.dat")
        out["runs"][str(n)] = _plain(timing)
        out["runs"][str(n)]["finite"] = bool(np.isfinite(table).all())
        out["runs"][str(n)]["months"] = int(len(table) - 1)
        end[n] = dict(zip(names, table[-1]))
    out["fit"] = fit_line(years, [out["runs"][str(n)]["min"] for n in years])
    out["_end_biomass"] = end
    return out


def agreement(pypath_names, pypath_end, ewef_end) -> Dict[str, object]:
    """How far PyPath's end biomasses are from EwE-F's, group by group."""
    index = {name: k for k, name in enumerate(pypath_names)}
    differences = {}
    for name, value in ewef_end.items():
        if name in index and value > 0:
            differences[name] = abs(pypath_end[index[name]] / value - 1.0)
    if not differences:
        return {"groups": 0}
    worst = max(differences, key=differences.get)
    return {
        "groups": len(differences),
        "median": float(np.median(list(differences.values()))),
        "max": float(differences[worst]),
        "worst": worst,
    }


# Above this median difference of end biomasses the two engines are not
# running the same case, and their ratio compares cost on a model of that
# size, not the same trajectory
SAME_CASE_MEDIAN = 0.01


def same_case(check: Dict[str, object]) -> str:
    """The agreement check as a sentence."""
    if not check.get("groups"):
        return "Same-case check: no group could be matched by name."
    verdict = (
        "the two ran the same case"
        if check["median"] <= SAME_CASE_MEDIAN
        else "the two did NOT run the same case, so the ratio is of cost on a "
        "model of this size, not of the same trajectory"
    )
    return (
        f"Same-case check: end biomasses differ by a median "
        f"{100 * check['median']:.3f} % (largest {100 * check['max']:.2f} %, "
        f"{check['worst']}; {check['groups']} groups): {verdict}."
    )


def binary_facts(binary: Path, makefile: Optional[Path]) -> Dict[str, object]:
    """What can be said of an executable's build without rebuilding it."""
    facts: Dict[str, object] = {
        "path": str(binary),
        "bytes": binary.stat().st_size,
        "built": time.strftime("%Y-%m-%d", time.localtime(binary.stat().st_mtime)),
    }
    if makefile is not None and makefile.is_file():
        facts["makefile"] = [
            line.strip()
            for line in makefile.read_text(errors="replace").splitlines()
            if re.match(r"\s*(FC|FCFLAGS|CFLAGS|LFLAGS|CPPDEFS)\s*=", line)
        ]
    return facts


def build_ecosim(ewef_root: Path, work: Path, optimise: bool) -> Dict[str, object]:
    """Build EwE-F's non-spatial Ecosim from its sources as they are now.

    The sources and the Makefile are copied to the work folder and built
    there; nothing is written in EwE-F's own folder and no source is
    changed. The preprocessor flags are the Makefile's without
    ``_Ecospace_`` (the spatial build reads another namelist). With
    ``optimise`` the copy's compile and link flags become ``-O2`` in place
    of the Makefile's debugging ones.

    A build of one's own is what makes the comparison fair: an executable
    left in EwE-F's folder may be older than its sources.
    """
    label = "optimised (-O2)" if optimise else "the Makefile's flags"
    folder = work / "_build" / ("O2" if optimise else "makefile")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copytree(ewef_root / "ecosim-f" / "src", folder / "src")
    for name in ("obj", "mod", "bin"):
        (folder / name).mkdir()
    lines = (
        (ewef_root / "ecosim-f" / "Makefile").read_text(errors="replace").splitlines()
    )
    defines = []
    for k, line in enumerate(lines):
        define = re.match(r"\s*CPPDEFS\s*\+?=\s*(\S.*)", line)
        if define and "_Ecospace_" not in define.group(1):
            defines.append(define.group(1).strip())
        if optimise and re.match(r"\s*CFLAGS\s*=", line):
            lines[k] = " CFLAGS = -c -O2 $(FCFLAGS)"
        elif optimise and re.match(r"\s*LFLAGS\s*=", line):
            lines[k] = " LFLAGS = -O2 $(FCFLAGS)"
    (folder / "Makefile").write_text("\n".join(lines) + "\n")
    done = subprocess.run(
        ["make", f"CPPDEFS={' '.join(defines)}"],
        cwd=folder,
        capture_output=True,
        text=True,
    )
    binary = folder / "bin" / "ecosim.out"
    if done.returncode != 0 or not binary.is_file():
        raise RuntimeError(f"EwE-F did not build ({label}): {done.stdout[-600:]}")
    compiler = subprocess.run(["gfortran", "--version"], capture_output=True, text=True)
    facts = binary_facts(binary, folder / "Makefile")
    facts["label"] = label
    facts["defines"] = defines
    facts["compiler"] = compiler.stdout.splitlines()[0] if compiler.stdout else "?"
    facts["sources"] = "EwE-F's sources as they are now"
    return facts


def run_ewef(args) -> Dict[str, object]:
    from pypath.core.ecopath import rpath
    from pypath.io.ewef import ecosim_scenario_from_ewef, read_ewef

    root = Path(args.ewef_root).expanduser()
    work = Path(args.work).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    ecopath_bin = root / "ecopath-f" / "bin" / "ecopath.out"
    years = sorted(args.years)

    # the Ecosim executables to time, by label
    builds: Dict[str, Dict[str, object]] = {}
    for item in args.ecosim_binary or []:
        label, _, path = item.partition("=")
        builds[label] = binary_facts(Path(path).expanduser(), None)
    if args.build:
        for optimise in (False, True):
            facts = build_ecosim(root, work, optimise)
            builds[facts["label"]] = facts
            print(f"built EwE-F Ecosim, {facts['label']}", flush=True)
    if not builds:
        as_built = root / "ecosim-f" / "bin-api" / "ecosim.out"
        builds["as found in bin-api"] = binary_facts(
            as_built, root / "ecosim-f" / "Makefile"
        )

    result: Dict[str, object] = {
        "kind": "ewef",
        "environment": environment(),
        "ewef": {
            "ecopath": binary_facts(ecopath_bin, root / "ecopath-f" / "Makefile"),
            "ecosim": builds,
            "console_output": "dropped (stdout to the null device)",
        },
        "years": years,
        "repeats": args.repeats,
        "cases": {},
    }
    for name in args.cases:
        print(f"{name} ...", flush=True)
        case = prepare_case(root, name, work, 12 * max(years))
        if case["capped_by"]:
            print(f"  skipped: it names {case['capped_by']}, which are not lengthened")
            result["cases"][name] = {"skipped": case["capped_by"]}
            continue
        ecopath_dir, ecosim_dir = case["ecopath_case"], case["ecosim_case"]
        entry: Dict[str, object] = {"lengthened": case["lengthened"]}
        # Ecopath: the solve alone, then each engine's whole run from files
        params = read_ewef(ecopath_dir)
        entry["pypath_ecopath_solve"] = pypath_ecopath(params, args.repeats)
        entry["ewef_ecopath_whole"] = _plain(
            time_calls(
                lambda: run_executable(ecopath_bin, case["ecopath_run"]), args.repeats
            )
        )
        entry["pypath_ecopath_whole"] = pypath_whole_run(
            "ecopath", ecopath_dir, ecosim_dir, 0, work, args.whole_repeats
        )
        # Ecosim: the slope over run lengths, and a whole run at the shortest
        ours = pypath_ecosim(
            lambda n: ecosim_scenario_from_ewef(
                ecopath_dir, ecosim_dir, years=range(1, n + 1)
            ),
            years,
            args.repeats,
        )
        names, end = ours.pop("_names"), ours.pop("_end_biomass")
        entry["pypath_ecosim"] = ours
        entry["pypath_ecosim_whole"] = pypath_whole_run(
            "ecosim", ecopath_dir, ecosim_dir, years[0], work, args.whole_repeats
        )
        entry["ewef_ecosim"], entry["agreement"] = {}, {}
        for label, facts in builds.items():
            theirs = ewef_ecosim(Path(facts["path"]), case, years, args.repeats)
            ewef_end = theirs.pop("_end_biomass")
            entry["ewef_ecosim"][label] = theirs
            entry["agreement"][label] = {
                str(n): agreement(names, end[("RK4", n)], ewef_end[n]) for n in years
            }
        entry["ewef_steps_per_month"] = _unquoted(
            case["namelist"].get("StepsPerMonth", "?")
        )
        entry["groups"] = int(rpath(params).NUM_GROUPS)
        result["cases"][name] = entry
    result["environment"]["load_average_end"] = environment()["load_average"]
    return result


# --- an EwE database ------------------------------------------------------------


def run_database(args) -> Dict[str, object]:
    from pypath.core.ecopath import rpath
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb, read_ewemdb

    database = str(Path(args.database).expanduser())
    years = sorted(args.years)
    start = time.perf_counter()
    params = read_ewemdb(database)
    read_seconds = time.perf_counter() - start
    dataset = args.dataset if args.dataset else None
    ours = pypath_ecosim(
        lambda n: ecosim_scenario_from_ewemdb(
            database,
            scenario=args.scenario,
            years=range(1, n + 1),
            timeseries_dataset=dataset,
        ),
        years,
        args.repeats,
    )
    ours.pop("_names"), ours.pop("_end_biomass")
    return {
        "kind": "database",
        "environment": environment(),
        "database": Path(database).name,
        "scenario": args.scenario,
        "dataset": dataset,
        "years": years,
        "repeats": args.repeats,
        "read_seconds": read_seconds,
        "groups": int(rpath(params).NUM_GROUPS),
        "pypath_ecopath_solve": pypath_ecopath(params, args.repeats),
        "pypath_ecosim": ours,
    }


# --- the table ------------------------------------------------------------------


def _ms(seconds: float) -> str:
    return f"{1e3 * seconds:,.2f}" if seconds < 0.1 else f"{1e3 * seconds:,.0f}"


def _spread(timing: Dict[str, object]) -> str:
    return (
        f"{_ms(timing['median'])} ms ({_ms(timing['min'])} to {_ms(timing['max'])}; "
        f"first {_ms(timing['first'])})"
    )


def _slope(fit: Dict[str, float]) -> str:
    return (
        f"{1e3 * fit['slope']:,.2f} ms per year "
        f"(fixed {1e3 * fit['intercept']:,.0f} ms, R2 {fit['r2']:.4f})"
    )


def read_ewe_csv(path: Path) -> Dict[str, Dict[str, object]]:
    """EwE 6's times: rows ``model,what,years,repeat,seconds``.

    ``what`` is ``ecopath`` (repeat 0 is the first call), ``ecosim`` (repeats
    from 1) or ``ecosim_warmup`` (runs before the timed ones: .NET compiles
    in tiers, so a method is fast only after many calls). Returns by model
    the Ecopath timing and the Ecosim fit over run lengths.
    """
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: Dict[str, Dict[str, object]] = {}
    for model in sorted({row["model"] for row in rows}):
        mine = [row for row in rows if row["model"] == model]
        entry: Dict[str, object] = {}

        def timing(what, years=None):
            picked = [
                row
                for row in mine
                if row["what"] == what and (years is None or int(row["years"]) == years)
            ]
            first = [float(r["seconds"]) for r in picked if int(r["repeat"]) == 0]
            rest = [float(r["seconds"]) for r in picked if int(r["repeat"]) > 0]
            if what == "ecosim":  # its first calls are the warm-up rows
                first = [
                    float(r["seconds"]) for r in mine if r["what"] == "ecosim_warmup"
                ]
            return {
                "first": first[0] if first else float("nan"),
                "median": float(np.median(rest)),
                "min": min(rest),
                "max": max(rest),
            }

        if any(row["what"] == "ecopath" for row in mine):
            entry["ecopath"] = timing("ecopath")
        lengths = sorted({int(row["years"]) for row in mine if row["what"] == "ecosim"})
        if lengths:
            runs = {str(n): timing("ecosim", n) for n in lengths}
            entry["ecosim"] = {
                "runs": runs,
                "fit": fit_line(lengths, [runs[str(n)]["min"] for n in lengths]),
            }
        out[model] = entry
    return out


def report(paths: Sequence[str], ewe_csv: Optional[str]) -> str:
    lines: List[str] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        env = data["environment"]
        lines += [
            f"## {Path(path).name}: {data['kind']}",
            "",
            f"{env['cpu']}, {env['cores']} cores, load {env['load_average']}; "
            f"{env['platform']}; Python {env['python']}, numpy {env['numpy']}, "
            f"numba {env['numba']}, pypath {env['pypath']}; "
            f"{data['repeats']} repeats after the first call",
            "",
        ]
        if data["kind"] == "ewef":
            facts = data["ewef"]["ecopath"]
            lines.append(f"EwE-F Ecopath: {facts['path']} (built {facts['built']})")
            for label, facts in data["ewef"]["ecosim"].items():
                detail = [f"built {facts['built']}"]
                if "compiler" in facts:
                    detail += [facts["compiler"], facts["sources"]]
                    detail += ["defines " + " ".join(facts["defines"])]
                detail += [x for x in facts.get("makefile", []) if "FLAGS" in x]
                lines.append(f"EwE-F Ecosim, {label}: " + "; ".join(detail))
            lines.append("")
            shortest = str(data["years"][0])
            for name, case in data["cases"].items():
                if "skipped" in case:
                    lines += [f"### {name}: skipped ({case['skipped']})", ""]
                    continue
                ours, builds = case["pypath_ecosim"], case["ewef_ecosim"]
                labels = list(builds)
                head = " | ".join(f"EwE-F, {label}" for label in labels)

                def row(title, mine, theirs):
                    return f"| {title} | {mine} | " + " | ".join(theirs) + " |"

                lines += [
                    f"### {name} ({case['groups']} groups)",
                    "",
                    f"| | PyPath | {head} |",
                    "|---|---|" + "---|" * len(labels),
                    row(
                        "Ecopath, the solve alone",
                        _spread(case["pypath_ecopath_solve"]),
                        ["not observable"] * len(labels),
                    ),
                    row(
                        "Ecopath, a whole run from files",
                        _spread(case["pypath_ecopath_whole"]),
                        [_spread(case["ewef_ecopath_whole"])]
                        + ["(the same)"] * (len(labels) - 1),
                    ),
                    row(
                        "Ecosim RK4, per simulated year",
                        _slope(ours["methods"]["RK4"]["fit"]),
                        [_slope(builds[label]["fit"]) for label in labels],
                    ),
                    row(
                        "Ecosim AB, per simulated year",
                        _slope(ours["methods"]["AB"]["fit"]),
                        ["no such method"] * len(labels),
                    ),
                    row(
                        f"Ecosim, a whole run of {shortest} years from files",
                        _spread(case["pypath_ecosim_whole"]),
                        [_spread(builds[label]["runs"][shortest]) for label in labels],
                    ),
                    "",
                    f"PyPath rule: {ours['fast_group_rule']}; EwE-F steps per month: "
                    f"{case['ewef_steps_per_month']}.",
                ]
                mine = ours["methods"]["RK4"]["fit"]["slope"]
                for label in labels:
                    ratio = mine / builds[label]["fit"]["slope"]
                    check = case["agreement"][label][shortest]
                    lines.append(
                        f"- EwE-F, {label}: per simulated year (RK4) PyPath takes "
                        f"{ratio:.1f} times its time. After {shortest} years: "
                        f"{same_case(check)}"
                    )
                lines.append("")
        else:
            ours = data["pypath_ecosim"]
            lines += [
                f"### {data['database']}, scenario {data['scenario']} ({data['groups']} groups)",
                "",
                "| | PyPath |",
                "|---|---|",
                f"| Ecopath, the solve alone | {_spread(data['pypath_ecopath_solve'])} |",
                f"| Ecosim RK4, per simulated year | {_slope(ours['methods']['RK4']['fit'])} |",
                f"| Ecosim AB, per simulated year | {_slope(ours['methods']['AB']['fit'])} |",
                "",
                f"PyPath rule: {ours['fast_group_rule']}.",
                "",
            ]
    if ewe_csv:
        lines += ["## EwE 6", ""]
        for model, entry in read_ewe_csv(Path(ewe_csv)).items():
            lines.append(f"### {model}")
            lines.append("")
            if "ecopath" in entry:
                lines.append(f"- Ecopath, the solve alone: {_spread(entry['ecopath'])}")
            if "ecosim" in entry:
                lines.append(
                    f"- Ecosim, per simulated year: {_slope(entry['ecosim']['fit'])}"
                )
            lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    modes = parser.add_subparsers(dest="mode", required=True)

    def common(sub):
        sub.add_argument("--years", type=int, nargs="+", default=[10, 20, 40])
        sub.add_argument("--repeats", type=int, default=5)
        sub.add_argument("--out", required=True, help="result file (JSON)")

    ewef = modes.add_parser("ewef", help="PyPath beside EwE-F on EwE-F's test cases")
    ewef.add_argument("--ewef-root", required=True, help="EwE-F's folder")
    ewef.add_argument("--cases", nargs="+", default=["Black_Sea"])
    ewef.add_argument("--work", required=True, help="scratch folder for the copies")
    ewef.add_argument(
        "--build",
        action="store_true",
        help="build EwE-F's non-spatial Ecosim from its sources as they are now, "
        "in the work folder, with the Makefile's flags and with -O2, and time both",
    )
    ewef.add_argument(
        "--ecosim-binary",
        nargs="*",
        metavar="LABEL=PATH",
        help="EwE-F Ecosim executables to time as they are; without this and "
        "without --build: the one in ecosim-f/bin-api",
    )
    ewef.add_argument("--whole-repeats", type=int, default=3)
    common(ewef)

    database = modes.add_parser("database", help="PyPath on an EwE database")
    database.add_argument("--database", required=True)
    database.add_argument("--scenario", type=int, default=1)
    database.add_argument("--dataset", type=int, default=0, help="0: none")
    common(database)

    table = modes.add_parser("report", help="the table, from result files")
    table.add_argument("results", nargs="+")
    table.add_argument(
        "--ewe-csv", help="EwE 6's times (model,what,years,repeat,seconds)"
    )

    args = parser.parse_args(argv)
    if args.mode == "report":
        print(report(args.results, args.ewe_csv))
        return 0
    warnings.simplefilter("ignore")
    logging.disable(logging.CRITICAL)
    result = run_ewef(args) if args.mode == "ewef" else run_database(args)
    Path(args.out).write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8"
    )
    print(report([args.out], None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
