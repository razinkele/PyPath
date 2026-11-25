"""The pure helpers of scripts/benchmark_engines.py.

The script times PyPath beside EwE-F and EwE 6; what it measures depends on
the machine, but how it fits a slope, lengthens a monthly file and reads a
table of times does not. The files here are written by the tests.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_engines.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file(), reason="the repository's scripts are not in this tree"
)


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("benchmark_engines", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_first_call_is_kept_apart_from_the_repeats(bench):
    calls = []
    timing = bench.time_calls(lambda: calls.append(1) or len(calls), 4)
    assert len(calls) == 5 and timing["result"] == 5
    assert len(timing["times"]) == 4
    assert timing["min"] <= timing["median"] <= timing["max"]
    assert "result" not in bench._plain(timing)


def test_a_slope_leaves_the_fixed_part_out(bench):
    years = [10, 20, 40]
    fit = bench.fit_line(years, [0.5 + 0.003 * n for n in years])
    assert fit["slope"] == pytest.approx(0.003) and fit["intercept"] == pytest.approx(
        0.5
    )
    assert fit["r2"] == pytest.approx(1.0)
    off = bench.fit_line(years, [0.05, 0.12, 0.07])  # not a line
    assert off["r2"] < 0.5
    with pytest.raises(ValueError, match="at least two"):
        bench.fit_line([10], [1.0])


def test_a_forcing_table_is_lengthened_by_repeating_its_own_months(bench, tmp_path):
    path = tmp_path / "Case_F.csv"
    path.write_text(
        '"Number of months",3,,\n"Number of groups",2\n"Group number",1,2\n'
        '"Forcing type",5,5\n"Group name","A","B"\n'
        '"Month 1",1,2\n"Month 2",3,4\n"Month 3",5,6\n'
    )
    assert bench.lengthen_monthly_file(path, 7) == 3
    lines = path.read_text().splitlines()
    assert lines[0] == '"Number of months",7,,'  # the padding of the row is kept
    assert lines[5:] == [
        '"Month 1",1,2',
        '"Month 2",3,4',
        '"Month 3",5,6',
        '"Month 4",1,2',
        '"Month 5",3,4',
        '"Month 6",5,6',
        '"Month 7",1,2',
    ]
    before = path.read_text()
    assert bench.lengthen_monthly_file(path, 5) == 7  # long enough: untouched
    assert path.read_text() == before


def test_a_nutrient_series_is_lengthened_too_and_other_files_are_refused(
    bench, tmp_path
):
    path = tmp_path / "Case_NutForce.csv"
    path.write_text(
        "nrows 2\nncols 1\nnutrient_concentration\nmonth_1 0.85\nmonth_2 1.0\n"
    )
    assert bench.lengthen_monthly_file(path, 5) == 2
    assert path.read_text().splitlines() == [
        "nrows 5",
        "ncols 1",
        "nutrient_concentration",
        "month_1 0.85",
        "month_2 1.0",
        "month_3 0.85",
        "month_4 1.0",
        "month_5 0.85",
    ]
    other = tmp_path / "other.csv"
    other.write_text("time,value\n1,2\n")
    with pytest.raises(ValueError, match="not a monthly file"):
        bench.lengthen_monthly_file(other, 5)


def test_a_namelist_is_read_and_written_for_a_plain_ecosim(bench, tmp_path):
    source = tmp_path / "Case_filenames.nml"
    source.write_text(
        "! a comment\n&filenames\n"
        'HDF5_fname            = "testcases/Case/Case.h5"\n'
        'Forcing_fname = "testcases/Case/Case_F.csv"   ! the fishing\n'
        'SpatialGrid_fname     = "testcases/Case/Case_grid.csv"\n'
        "tf              = 2\nStepsPerMonth   = 1\nrelax = 0.5\n/\n"
    )
    pairs = bench.read_namelist(source)
    assert (
        pairs["tf"] == "2" and pairs["Forcing_fname"] == '"testcases/Case/Case_F.csv"'
    )
    bench.write_ecosim_namelist(tmp_path, pairs, 25)
    written = (tmp_path / "filenames.nml").read_text()
    assert "tf = 25" in written and 'HDF5_fname = "testcases/Case/Case.h5"' in written
    # a spatial file has no place in the plain build's namelist
    assert "SpatialGrid" not in written
    assert written.startswith("&filenames") and written.rstrip().endswith("/")


def test_the_same_case_check_says_which_it_is(bench):
    names = ["Outside", "Cod", "Sprat", "Detritus"]
    ours = np.array([1.0, 2.0, 4.0, 10.0])
    near = bench.agreement(names, ours, {"Cod": 2.0002, "Sprat": 4.0, "Seal": 3.0})
    assert near["groups"] == 2 and near["worst"] == "Cod"
    assert near["max"] == pytest.approx(1e-4, rel=1e-2)
    assert "the two ran the same case" in bench.same_case(near)
    far = bench.agreement(names, ours, {"Cod": 1.0, "Sprat": 8.0})
    assert far["median"] == pytest.approx(0.75)
    assert "did NOT run the same case" in bench.same_case(far)
    assert "no group could be matched" in bench.same_case(
        bench.agreement(names, ours, {"Seal": 3.0})
    )


def test_ewe_times_are_read_with_their_warm_up_apart(bench, tmp_path):
    path = tmp_path / "timings.csv"
    rows = ["model,what,years,repeat,seconds,steps"]
    rows += [
        f"M,ecopath,0,{r},{s},0" for r, s in enumerate([0.04, 0.003, 0.0031, 0.0029])
    ]
    rows += [f"M,ecosim_warmup,40,{w},{s},480" for w, s in enumerate([0.9, 0.3])]
    for r in (1, 2, 3):
        for years in (10, 20, 40):
            rows.append(
                f"M,ecosim,{years},{r},{0.02 + 0.001 * years + 0.0005 * r},{12 * years}"
            )
    path.write_text("\n".join(rows) + "\n")
    times = bench.read_ewe_csv(path)["M"]
    assert times["ecopath"]["first"] == 0.04
    assert times["ecopath"]["median"] == pytest.approx(0.003)
    fit = times["ecosim"]["fit"]  # through the fastest repeat of each length
    assert fit["slope"] == pytest.approx(0.001) and fit["intercept"] == pytest.approx(
        0.0205
    )
    assert times["ecosim"]["runs"]["40"]["first"] == 0.9  # the first warm-up run
