"""The monthly box file: a SHYFEM box model's output as PyPath uses it.

``boxes_2d.txt`` holds every output record as text (a year of hourly output
of 29 boxes is 54 MB). A run reads its monthly means only, so
``write_box_netcdf`` keeps those, with the minimum, maximum and spread of the
records of each month, in a classic NetCDF file ``read_box_series`` reads
back as the same series.

The files here are written by the tests; no SHYFEM output is used.
"""

import struct
import subprocess
import sys

import numpy as np
import pytest

from pypath.io import shyfem
from pypath.io.shyfem import ShyfemBoxSeries, read_box_series, write_box_netcdf

BOXES = (1, 2, 3)


def _text(tmp_path, hours=24 * 59, sections=True, name="boxes_2d.txt"):
    """An hourly boxes_2d.txt from 2001-01-01, values changing every hour.

    temp swings by hour and box, salt is constant in time, volume is about
    1e10 with a swing of a few thousand (a spread far below its size).
    Section 1 joins boxes 1 and 2 and changes sign; section 2 is a river.
    """
    start = np.datetime64("2001-01-01T00:00:00")
    lines = ["#      idbox file_type     nvers", "#     473226         2         7"]
    for h in range(1, hours + 1):
        stamp = str(start + np.timedelta64(h, "h")).replace("T", "::")
        lines += [
            "# time record",
            f"   {252460800 + 3600 * h:.8f}      {stamp}",
            "#      boxes       nvars",
            f"          {len(BOXES)}           3",
            "# box      temp      salt        volume",
        ]
        for b in reversed(BOXES):  # out of order on purpose
            temp = 4.0 + b + 3.0 * np.sin(h / 17.0 + b)
            volume = b * 1.0e10 + 4000.0 * np.cos(h / 29.0)
            lines.append(f"   {b}  {temp:.5f}  {float(b):.5f}  {volume:.9E}")
        if sections:
            total = 60.0 * np.sin(h / 11.0)
            lines += [
                "#   sections",
                "          2",
                "# section  from    to         total      positive      negative",
                f"        1     1    2  {total:.6E}  {max(total, 0.0):.6E}"
                f"  {max(-total, 0.0):.6E}",
                f"        2    -1    1  {12.0 + h % 5:.6E}  {12.0 + h % 5:.6E}"
                "  0.000000E+00",
            ]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def _by_month(series):
    """The rows of an every-record series that fall in each month."""
    months = (series.times - np.timedelta64(1, "s")).astype("datetime64[M]")
    return [np.flatnonzero(months == m) for m in np.unique(months)]


# --- the statistics of a month ----------------------------------------------


def test_monthly_reading_keeps_the_statistics_of_each_month(tmp_path):
    path = _text(tmp_path)
    hourly = read_box_series(path)
    monthly = read_box_series(path, monthly=True)
    rows = _by_month(hourly)
    assert monthly.counts.tolist() == [len(r) for r in rows] == [31 * 24, 28 * 24]
    for table, low, high, spread in (
        (hourly.values, monthly.values_min, monthly.values_max, monthly.values_std),
        (
            hourly.section_flow,
            monthly.section_flow_min,
            monthly.section_flow_max,
            monthly.section_flow_std,
        ),
    ):
        np.testing.assert_array_equal(low, [table[r].min(axis=0) for r in rows])
        np.testing.assert_array_equal(high, [table[r].max(axis=0) for r in rows])
        np.testing.assert_allclose(
            spread, [table[r].std(axis=0) for r in rows], rtol=1e-7, atol=1e-12
        )
    # the section that changes sign has both signs among its extremes
    assert monthly.section_flow_min[0, 0, 0] < 0 < monthly.section_flow_max[0, 0, 0]


def test_the_spread_of_a_large_and_steady_variable_is_not_lost_in_its_size(tmp_path):
    """Volume is about 1e10 and swings by a few thousand: a spread from a
    plain sum of squares would be noise; a constant gives exactly nothing."""
    path = _text(tmp_path)
    hourly = read_box_series(path)
    monthly = read_box_series(path, monthly=True)
    rows = _by_month(hourly)
    volume = hourly.variable("volume")
    expected = np.array([volume[r].std(axis=0) for r in rows])
    assert 1000.0 < expected.min() < expected.max() < 4000.0
    np.testing.assert_allclose(monthly.variable("volume", "std"), expected, rtol=1e-6)
    assert np.all(monthly.variable("salt", "std") == 0.0)
    np.testing.assert_array_equal(
        monthly.variable("salt", "min"), monthly.variable("salt", "max")
    )


def test_the_means_are_what_they_were(tmp_path):
    """The statistics ride along: the monthly means are still sum over count."""
    path = _text(tmp_path)
    hourly = read_box_series(path)
    monthly = read_box_series(path, monthly=True)
    for r, mean, flow in zip(_by_month(hourly), monthly.values, monthly.section_flow):
        np.testing.assert_array_equal(mean, hourly.values[r].sum(axis=0) / len(r))
        np.testing.assert_array_equal(flow, hourly.section_flow[r].sum(axis=0) / len(r))


def test_a_series_of_every_record_has_no_statistics(tmp_path):
    hourly = read_box_series(_text(tmp_path, hours=48))
    assert hourly.counts is None and hourly.values_min is None
    assert hourly.section_flow_std is None
    np.testing.assert_array_equal(hourly.variable("temp"), hourly.values[:, :, 0])
    np.testing.assert_array_equal(
        hourly.variable("temp", "mean"), hourly.values[:, :, 0]
    )
    with pytest.raises(ValueError, match="holds no monthly statistics"):
        hourly.variable("temp", "max")
    with pytest.raises(ValueError, match="statistic must be one of"):
        hourly.variable("temp", "median")


# --- the file -----------------------------------------------------------------


FIELDS = (
    "times",
    "boxes",
    "values",
    "section_from",
    "section_to",
    "section_flow",
    "counts",
    "values_min",
    "values_max",
    "values_std",
    "section_flow_min",
    "section_flow_max",
    "section_flow_std",
)


def _same_series(a, b):
    assert a.variables == b.variables and a.monthly is b.monthly is True
    for name in FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.dtype == y.dtype and x.shape == y.shape, name
            np.testing.assert_array_equal(x, y, err_msg=name)


def test_a_file_gives_back_the_series_it_was_written_from(tmp_path):
    monthly = read_box_series(_text(tmp_path), monthly=True)
    out = tmp_path / "boxes_monthly.nc"
    assert write_box_netcdf(monthly, out) == out
    # whatever the caller asks, a monthly file is a monthly series
    for kwargs in ({}, {"monthly": True}, {"monthly": False}):
        _same_series(read_box_series(out, **kwargs), monthly)
    assert read_box_series(str(out)).times.dtype == np.dtype("datetime64[s]")


def test_the_file_is_small_and_says_what_it_holds(tmp_path):
    from scipy.io import netcdf_file

    text = _text(tmp_path)
    monthly = read_box_series(text, monthly=True)
    out = write_box_netcdf(monthly, tmp_path / "m.nc", source="boxes_2d.txt")
    assert out.stat().st_size * 100 < text.stat().st_size
    assert out.read_bytes()[:3] == b"CDF"
    with netcdf_file(str(out), "r", mmap=False) as nc:
        assert nc.format == b"pypath-shyfem-boxes" and nc.format_version == 1
        assert nc.box_variables == b"temp salt volume"
        assert nc.statistics == b"mean min max std"
        assert nc.source == b"boxes_2d.txt"
        assert b"population standard deviation" in nc.comment
        assert set(nc.dimensions) == {"time", "box", "section"}
        assert nc.variables["temp"].units == b"celsius"
        assert nc.variables["volume_std"].units == b"m3"
        assert nc.variables["flow_positive_max"].units == b"m3/s"
        assert nc.variables["time"].units == b"days since 1970-01-01 00:00:00"
        assert nc.variables["records"][:].tolist() == [744, 672]
        assert nc.variables["temp"].shape == (2, 3)


def test_the_source_is_recorded_as_a_name_never_as_a_path(tmp_path):
    from scipy.io import netcdf_file

    monthly = read_box_series(_text(tmp_path), monthly=True)
    out = write_box_netcdf(
        monthly, tmp_path / "m.nc", source=str(tmp_path / "deep" / "boxes_2d.txt")
    )
    with netcdf_file(str(out), "r", mmap=False) as nc:
        assert nc.source == b"boxes_2d.txt"
    assert str(tmp_path).encode() not in out.read_bytes()


def test_the_standard_netcdf_library_opens_it(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    monthly = read_box_series(_text(tmp_path), monthly=True)
    out = write_box_netcdf(monthly, tmp_path / "m.nc")
    with netCDF4.Dataset(str(out)) as ds:
        assert ds.data_model.startswith("NETCDF3")
        np.testing.assert_array_equal(
            np.asarray(ds["temp_max"][:]), monthly.variable("temp", "max")
        )
        times = netCDF4.num2date(ds["time"][:], ds["time"].units)
        assert [f"{t.year}-{t.month:02d}-{t.day:02d}" for t in times] == [
            "2001-01-01",
            "2001-02-01",
        ]


def test_a_run_reads_the_same_from_the_file_as_from_the_text(tmp_path):
    """The fields on a grid and the water flows: a monthly file is placed by
    its months, as monthly means made while reading the text are."""
    from pypath.spatial import EcospaceGrid
    from pypath.spatial.transport import WaterTransport
    from pypath.spatial.zone_fields import shyfem_box_layers

    text = _text(tmp_path)
    file = read_box_series(
        write_box_netcdf(read_box_series(text, monthly=True), tmp_path / "m.nc")
    )
    grid = EcospaceGrid.from_regular_grid(bounds=(0, 0, 3, 1), nx=3, ny=1)
    zones = np.array(BOXES)
    for other in (read_box_series(text), read_box_series(text, monthly=True)):
        a = shyfem_box_layers(file, grid, zones_of_patches=zones)
        b = shyfem_box_layers(other, grid, zones_of_patches=zones)
        for name in ("temp", "salt", "volume"):
            np.testing.assert_allclose(
                a.layers[name].values, b.layers[name].values, rtol=1e-12
            )
            np.testing.assert_array_equal(a.layers[name].times, b.layers[name].times)
        ta = WaterTransport.from_shyfem_box_series(file, groups=[0])
        tb = WaterTransport.from_shyfem_box_series(other, groups=[0])
        np.testing.assert_allclose(ta.flow, tb.flow, rtol=1e-12)
        np.testing.assert_allclose(ta.volume, tb.volume, rtol=1e-12)


def test_a_series_without_statistics_or_sections_makes_a_file_without_them(tmp_path):
    from scipy.io import netcdf_file

    means = ShyfemBoxSeries(
        times=np.array(["2001-01-01", "2001-02-01"], dtype="datetime64[s]"),
        boxes=np.array([4, 9]),
        variables=["temp"],
        values=np.array([[[1.5], [2.5]], [[3.5], [4.5]]]),
        section_from=np.empty(0, dtype=int),
        section_to=np.empty(0, dtype=int),
        section_flow=np.empty((2, 0, 3)),
        monthly=True,
    )
    out = write_box_netcdf(means, tmp_path / "means.nc")
    with netcdf_file(str(out), "r", mmap=False) as nc:
        assert nc.statistics == b"mean"
        assert set(nc.dimensions) == {"time", "box"}  # no section of length 0
        assert set(nc.variables) == {"time", "box", "temp"}
    back = read_box_series(out)
    _same_series(back, means)
    assert back.section_flow.shape == (2, 0, 3) and back.counts is None


def test_only_the_first_months_on_request(tmp_path):
    monthly = read_box_series(_text(tmp_path), monthly=True)
    out = write_box_netcdf(monthly, tmp_path / "m.nc")
    first = read_box_series(out, max_records=1)
    assert len(first.times) == 1 and first.values.shape == (1, 3, 3)
    assert first.section_flow_max.shape == (1, 2, 3) and first.counts.tolist() == [744]


# --- what the writer refuses ----------------------------------------------------


def _monthly(variables=("temp",), boxes=(1, 2), **changes):
    n = len(variables)
    fields = dict(
        times=np.array(["2001-01-01"], dtype="datetime64[s]"),
        boxes=np.array(boxes),
        variables=list(variables),
        values=np.ones((1, len(boxes), n)),
        section_from=np.empty(0, dtype=int),
        section_to=np.empty(0, dtype=int),
        section_flow=np.empty((1, 0, 3)),
        monthly=True,
    )
    fields.update(changes)
    return ShyfemBoxSeries(**fields)


def test_the_writer_takes_monthly_series_only(tmp_path):
    hourly = read_box_series(_text(tmp_path, hours=30))
    with pytest.raises(ValueError, match="monthly=True"):
        write_box_netcdf(hourly, tmp_path / "h.nc")
    assert not (tmp_path / "h.nc").exists()


@pytest.mark.parametrize(
    "variables, message",
    [
        (("time",), "'time' is a name the file uses itself"),
        (("temp", "temp_min"), "'temp_min' would name two tables"),
        (("temp_min", "temp"), "'temp_min' would name two tables"),
        (("a_std_max", "a_std"), "'a_std_max' would name two tables"),
        (("flow_total",), "'flow_total' is a name the file uses itself"),
        (("water temp",), "cannot be a variable's name"),
        (("9lives",), "cannot be a variable's name"),
        (("temp", "temp"), "'temp' is named twice"),
    ],
)
def test_variable_names_that_cannot_be_kept_apart_are_refused(
    tmp_path, variables, message
):
    with pytest.raises(ValueError, match=message):
        write_box_netcdf(_monthly(variables), tmp_path / "m.nc")
    assert not (tmp_path / "m.nc").exists()


def test_numbers_beyond_the_files_integers_are_refused(tmp_path):
    with pytest.raises(ValueError, match="box numbers"):
        write_box_netcdf(_monthly(boxes=(1, 2**31)), tmp_path / "m.nc")
    with pytest.raises(ValueError, match="no month"):
        write_box_netcdf(
            _monthly(
                times=np.empty(0, dtype="datetime64[s]"), values=np.ones((0, 2, 1))
            ),
            tmp_path / "m.nc",
        )


# --- what the reader refuses ----------------------------------------------------


def _file_bytes(tmp_path):
    monthly = read_box_series(_text(tmp_path, hours=24 * 40), monthly=True)
    return write_box_netcdf(monthly, tmp_path / "good.nc").read_bytes()


def test_a_netcdf4_file_is_named_as_such(tmp_path):
    path = tmp_path / "four.nc"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(200))
    with pytest.raises(ValueError, match="NetCDF-4 .HDF5. file"):
        read_box_series(path)


def test_a_file_cut_short_is_refused(tmp_path):
    data = _file_bytes(tmp_path)
    for cut in (5, 40, len(data) // 2, len(data) - 8):
        path = tmp_path / f"cut{cut}.nc"
        path.write_bytes(data[:cut])
        with pytest.raises(ValueError, match="not a monthly box file"):
            read_box_series(path)


def test_text_that_only_starts_like_netcdf_is_refused(tmp_path):
    path = tmp_path / "liar.nc"
    path.write_bytes(b"CDF\x01 this is not a header at all\n" * 50)
    with pytest.raises(ValueError, match="not a monthly box file"):
        read_box_series(path)


def test_a_header_that_announces_more_than_the_file_holds_is_refused(tmp_path):
    """A dimension of two thousand million boxes in a file of a few kB: the
    reader refuses it from the header, before any table is made."""
    import tracemalloc

    data = bytearray(_file_bytes(tmp_path))
    at = data.index(b"box\x00") + 4  # the length of the dimension "box"
    assert struct.unpack(">i", data[at : at + 4])[0] == len(BOXES)
    data[at : at + 4] = struct.pack(">i", 2**31 - 1)
    path = tmp_path / "huge.nc"
    path.write_bytes(bytes(data))
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="not a monthly box file"):
            read_box_series(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 50e6


def test_another_programs_netcdf_is_refused(tmp_path):
    from scipy.io import netcdf_file

    path = tmp_path / "other.nc"
    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("x", 3)
        nc.createVariable("depth", "d", ("x",))[:] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="not a monthly box file"):
        read_box_series(path)


def test_a_file_of_a_later_format_is_refused(tmp_path):
    data = _file_bytes(tmp_path)
    at = data.index(b"format_version")
    at = data.index(struct.pack(">i", 1), at + len(b"format_version") + 8)
    newer = data[:at] + struct.pack(">i", 2) + data[at + 4 :]
    path = tmp_path / "newer.nc"
    path.write_bytes(newer)
    with pytest.raises(ValueError, match="format 2.*newer PyPath"):
        read_box_series(path)


def test_a_file_whose_tables_disagree_is_refused(tmp_path):
    """A statistic of another shape than the means, written by hand."""
    from scipy.io import netcdf_file

    monthly = read_box_series(_text(tmp_path, hours=24 * 40), monthly=True)
    path = write_box_netcdf(monthly, tmp_path / "m.nc")
    bad = tmp_path / "bad.nc"
    with (
        netcdf_file(str(path), "r", mmap=False) as src,
        netcdf_file(str(bad), "w") as dst,
    ):
        dst.format, dst.format_version = "pypath-shyfem-boxes", 1
        dst.box_variables, dst.statistics = "temp", "mean"
        dst.createDimension("time", 2)
        dst.createDimension("box", 3)
        dst.createDimension("other", 2)
        dst.createVariable("time", "i", ("time",))[:] = src.variables["time"][:]
        dst.createVariable("box", "i", ("box",))[:] = [1, 2, 3]
        dst.createVariable("temp", "d", ("time", "other"))[:] = np.ones((2, 2))
    with pytest.raises(ValueError, match="not a monthly box file"):
        read_box_series(bad)


# --- the converter ----------------------------------------------------------------


def test_the_converter_writes_the_file_and_says_what_it_did(tmp_path, capsys):
    text = _text(tmp_path)
    out = tmp_path / "curonian_boxes.nc"
    assert shyfem.main([str(text), str(out)]) == 0
    said = capsys.readouterr().out
    assert "2 months (2001-01 to 2001-02)" in said and "3 boxes, 2 sections" in said
    assert "curonian_boxes.nc" in said
    _same_series(read_box_series(out), read_box_series(text, monthly=True))
    with netcdf_file_source(out) as source:
        assert source == b"boxes_2d.txt"


def netcdf_file_source(path):
    """The ``source`` attribute of a box file, as a context manager."""
    import contextlib

    from scipy.io import netcdf_file

    @contextlib.contextmanager
    def _open():
        with netcdf_file(str(path), "r", mmap=False) as nc:
            yield nc.source

    return _open()


def test_the_converter_never_writes_over_its_input_or_another_file(tmp_path, capsys):
    text = _text(tmp_path, hours=24 * 40)
    before = text.read_bytes()
    assert shyfem.main([str(text), str(text)]) == 2
    assert "is the input" in capsys.readouterr().err
    assert text.read_bytes() == before
    out = tmp_path / "m.nc"
    out.write_text("something of the user's")
    assert shyfem.main([str(text), str(out)]) == 2
    assert "exists" in capsys.readouterr().err
    assert out.read_text() == "something of the user's"
    assert shyfem.main([str(text), str(out), "--overwrite"]) == 0
    assert out.read_bytes()[:3] == b"CDF"
    # a monthly file is not converted again, and an unreadable one is named
    assert shyfem.main([str(out), str(tmp_path / "again.nc")]) == 2
    assert "already a monthly box file" in capsys.readouterr().err
    nothing = tmp_path / "empty.txt"
    nothing.write_text("# nothing here\n")
    assert shyfem.main([str(nothing), str(tmp_path / "none.nc")]) == 2
    assert "no time record" in capsys.readouterr().err
    assert not (tmp_path / "none.nc").exists()


def test_the_converter_runs_as_a_module_without_a_warning(tmp_path):
    text = _text(tmp_path, hours=24 * 35)
    out = tmp_path / "m.nc"
    done = subprocess.run(
        [sys.executable, "-W", "error", "-m", "pypath.io.shyfem", str(text), str(out)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr == "" and "2 months" in done.stdout
    assert read_box_series(out).counts.tolist() == [744, 96]


def test_a_file_object_gets_the_same_bytes_and_stays_open(tmp_path):
    import io

    monthly = read_box_series(_text(tmp_path, hours=24 * 40), monthly=True)
    on_disk = write_box_netcdf(monthly, tmp_path / "m.nc", source="boxes_2d.txt")
    buffer = io.BytesIO()
    assert write_box_netcdf(monthly, buffer, source="boxes_2d.txt") is buffer
    assert not buffer.closed
    assert buffer.getvalue() == on_disk.read_bytes()
    # and the same series twice gives the same file: nothing of the moment in it
    again = write_box_netcdf(monthly, tmp_path / "again.nc", source="boxes_2d.txt")
    assert again.read_bytes() == on_disk.read_bytes()
