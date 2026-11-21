"""Reading the link flows of an AQUABC input folder (pypath.io.aquabc).

The folder is written by the tests; no AQUABC input set is used.
"""

import numpy as np
import pytest

from pypath.io.aquabc import read_aquabc_flows, read_aquabc_series
from pypath.spatial import WaterTransport


def _series_file(path, title, days, values, scale=None, unit=None):
    nv = values.shape[1]
    scale = np.ones(nv) if scale is None else scale
    unit = np.ones(nv) if unit is None else unit
    lines = [
        f"# {title}",
        "# DATA_SIZE",
        str(len(days)),
        "# NUMBER_OF_VARIABLES",
        str(nv),
    ]
    lines += ["# SCALE FACTORS", "#", " ".join(f"{v:.8f}" for v in scale)]
    lines += ["# UNIT CONVERSION FACTORS", "#", " ".join(f"{v:.8f}" for v in unit)]
    lines += ["# INTERPOLATE (1=yes)", "1", "# TIME AND VALUES"]
    for d, row in zip(days, values):
        lines.append(f"{d:.6f} " + " ".join(f"{v:.6f}" for v in row))
    path.write_text("\n".join(lines) + "\n")


def _folder(tmp_path, days=60, columns=None, n_series=1):
    """Boxes 1 (2 m deep) and 2 (5 m): 1 <-> 2, a river (-2) into 1, the sea at 2.

    The link table names the flow columns out of order on purpose.
    """
    links = [(1, 2, 3), (2, 1, 1), (-2, 1, 5), (-1, 2, 2), (2, -1, 4)]
    if columns is not None:
        links = [(u, d, c) for (u, d, _), c in zip(links, columns)]
    text = [
        "# DESCRIPTION a set written by the tests",
        "# NUM_PELAGIC_BOXES",
        "                  2",
        "# NUM_PELAGIC_ADVECTIVE_LINKS",
        f"                  {len(links)}",
        "# NUM_FLOW_TS",
        f"                  {n_series}",
        "# ********************* INITIAL CONDITIONS *********************",
        "#   PELAGIC BOX NO   INIT COND SET NO   SURFACE ELEV   BOTTOM ELEV",
        "                   1                   1              0.0000             -2.0000",
        "                   2                   1              0.0000             -5.0000",
        "# ********************* ADVECTIVE LINKS *********************",
        "#  ADVECTIVE LINK NO        UPSTREAM BOX      DOWNSTREAM BOX             FLOW TS      FLOW TS VAR NO",
    ]
    for k, (u, d, c) in enumerate(links, 1):
        text.append(f"   {k}   {u}   {d}   1   {c}")
    (tmp_path / "PELAGIC_INPUTS.txt").write_text("\n".join(text) + "\n")
    # column values: 30, 5, 40, 15, 10 (so 1->2 is 40, 2->1 is 30, river 10, sea 5 / 15)
    values = np.tile([30.0, 5.0, 40.0, 15.0, 10.0], (days, 1))
    _series_file(tmp_path / "FLOW_TS.txt", "FLOWS m3/s", np.arange(days), values)
    for box, area in ((1, 5e7), (2, 6e7)):
        rows = [f"BATHYMETRY {box}", "NUM_LAYES", "        16"]
        rows.append(
            "  LAYER_NO  UPPER_ELEVATION  LOWER_ELEVATION  UPPER_AREA  LOWER_AREA  UPPER_LENGTH"
        )
        for k in range(16):  # layers of 0.5 m from -6 m to +2 m, one area
            rows.append(
                f"  {k + 1}  {-5.5 + 0.5 * k:.4f}  {-6.0 + 0.5 * k:.4f}  {area:.2f}  {area:.2f}  8000.00"
            )
        (tmp_path / f"BATHYMETRY_{box}.txt").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_read_a_series_file_with_scale_and_unit_factors(tmp_path):
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _series_file(
        tmp_path / "X_TS.txt", "X", [0.0, 1.0, 2.0], values,
        scale=np.array([2.0, 1.0]), unit=np.array([1.0, 0.5]),
    )  # fmt: skip
    days, out, interpolate = read_aquabc_series(tmp_path / "X_TS.txt")
    assert days.tolist() == [0.0, 1.0, 2.0] and interpolate is True
    np.testing.assert_allclose(out, values * [2.0, 0.5])


def test_a_series_file_that_is_cut_short_or_absurd_is_refused(tmp_path):
    path = tmp_path / "X_TS.txt"
    _series_file(path, "X", [0.0, 1.0], np.ones((2, 3)))
    text = path.read_text().splitlines()
    path.write_text("\n".join(text[:-1]) + "\n")
    with pytest.raises(ValueError, match="announced"):
        read_aquabc_series(path)
    path.write_text("# X\n# DATA_SIZE\n900000000\n# NUMBER_OF_VARIABLES\n900\n")
    with pytest.raises(ValueError, match="cannot be right"):
        read_aquabc_series(path)


def test_links_take_their_own_column_of_the_flow_series(tmp_path):
    flows = read_aquabc_flows(_folder(tmp_path), start="2012-01-01")
    assert flows.upstream.tolist() == [1, 2, -2, -1, 2]
    assert flows.downstream.tolist() == [2, 1, 1, 2, -1]
    np.testing.assert_allclose(flows.flows[0], [40.0, 30.0, 10.0, 5.0, 15.0])
    assert flows.times[0] == np.datetime64("2012-01-01T00:00:00")
    assert flows.times[-1] == np.datetime64("2012-02-29T00:00:00")
    assert flows.boxes.tolist() == [1, 2]
    np.testing.assert_allclose(flows.depth, [2.0, 5.0])
    np.testing.assert_allclose(flows.area, [5e7, 6e7])
    np.testing.assert_allclose(flows.volume, [1e8, 3e8])


def test_the_folder_gives_a_transport_whose_boxes_balance(tmp_path):
    flows = read_aquabc_flows(_folder(tmp_path), start="2012-01-01")
    transport = WaterTransport.from_link_flows(
        flows.times, flows.flows, flows.upstream, flows.downstream, flows.volume,
        groups=[0], zones=flows.boxes, throughflow=True,
    )  # fmt: skip
    assert transport.n_months == 2 and transport.open_boundaries == [-2]
    np.testing.assert_allclose(transport.flow[0], [[0.0, 40.0], [30.0, 0.0]])
    # each box: in from the other and the outside = out to them
    inflow = transport.flow[0].sum(axis=0) + [10.0, 5.0]
    outflow = transport.flow[0].sum(axis=1) + [0.0, 15.0]
    np.testing.assert_allclose(inflow, outflow)


def test_bad_link_tables_are_refused(tmp_path):
    with pytest.raises(ValueError, match="column 9"):
        read_aquabc_flows(
            _folder(tmp_path, columns=[3, 1, 5, 2, 9]), start="2012-01-01"
        )
    with pytest.raises(ValueError, match="one flow series"):
        read_aquabc_flows(_folder(tmp_path, n_series=2), start="2012-01-01")
    folder = _folder(tmp_path)
    text = (
        (folder / "PELAGIC_INPUTS.txt")
        .read_text()
        .replace("   1   1   2   1   3", "   1   7   2   1   3")
    )
    (folder / "PELAGIC_INPUTS.txt").write_text(text)
    with pytest.raises(ValueError, match=r"boxes \[7\]"):
        read_aquabc_flows(folder, start="2012-01-01")


def test_a_bathymetry_whose_area_changes_with_depth(tmp_path):
    folder = _folder(tmp_path)
    rows = [
        "BATHYMETRY 1",
        "NUM_LAYES",
        "  2",
        "  LAYER_NO UPPER LOWER UPPER_AREA LOWER_AREA LEN",
    ]
    rows.append("  1  -1.0000  -2.0000  4.0e7  2.0e7  8000.00")  # widens upward
    rows.append("  2   0.5000  -1.0000  6.0e7  4.0e7  8000.00")
    (folder / "BATHYMETRY_1.txt").write_text("\n".join(rows) + "\n")
    flows = read_aquabc_flows(folder, start="2012-01-01")
    # layer 1 whole: 1 m x 3e7; layer 2 up to the surface at 0: 1 m x mean(4e7, 5.333e7)
    assert flows.volume[0] == pytest.approx(
        3.0e7 + 0.5 * (4.0e7 + 4.0e7 + 2.0e7 * 1.0 / 1.5)
    )
    assert flows.area[0] == pytest.approx(4.0e7 + 2.0e7 * 1.0 / 1.5)
