"""SHYFEM box-model output as fields on Ecospace patches, and capacity from them.

The files here are written by the tests; no SHYFEM output is used.
"""

import numpy as np
import pytest

from pypath.io.shyfem import read_box_series
from pypath.spatial import (
    EcospaceGrid,
    EcospaceParams,
    capacity_from_layers,
    create_linear_response,
    rsim_run_spatial,
    zone_series_layer,
)
from pypath.spatial.zone_fields import shyfem_box_layers

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detritus fate of a detritus group to itself:UserWarning"
)


def _box_file(tmp_path, boxes=(1, 2, 3), hours=24 * 59, sections=True, name="b.txt"):
    """An hourly boxes_2d.txt for January and February 2001.

    temp = 10 x box in January and 20 x box in February; salt = box; volume
    = 1e6 x box. Records are stamped at the end of their hour, as SHYFEM's.
    """
    start = np.datetime64("2001-01-01T00:00:00")
    lines = ["#      idbox file_type     nvers", "#     473226         2         7"]
    for h in range(1, hours + 1):
        stamp = start + np.timedelta64(h, "h")
        month = (stamp - np.timedelta64(30, "m")).astype("datetime64[M]")
        factor = 10.0 if str(month) == "2001-01" else 20.0
        text = str(stamp).replace("T", "::")
        lines += [
            "# time record",
            f"   {252460800 + 3600 * h:.8f}      {text}",
            "#      boxes       nvars",
            f"          {len(boxes)}           3",
            "# box      temp      salt        volume",
        ]
        # boxes are written out of order on purpose
        for b in reversed(boxes):
            lines.append(f"   {b}  {factor * b:.5f}  {float(b):.5f}  {b * 1e6:.6E}")
        if sections:
            lines += [
                "#   sections",
                "          2",
                "# section  from    to         total      positive      negative",
                f"        1     {boxes[0]}    {boxes[1]}  0.500000E+02  0.700000E+02  0.200000E+02",
                f"        2    -1     {boxes[0]} -0.100000E+02  0.000000E+00  0.100000E+02",
            ]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def _zone_grd(tmp_path, zone=None, nx=6, ny=2, size=1000.0):
    """nx x ny squares of two triangles; types 1, 2, 3 west to east by default."""
    zone = zone or (lambda ix, iy: 1 + ix // 2)
    lines, node = [], {}
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            node[(ix, iy)] = len(node) + 1
            lines.append(f"1 {node[(ix, iy)]} 0 {ix * size} {iy * size} 2.0")
    number = 0
    for iy in range(ny):
        for ix in range(nx):
            a, b = node[(ix, iy)], node[(ix + 1, iy)]
            c, d = node[(ix + 1, iy + 1)], node[(ix, iy + 1)]
            for tri in ((a, b, c), (a, c, d)):
                number += 1
                lines.append(f"2 {number} {zone(ix, iy)} 3 {tri[0]} {tri[1]} {tri[2]}")
    path = tmp_path / "zones.grd"
    path.write_text("\n".join(lines) + "\n")
    return path


# --- the reader -----------------------------------------------------------


def test_read_box_series_values_times_and_sections(tmp_path):
    series = read_box_series(_box_file(tmp_path, hours=48))
    assert series.boxes.tolist() == [1, 2, 3]  # sorted, whatever the file's order
    assert series.variables == ["temp", "salt", "volume"]
    assert series.values.shape == (48, 3, 3)
    assert series.times[0] == np.datetime64("2001-01-01T01:00:00")
    assert series.times[-1] == np.datetime64("2001-01-03T00:00:00")
    np.testing.assert_allclose(series.variable("temp")[0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(series.variable("volume")[5], [1e6, 2e6, 3e6])
    assert series.section_from.tolist() == [1, -1]
    assert series.section_to.tolist() == [2, 1]
    assert series.section_flow.shape == (48, 2, 3)
    np.testing.assert_allclose(series.section_flow[0, 0], [50.0, 70.0, 20.0])
    with pytest.raises(KeyError, match="oxygen"):
        series.variable("oxygen")


def test_read_box_series_without_sections_and_with_a_record_limit(tmp_path):
    series = read_box_series(
        _box_file(tmp_path, hours=30, sections=False), max_records=10
    )
    assert series.values.shape == (10, 3, 3)
    assert series.section_flow.shape == (10, 0, 3)


def test_read_box_series_refuses_absurd_counts_and_empty_files(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text(
        "# time record\n 1.0 2001-01-01::01:00:00\n# boxes nvars\n 90000000 3\n"
    )
    with pytest.raises(ValueError, match="90000000 boxes"):
        read_box_series(path)
    path.write_text("# nothing here\n")
    with pytest.raises(ValueError, match="no time record"):
        read_box_series(path)


def test_read_box_series_refuses_a_record_whose_boxes_change(tmp_path):
    text = _box_file(tmp_path, hours=2, sections=False).read_text().splitlines()
    text[-1] = text[-1].replace("   1 ", "   7 ", 1)
    path = tmp_path / "changed.txt"
    path.write_text("\n".join(text) + "\n")
    with pytest.raises(ValueError, match="boxes differ"):
        read_box_series(path)


# --- series onto patches --------------------------------------------------


def test_monthly_means_by_zone_id_not_by_position(tmp_path):
    series = read_box_series(_box_file(tmp_path))
    grid = EcospaceGrid.from_shyfem_grd(_zone_grd(tmp_path), patches="type")
    layer = zone_series_layer(
        series.times,
        series.variable("temp"),
        series.boxes,
        grid,
        name="temp",
        units="celsius",
    )
    assert layer.values.shape == (2, 3)
    np.testing.assert_allclose(layer.values[0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(layer.values[1], [20.0, 40.0, 60.0])
    np.testing.assert_allclose(layer.times, [0.5 / 12, 1.5 / 12])
    # held for the month: any time in February gives February's mean
    np.testing.assert_allclose(layer.get_value_at_time(1.01 / 12), [20, 40, 60])
    np.testing.assert_allclose(layer.get_value_at_time(1.99 / 12), [20, 40, 60])
    np.testing.assert_allclose(layer.get_value_at_time(0.0), [10, 20, 30])

    # the series in another zone order gives the same layer
    order = np.array([2, 0, 1])
    shuffled = zone_series_layer(
        series.times,
        series.variable("temp")[:, order],
        series.boxes[order],
        grid,
        name="temp",
        units="celsius",
    )
    np.testing.assert_allclose(shuffled.values, layer.values)


def test_a_start_date_moves_the_time_axis_and_months_before_it_are_dropped(tmp_path):
    series = read_box_series(_box_file(tmp_path))
    grid = EcospaceGrid.from_shyfem_grd(_zone_grd(tmp_path), patches="type")
    kwargs = dict(name="temp", units="celsius")
    early = zone_series_layer(
        series.times, series.variable("temp"), series.boxes, grid,
        start="2000-07-01", **kwargs,
    )  # fmt: skip
    np.testing.assert_allclose(early.times, [6.5 / 12, 7.5 / 12])
    late = zone_series_layer(
        series.times, series.variable("temp"), series.boxes, grid,
        start="2001-02-01", **kwargs,
    )  # fmt: skip
    np.testing.assert_allclose(late.times, [0.5 / 12])
    np.testing.assert_allclose(late.values[0], [20.0, 40.0, 60.0])
    with pytest.raises(ValueError, match="no data on or after"):
        zone_series_layer(
            series.times, series.variable("temp"), series.boxes, grid,
            start="2005-01-01", **kwargs,
        )  # fmt: skip


def test_elements_take_the_value_of_their_box(tmp_path):
    series = read_box_series(_box_file(tmp_path))
    grd = _zone_grd(tmp_path)
    elements = EcospaceGrid.from_shyfem_grd(grd, patches="element")
    assert (
        elements.cell_metadata["type"].tolist()
        == [1] * 4 + [2] * 4 + [3] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    )
    layers = shyfem_box_layers(series, elements, variables=["salt"])
    salt = layers.get_layer_at_time("salt", 0.0)
    np.testing.assert_allclose(salt, elements.cell_metadata["type"].to_numpy())


def test_a_patch_whose_zone_is_not_in_the_series_is_named(tmp_path):
    series = read_box_series(_box_file(tmp_path, boxes=(1, 2)))
    grid = EcospaceGrid.from_shyfem_grd(_zone_grd(tmp_path), patches="type")
    with pytest.raises(ValueError, match=r"zones? 3"):
        shyfem_box_layers(series, grid)


def test_shyfem_box_layers_names_units_and_skips_volume_by_default(tmp_path):
    series = read_box_series(_box_file(tmp_path))
    grid = EcospaceGrid.from_shyfem_grd(_zone_grd(tmp_path), patches="type")
    drivers = shyfem_box_layers(series, grid)
    assert sorted(drivers.layer_names) == ["salt", "temp", "volume"]
    assert drivers.layers["temp"].units == "celsius"
    assert drivers.layers["salt"].units == "psu"
    assert drivers.n_patches == 3


# --- capacity from layers -------------------------------------------------


def _drivers(tmp_path, grid):
    return shyfem_box_layers(read_box_series(_box_file(tmp_path)), grid)


def test_capacity_is_one_for_groups_without_a_response(tmp_path):
    grid = EcospaceGrid.from_shyfem_grd(_zone_grd(tmp_path), patches="type")
    capacity = capacity_from_layers(_drivers(tmp_path, grid), {}, n_groups=5)
    assert capacity.shape == (5, 3)
    assert np.all(capacity == 1.0)


def test_capacity_is_the_time_mean_of_the_response_by_group_name(tmp_path):
    grid = EcospaceGrid.from_shyfem_grd(_zone_grd(tmp_path), patches="type")
    drivers = _drivers(tmp_path, grid)
    # fresh-water group: suitable at salt 1, unsuitable from salt 3
    fresh = create_linear_response(1.0, 3.0, increasing=False)
    warm = create_linear_response(0.0, 60.0)  # rises with temperature
    names = ["Phyto", "Zoo", "Fish", "Det", "Fleet"]
    capacity = capacity_from_layers(
        drivers,
        {"Fish": {"salt": fresh}, 1: {"temp": warm, "salt": fresh}},
        group_names=names,
    )
    np.testing.assert_allclose(capacity[2], [1.0, 0.5, 0.0])
    # Zoo: mean over January and February of temp response x salt response
    january = np.array([10, 20, 30]) / 60.0
    february = np.array([20, 40, 60]) / 60.0
    np.testing.assert_allclose(
        capacity[1], 0.5 * (january + february) * [1.0, 0.5, 0.0]
    )
    assert np.all(capacity[[0, 3, 4]] == 1.0)
    at_january = capacity_from_layers(
        drivers, {"Zoo": {"temp": warm}}, group_names=names, when=0.5 / 12
    )
    np.testing.assert_allclose(at_january[1], january)
    with pytest.raises(KeyError, match="Cod"):
        capacity_from_layers(drivers, {"Cod": {"salt": fresh}}, group_names=names)
    with pytest.raises(KeyError, match="oxygen"):
        capacity_from_layers(drivers, {"Fish": {"oxygen": fresh}}, group_names=names)


def test_boxes_and_their_elements_run_alike_without_movement(
    tmp_path, spatial_scenario
):
    """A box field broadcast to elements: the element run, summed to boxes, is the box run."""
    scenario, _ = spatial_scenario
    grd = _zone_grd(
        tmp_path, zone=lambda ix, iy: 1 if ix == 0 else (2 if ix < 3 else 3)
    )
    series = read_box_series(_box_file(tmp_path))
    n_rows = scenario.params.NUM_GROUPS + 1
    fresh = create_linear_response(0.0, 6.0, increasing=False)

    def run(patches):
        grid = EcospaceGrid.from_shyfem_grd(grd, patches=patches)
        capacity = capacity_from_layers(
            shyfem_box_layers(series, grid), {2: {"salt": fresh}}, n_groups=n_rows
        )
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_rows, grid.n_patches)),
            habitat_capacity=capacity,
            dispersal_rate=np.zeros(n_rows),
            advection_enabled=np.zeros(n_rows, dtype=bool),
            gravity_strength=np.zeros(n_rows),
        )
        out = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))
        return grid, out.out_Biomass_spatial

    boxes, by_box = run("type")
    assert boxes.patch_areas.tolist() == pytest.approx([2.0, 4.0, 6.0])  # unequal
    elements, by_element = run("element")
    box_of = np.searchsorted(boxes.patch_ids, elements.cell_metadata["type"])
    summed = np.zeros_like(by_box)
    for patch in range(elements.n_patches):
        summed[:, :, box_of[patch]] += by_element[:, :, patch]
    np.testing.assert_allclose(summed, by_box, rtol=1e-9, atol=1e-12)
    assert not np.allclose(by_box[-1, 3, 0] / 2.0, by_box[-1, 3, 2] / 6.0)  # it varies
