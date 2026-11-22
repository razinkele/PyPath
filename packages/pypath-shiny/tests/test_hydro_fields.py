"""A SHYFEM box model's output on the Ecospace page (services.hydro_fields)."""

import numpy as np
import pytest
from pypath.core.ecosim import rsim_scenario
from pypath.spatial import EcospaceParams, rsim_run_spatial

from pypath_shiny.services import hydro_fields
from pypath_shiny.services.hydro_fields import (
    DEFAULT_SUBSTEPS,
    build_transport,
    compare_substeps,
    field_on_patches,
    field_range,
    month_labels,
    read_uploaded_boxes,
    summarise_boxes,
    transport_groups,
    transport_scenario,
)
from pypath_shiny.services.mesh_grids import build_mesh_grid

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def box_file(tmp_path, boxes=(1, 2, 3), months=14, sections=True, name="boxes.txt"):
    """Daily box output from 2001-01: salt = box + month / 100, temp = 10 x box.

    Two sections exchange 40 m3/s each way between neighbouring boxes.
    """
    start = np.datetime64("2001-01-01T00:00:00")
    end = (np.datetime64("2001-01") + np.timedelta64(months, "M")).astype(
        "datetime64[s]"
    )
    days = int((end - start) / np.timedelta64(1, "D"))
    lines = ["#      idbox file_type     nvers", "#     473226         2         7"]
    for d in range(1, days + 1):
        stamp = start + np.timedelta64(d, "D")
        month = (stamp - np.timedelta64(12, "h")).astype("datetime64[M]")
        k = int((month - np.datetime64("2001-01")) / np.timedelta64(1, "M"))
        lines += [
            "# time record",
            f"   {252460800 + 86400 * d:.8f}      {str(stamp).replace('T', '::')}",
            "#      boxes       nvars",
            f"          {len(boxes)}           3",
            "# box      temp      salt        volume",
        ]
        for b in boxes:
            lines.append(f"   {b}  {10.0 * b:.5f}  {b + k / 100:.5f}  {1e8:.6E}")
        if sections:
            lines += [
                "#   sections",
                f"          {len(boxes) - 1}",
                "# section  from    to         total      positive      negative",
            ]
            for s, (a, b) in enumerate(zip(boxes[:-1], boxes[1:]), start=1):
                lines.append(
                    f"        {s}     {a}    {b}  0.000000E+00  0.400000E+02"
                    "  0.400000E+02"
                )
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def zone_grd(tmp_path, nx=6, ny=2, size=0.01, zones=3):
    """nx x ny squares of two triangles near 21 E 55 N; types 1..zones west to east."""
    lines, node = [], {}
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            node[(ix, iy)] = len(node) + 1
            lines.append(f"1 {node[(ix, iy)]} 0 {21 + ix * size} {55 + iy * size} 2.0")
    number = 0
    for iy in range(ny):
        for ix in range(nx):
            kind = 1 + ix * zones // nx
            a, b = node[(ix, iy)], node[(ix + 1, iy)]
            c, d = node[(ix + 1, iy + 1)], node[(ix, iy + 1)]
            for tri in ((a, b, c), (a, c, d)):
                number += 1
                lines.append(f"2 {number} {kind} 3 {tri[0]} {tri[1]} {tri[2]}")
    path = tmp_path / "zones.grd"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def series(tmp_path):
    return read_uploaded_boxes(box_file(tmp_path))


@pytest.fixture
def grid(tmp_path):
    return build_mesh_grid(zone_grd(tmp_path), patches="type")


@pytest.fixture(scope="module")
def scenario(rpath_params, balanced_rpath_model):
    return rsim_scenario(balanced_rpath_model, rpath_params, years=range(1, 3))


# --- the file ---------------------------------------------------------------


def test_the_file_is_read_as_monthly_means(series):
    assert series.monthly and len(series.times) == 14
    assert month_labels(series)[:2] == ["2001-01", "2001-02"]
    assert month_labels(series)[-1] == "2002-02"
    np.testing.assert_allclose(series.variable("salt")[3], [1.03, 2.03, 3.03])


def test_a_file_that_is_too_large_or_has_too_many_boxes_is_refused(
    tmp_path, monkeypatch
):
    path = box_file(tmp_path, months=2)
    monkeypatch.setattr(hydro_fields, "MAX_BOX_BYTES", 1000)
    with pytest.raises(ValueError, match="Cut the file"):
        read_uploaded_boxes(path)
    monkeypatch.undo()
    monkeypatch.setattr(hydro_fields, "MAX_BOXES", 2)
    with pytest.raises(ValueError, match="3 boxes"):
        read_uploaded_boxes(path)
    monkeypatch.undo()
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n")
    with pytest.raises(ValueError):
        read_uploaded_boxes(empty)


def test_the_summary_says_what_the_file_holds_and_how_it_meets_the_grid(
    series, grid, tmp_path
):
    alone = summarise_boxes(series)
    assert (alone.n_months, alone.first_month, alone.last_month) == (
        14,
        "2001-01",
        "2002-02",
    )
    assert alone.variables == ["temp", "salt", "volume"]
    assert (alone.n_boxes, alone.n_sections, alone.has_flows) == (3, 2, True)
    assert alone.missing_zones is None and not alone.fits_grid and not alone.problems
    on_grid = summarise_boxes(series, grid)
    assert on_grid.fits_grid and (on_grid.n_patches, on_grid.n_zones) == (3, 3)
    # a grid with a zone the file does not have
    other = build_mesh_grid(zone_grd(tmp_path, zones=6), patches="type")
    lacking = summarise_boxes(series, other)
    assert lacking.missing_zones == [4, 5, 6] and not lacking.fits_grid
    assert "zones 4, 5, 6" in lacking.problems[0]


def test_a_file_without_sections_shows_fields_but_gives_no_transport(tmp_path):
    series = read_uploaded_boxes(box_file(tmp_path, months=2, sections=False))
    summary = summarise_boxes(series)
    assert not summary.has_flows and "water transport needs both" in summary.problems[0]
    with pytest.raises(ValueError, match="no section flows"):
        build_transport(series, [0])


# --- the fields -------------------------------------------------------------


def test_a_field_by_patch_for_boxes_and_for_elements(series, grid, tmp_path):
    np.testing.assert_allclose(
        field_on_patches(series, grid, "salt", 5), [1.05, 2.05, 3.05]
    )
    cells = build_mesh_grid(zone_grd(tmp_path), patches="element")
    by_cell = field_on_patches(series, cells, "temp", 0)
    assert by_cell.shape == (24,) and set(by_cell) == {10.0, 20.0, 30.0}
    assert field_range(series, "salt") == pytest.approx((1.0, 3.13))
    assert field_range(series, "volume") == (1e8, 1e8 + 1.0)
    # a patch in a zone the file lacks is left blank
    other = build_mesh_grid(zone_grd(tmp_path, zones=6), patches="type")
    blank = field_on_patches(series, other, "salt", 0)
    assert np.isnan(blank[3:]).all() and np.isfinite(blank[:3]).all()


# --- the groups and the transport -------------------------------------------


def test_groups_are_offered_by_what_the_run_can_carry(scenario):
    choices = transport_groups(scenario)
    params = scenario.params
    assert len(choices) == params.NUM_LIVING + params.NUM_DEAD
    assert [c.row for c in choices] == list(range(len(choices)))
    assert choices[0].name == str(params.spname[1]).strip()
    flags = np.asarray(params.NoIntegrate)
    for c in choices:
        expected = bool(getattr(params, "RK4_FAST_EQUILIBRIUM", True)) and bool(
            flags[c.row + 1] > 0
        )
        assert c.fast == expected
        if c.row >= params.NUM_LIVING and c.fast:
            assert not c.eligible and "detritus" in c.reason
        else:
            assert c.eligible


def test_fast_groups_are_integrated_and_the_others_left_alone(scenario):
    choices = transport_groups(scenario)
    slow = [c for c in choices if c.eligible and not c.fast]
    same, integrated = transport_scenario(scenario, choices, [slow[0].row])
    assert same is scenario and integrated == []
    fast = [c for c in choices if c.eligible and c.fast]
    if fast:
        changed, integrated = transport_scenario(scenario, choices, [fast[0].row])
        assert integrated == [fast[0].name] and changed is not scenario
        assert changed.params.RK4_STEPS == DEFAULT_SUBSTEPS
        assert changed.params.NoIntegrate[fast[0].row + 1] == 0
        assert scenario.params.NoIntegrate[fast[0].row + 1] != 0  # not changed
    refused = [c for c in choices if not c.eligible]
    for c in refused:
        with pytest.raises(ValueError, match="cannot carry"):
            transport_scenario(scenario, choices, [c.row])


def test_transport_from_the_file_runs_and_moves_the_group(series, grid, scenario):
    choices = transport_groups(scenario)
    carried = next(c for c in choices if c.eligible and not c.fast)
    transport = build_transport(series, [carried.row])
    assert transport.beyond == "repeat_year" and transport.n_months == 14
    n = scenario.params.NUM_GROUPS

    def run(with_transport, scen=scenario):
        capacity = np.ones((n, 3))
        capacity[carried.row] = [0.4, 1.0, 0.7]
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n, 3)),
            habitat_capacity=capacity,
            dispersal_rate=np.zeros(n),
            advection_enabled=np.zeros(n, dtype=bool),
            gravity_strength=np.zeros(n),
            water_transport=transport if with_transport else None,
        )
        return rsim_run_spatial(scen, ecospace=ecospace, years=range(1, 3))

    still, moved = run(False), run(True)  # 24 months on a 14-month file
    assert not np.allclose(
        moved.out_Biomass_spatial[-1, carried.row + 1],
        still.out_Biomass_spatial[-1, carried.row + 1],
    )
    assert np.isfinite(moved.out_Biomass_spatial).all()
    check = compare_substeps(still, still, scenario, 1)
    assert check.converged and check.max_difference == 0.0 and "passed" in check.message
    assert (check.substeps, check.finer) == (1, 2)


def test_a_short_file_holds_its_nearest_month(tmp_path):
    series = read_uploaded_boxes(box_file(tmp_path, months=3))
    assert build_transport(series, [0]).beyond == "hold"
    assert build_transport(series, [0], beyond="refuse").beyond == "refuse"
    with pytest.raises(ValueError, match="beyond must be"):
        build_transport(series, [0], beyond="loop")


def test_the_substep_comparison_names_the_worst_group_and_fails_on_bad_values(
    scenario,
):
    class Out:
        def __init__(self, biomass):
            self.out_Biomass_spatial = biomass

    n = scenario.params.NUM_GROUPS + 1
    a = np.ones((5, n, 2))
    b = a.copy()
    b[3, 2, 1] = 1.25
    result = compare_substeps(Out(a), Out(b), scenario, 4)
    assert not result.converged and result.max_difference == pytest.approx(0.2)
    assert (result.month, result.group) == (3, str(scenario.params.spname[2]).strip())
    assert "20.0%" in result.message and "8 sub-steps" in result.message
    b[3, 2, 1] = np.nan
    failed = compare_substeps(Out(a), Out(b), scenario, 4)
    assert failed.max_difference == np.inf and "failed" in failed.message
    # only the groups are compared, not the Outside row
    c = a.copy()
    c[:, 0] = 99.0
    assert compare_substeps(Out(a), Out(c), scenario, 4).converged
