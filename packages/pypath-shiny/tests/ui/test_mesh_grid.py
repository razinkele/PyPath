"""Headless drive of the Ecospace page's SHYFEM mesh grid type.

A ``.grd`` written by the test is uploaded; the page shows what it read,
makes a grid at each resolution, refuses a mesh in metres without an EPSG
code, and a short spatial run on the mesh grid finishes.

CI ignores ``tests/ui``. Run by hand (about a minute):

    pytest packages/pypath-shiny/tests/ui/test_mesh_grid.py -v

Needs Playwright with Chromium and Linux, as ``test_ecosim_async_runs.py``,
whose server and page helpers this file uses.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

from tests.ui.test_ecosim_async_runs import (  # noqa: E402
    _click_for_note,
    _create_scenario,
    _new_page,
    _notes,
    _setup,
    _text,
    _wait_text,
    browser,  # noqa: F401  (fixture)
    pytestmark,  # noqa: F401  (slow; needs /proc)
    server,  # noqa: F401  (fixture)
)


def _grd(path, size, x0, y0, nx=6, ny=3):
    """nx x ny squares of two triangles; three element types, west to east."""
    lines, node = [], {}
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            node[(ix, iy)] = len(node) + 1
            lines.append(
                f"1 {node[(ix, iy)]} 0 {x0 + ix * size} {y0 + iy * size} {2.0 + ix}"
            )
    number = 0
    for iy in range(ny):
        for ix in range(nx):
            a, b = node[(ix, iy)], node[(ix + 1, iy)]
            c, d = node[(ix + 1, iy + 1)], node[(ix, iy + 1)]
            for tri in ((a, b, c), (a, c, d)):
                number += 1
                lines.append(f"2 {number} {1 + ix // 2} 3 {tri[0]} {tri[1]} {tri[2]}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _cells(page) -> int:
    """Patches the grid info line reports."""
    text = _text(page, "#grid_info")
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    return int(digits[0]) if digits else -1


def test_mesh_upload_makes_a_grid_at_both_resolutions(server, browser, tmp_path):  # noqa: F811
    errors: list[str] = []
    page = _new_page(browser, errors)
    _setup(page, server.url)
    _create_scenario(page)
    page.click("a.dropdown-toggle:has-text('Advanced')")
    page.click("a[data-value='Ecospace']")
    page.wait_for_selector("#create_grid", state="visible", timeout=30000)

    page.select_option("#grid_type", "shyfem")
    page.wait_for_selector("#mesh_file_upload", state="attached", timeout=10000)
    # no file yet: a warning, and no grid
    assert _click_for_note(page, "#create_grid", "Upload a SHYFEM grid", 15), _notes(
        page
    )

    degrees = _grd(tmp_path / "lagoon.grd", 0.05, 21.0, 55.2)
    page.set_input_files("#mesh_file_upload", degrees)
    assert _wait_text(page, "#mesh_summary_text", "36 elements", 20), _text(
        page, "#mesh_summary"
    )
    summary = _text(page, "#mesh_summary_text")
    assert "28 nodes" in summary and "3 element types" in summary
    assert "degrees" in summary and "(detected)" in summary
    assert "depth" in summary and "makes 3 patches" in summary

    assert _click_for_note(page, "#create_grid", "3 patches", 30), _notes(page)
    assert _wait_text(page, "#grid_info", "3", 15)
    assert _cells(page) == 3

    page.check("input[name='mesh_patches'][value='element']")
    assert _wait_text(page, "#mesh_summary_text", "makes 36 patches", 15)
    assert _click_for_note(page, "#create_grid", "36 patches", 30), _notes(page)
    assert _cells(page) == 36

    # the cells are on the map: the grid map's canvas is drawn
    page.wait_for_selector("#ecospace_grid_map canvas", state="attached", timeout=30000)

    # a short spatial run on the mesh grid finishes
    assert _click_for_note(
        page, "#run_spatial_sim", "Running spatial simulation...", 60
    ), _notes(page)
    assert _wait_text(page, "#spatial_run_status", "complete", 240), _text(
        page, "#spatial_run_status"
    )

    # metres: refused without an EPSG code, accepted with one
    metres = _grd(tmp_path / "utm.grd", 1000.0, 499000.0, 4170000.0)
    page.set_input_files("#mesh_file_upload", metres)
    assert _wait_text(page, "#mesh_summary_text", "metres", 20), _text(
        page, "#mesh_summary"
    )
    assert "EPSG code" in _text(page, "#mesh_summary_text")
    assert _click_for_note(page, "#create_grid", "EPSG code", 30), _notes(page)
    page.fill("#mesh_epsg", "32630")
    assert _click_for_note(page, "#create_grid", "36 patches", 30), _notes(page)

    assert errors == []
    assert server.tracebacks() == 0
