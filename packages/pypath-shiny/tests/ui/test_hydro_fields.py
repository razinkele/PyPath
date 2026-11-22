"""Headless drive of the Ecospace page's hydrodynamic model panel.

A ``.grd`` and a ``boxes_2d.txt`` written by the test are uploaded. The page
says what the box output holds and that it fits the grid, draws a field by
month, and a spatial run with a group carried by the water finishes with
its sub-step check.

CI ignores ``tests/ui``. Run by hand (about a minute):

    pytest packages/pypath-shiny/tests/ui/test_hydro_fields.py -v

Set ``PYPATH_UI_SHOTS`` to a folder to keep screenshots of the two views.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("playwright.sync_api")

from tests.test_hydro_fields import box_file, zone_grd  # noqa: E402
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


def _shot(page, name):
    folder = os.environ.get("PYPATH_UI_SHOTS")
    if folder:
        page.screenshot(path=os.path.join(folder, name), full_page=True)


def test_box_output_shows_fields_and_carries_a_group(server, browser, tmp_path):  # noqa: F811
    errors: list[str] = []
    page = _new_page(browser, errors)
    _setup(page, server.url)
    _create_scenario(page)
    page.click("a.dropdown-toggle:has-text('Advanced')")
    page.click("a[data-value='Ecospace']")
    page.wait_for_selector("#create_grid", state="visible", timeout=30000)

    page.select_option("#grid_type", "shyfem")
    page.wait_for_selector("#mesh_file_upload", state="attached", timeout=10000)
    page.set_input_files("#mesh_file_upload", str(zone_grd(tmp_path)))
    assert _wait_text(page, "#mesh_summary_text", "makes 3 patches", 20)
    assert _click_for_note(page, "#create_grid", "3 patches", 30), _notes(page)

    # the box output: what it holds, and that it fits the grid
    page.click("button.accordion-button:has-text('Hydrodynamic Model')")
    page.wait_for_selector("#box_file_upload", state="attached", timeout=10000)
    page.set_input_files("#box_file_upload", str(box_file(tmp_path)))
    assert _wait_text(page, "#box_summary_text", "14 months", 30), _text(
        page, "#box_summary"
    )
    summary = _text(page, "#box_summary_text")
    assert "2001-01 to 2002-02" in summary and "3 boxes, 2 sections" in summary
    assert "Every one of the grid's 3 patches" in summary

    # the field map, by month
    page.click("#ecospace_tabs a:has-text('Hydrodynamic Fields')")
    assert _wait_text(page, "#field_map_title", "Salinity, mean of 2001-01", 20)
    page.wait_for_selector(
        "#ecospace_field_map canvas", state="attached", timeout=30000
    )
    page.evaluate(
        "$('#box_month').data('ionRangeSlider').update({from: 6});"
        "$('#box_month').trigger('change')"
    )
    assert _wait_text(page, "#field_map_title", "2001-06", 20), _text(
        page, "#field_map_title"
    )
    page.wait_for_timeout(1500)
    _shot(page, "hydro_fields_map.png")

    # carry the first group offered
    page.wait_for_function(
        "() => Object.keys(document.querySelector('#transport_groups')"
        ".selectize.options).length > 0",
        timeout=20000,
    )
    carried = page.evaluate(
        "() => { const s = document.querySelector('#transport_groups').selectize;"
        " const k = Object.keys(s.options)[0]; s.setValue([k]);"
        " return s.options[k].label; }"
    )
    assert _wait_text(page, "#transport_note_text", "Only water that goes", 15)
    assert _click_for_note(
        page, "#run_spatial_sim", "Running spatial simulation...", 60
    ), _notes(page)
    assert _wait_text(page, "#spatial_run_status", "complete", 400), _text(
        page, "#spatial_run_status"
    )
    done = _text(page, "#spatial_transport_text")
    assert "Water carried " + carried.split(" (fast")[0] in done, done
    assert "Sub-step check" in done, done
    _shot(page, "hydro_transport_run.png")

    assert errors == []
    assert server.tracebacks() == 0
