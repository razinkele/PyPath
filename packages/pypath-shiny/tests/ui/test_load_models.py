"""Headless drive of the routes that load a model without building it.

- An example model opened from the Ecosim page alone: it is balanced and its
  own Ecosim settings are selected.
- The Curonian Lagoon example brings its 29-box grid to the Ecospace page,
  and a spatial run on it finishes.
- An imported EwE database's Ecospace scenario is loaded on the Ecospace
  page and runs.

CI ignores ``tests/ui``. Run by hand (a few minutes):

    pytest packages/pypath-shiny/tests/ui/test_load_models.py -v

Set ``PYPATH_UI_SHOTS`` to a folder to keep screenshots, and
``PYPATH_UI_BOXFILE`` to the 29-box model's ``boxes_2d.txt`` to check that
it fits the example's grid.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from tests.ui.test_ecosim_async_runs import (  # noqa: E402
    _click_for_note,
    _new_page,
    _notes,
    _text,
    _wait_text,
    browser,  # noqa: F401  (fixture)
    pytestmark,  # noqa: F401  (slow; needs /proc)
    server,  # noqa: F401  (fixture)
)

PELAGIC = Path(__file__).resolve().parents[4] / "Data" / "curonian_pelagic.sqlite"


def _shot(page, name):
    folder = os.environ.get("PYPATH_UI_SHOTS")
    if folder:
        page.screenshot(path=os.path.join(folder, name), full_page=True)


def _connect(page, url):
    page.goto(url)
    page.wait_for_function(
        "() => window.Shiny && Shiny.shinyapp && Shiny.shinyapp.isConnected()",
        timeout=60000,
    )


def _open_on_ecosim(page, name):
    """Open an example from the Ecosim page; nothing else is touched."""
    page.click("a[data-value='Ecosim']")
    page.wait_for_selector("#btn_ecosim_open_example", state="visible", timeout=30000)
    page.select_option("#ecosim_example_model", name)
    assert _click_for_note(
        page, "#btn_ecosim_open_example", "opened and balanced", 120
    ), _notes(page)
    # its own Ecosim settings are selected
    own = page.locator("input[name='scenario_source'][value='ewe']")
    own.wait_for(state="attached", timeout=30000)
    assert own.is_checked()


def _to_ecospace(page):
    page.click("a.dropdown-toggle:has-text('Advanced')")
    page.click("a[data-value='Ecospace']")
    page.wait_for_selector("#create_grid", state="visible", timeout=30000)


def _patches(page) -> int:
    text = _text(page, "#grid_info")
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    return int(digits[0]) if digits else -1


def test_an_example_opens_from_the_ecosim_page_alone(server, browser):  # noqa: F811
    errors: list[str] = []
    page = _new_page(browser, errors)
    _connect(page, server.url)
    page.click("a[data-value='Ecosim']")
    page.wait_for_selector("#btn_ecosim_open_example", state="visible", timeout=30000)
    # nothing chosen: a warning, and no model
    assert _click_for_note(
        page, "#btn_ecosim_open_example", "Choose an example model first", 15
    ), _notes(page)

    _open_on_ecosim(page, "Lithuanian_Coast")
    assert "Lithuanian sandy coast" in _text(page, "#ewe_scenario_source_ui")
    _shot(page, "ecosim_example_opened.png")
    assert _click_for_note(
        page, "#btn_create_scenario", "Scenario created successfully", 90
    ), _notes(page)

    # the Ecopath page holds the model: a second balance works and keeps
    # the example's settings on offer
    page.click("a[data-value='Ecopath Model']")
    page.wait_for_selector("#btn_balance", state="visible", timeout=30000)
    # The page's charts are drawn as it is first shown. A Balance within that
    # moment closes them half-made, which the browser logs ("[anywidget]
    # Failed to initialize model"; the shinywidgets quirk, harmless): wait as
    # a person would.
    page.wait_for_timeout(3000)
    assert _click_for_note(page, "#btn_balance", "Model balanced successfully", 180), (
        _notes(page)
    )
    page.click("a[data-value='Ecosim']")
    assert page.locator("input[name='scenario_source'][value='ewe']").count() == 1

    # this example has no grid: the Ecospace page says so
    _to_ecospace(page)
    assert _wait_text(page, "#stored_ecospace_hint", "has no spatial grid", 20), _text(
        page, "#stored_ecospace_ui"
    )
    assert errors == []
    assert server.tracebacks() == 0


def test_the_curonian_example_brings_its_grid_and_runs(server, browser):  # noqa: F811
    errors: list[str] = []
    page = _new_page(browser, errors)
    _connect(page, server.url)
    # with no model the Ecospace page says how to get one
    _to_ecospace(page)
    assert _wait_text(page, "#stored_ecospace_hint", "To load a spatial model", 20)

    _open_on_ecosim(page, "Curonian_Lagoon")
    assert _click_for_note(
        page, "#btn_create_scenario", "Scenario created successfully", 90
    ), _notes(page)
    _to_ecospace(page)
    page.wait_for_selector("#btn_load_example_grid", state="visible", timeout=30000)
    assert _click_for_note(page, "#btn_load_example_grid", "29 patches", 60), _notes(
        page
    )
    assert _wait_text(page, "#grid_info", "29", 20) and _patches(page) == 29
    # the default gradient would empty the western boxes
    page.wait_for_function(
        "() => document.querySelector('#habitat_pattern').value === 'uniform'",
        timeout=15000,
    )
    page.wait_for_selector("#ecospace_grid_map canvas", state="attached", timeout=30000)
    page.wait_for_timeout(1500)
    _shot(page, "ecospace_example_grid.png")

    assert _click_for_note(
        page, "#run_spatial_sim", "Running spatial simulation...", 60
    ), _notes(page)
    assert _wait_text(page, "#spatial_run_status", "complete", 400), _text(
        page, "#spatial_run_status"
    )

    # the box model's own output fits the grid, when one is at hand
    box_file = os.environ.get("PYPATH_UI_BOXFILE")
    if box_file:
        page.click("button.accordion-button:has-text('Hydrodynamic Model')")
        page.wait_for_selector("#box_file_upload", state="attached", timeout=10000)
        page.set_input_files("#box_file_upload", box_file)
        assert _wait_text(
            page, "#box_summary_text", "Every one of the grid's 29 patches", 180
        ), _text(page, "#box_summary")
        page.click("#ecospace_tabs a:has-text('Hydrodynamic Fields')")
        assert _wait_text(page, "#field_map_title", "Salinity", 30)
        page.wait_for_timeout(2000)
        _shot(page, "ecospace_example_grid_salinity.png")

    assert errors == []
    assert server.tracebacks() == 0


def test_a_database_ecospace_scenario_is_loaded_and_runs(server, browser):  # noqa: F811
    if not PELAGIC.exists():
        pytest.skip("the Curonian pelagic database is not here")
    errors: list[str] = []
    page = _new_page(browser, errors)
    _connect(page, server.url)

    page.click("a[data-value='Data Import']")
    page.click("a:has-text('EwE File')")
    page.wait_for_selector("#btn_import_ewemdb", state="visible", timeout=30000)
    page.set_input_files("#ewemdb_upload", str(PELAGIC))
    # Import reads the uploaded file: wait until it is all there
    page.wait_for_selector(
        "#ewemdb_upload_progress .progress-bar:has-text('Upload complete')",
        timeout=120000,
    )
    page.click("#btn_import_ewemdb")
    page.wait_for_selector("#btn_use_imported_ewe", state="visible", timeout=120000)
    assert _click_for_note(page, "#btn_use_imported_ewe", "Model transferred", 30), (
        _notes(page)
    )

    page.click("a[data-value='Ecopath Model']")
    page.wait_for_selector("#btn_balance", state="visible", timeout=30000)
    # before the balance the scenario cannot be read: the page says why
    _to_ecospace(page)
    page.wait_for_selector("#btn_load_stored_ecospace", state="visible", timeout=30000)
    assert "94 x 51 cells of 1 km" in _text(page, "#stored_ecospace_ui")
    # the cells are drawn where they lie on the ground unless EwE's
    # coordinates are chosen
    assert page.input_value("#stored_ecospace_placement") == "ground"
    assert _click_for_note(
        page, "#btn_load_stored_ecospace", "Balance the model first", 30
    ), _notes(page)

    page.click("a[data-value='Ecopath Model']")
    assert _click_for_note(page, "#btn_balance", "Model balanced successfully", 180), (
        _notes(page)
    )
    page.click("a[data-value='Ecosim']")
    page.wait_for_selector("#btn_create_scenario", state="visible")
    page.fill("#sim_years", "2")
    time.sleep(0.5)
    assert _click_for_note(
        page, "#btn_create_scenario", "Scenario created successfully", 90
    ), _notes(page)

    _to_ecospace(page)
    page.wait_for_selector("#btn_load_stored_ecospace", state="visible", timeout=30000)
    assert _click_for_note(
        page, "#btn_load_stored_ecospace", "Loaded the Ecospace scenario", 90
    ), _notes(page)
    note = " ".join(_notes(page))
    assert "1,576 water cells of 1 km" in note and "takes about" in note
    assert _wait_text(page, "#grid_info", "1576", 30) or _patches(page) == 1576
    assert _wait_text(
        page, "#ecospace_config_note", "habitat capacity and dispersal rates of", 20
    ), _text(page, "#wizard_config_note")
    page.click("#ecospace_tabs a:has-text('Habitat Map')")
    page.wait_for_selector(
        "#ecospace_habitat_map canvas", state="attached", timeout=30000
    )
    page.wait_for_timeout(2500)
    _shot(page, "ecospace_stored_scenario.png")

    assert _click_for_note(
        page, "#run_spatial_sim", "Running with the Ecospace scenario", 60
    ), _notes(page)
    assert _wait_text(page, "#spatial_run_status", "complete", 500), _text(
        page, "#spatial_run_status"
    )

    # a re-balance of the same groups keeps it
    page.click("a[data-value='Ecopath Model']")
    page.wait_for_selector("#btn_balance", state="visible", timeout=30000)
    page.wait_for_timeout(3000)  # the page's charts settle (see the first test)
    assert _click_for_note(page, "#btn_balance", "Model balanced successfully", 180), (
        _notes(page)
    )
    page.click("a[data-value='Ecosim']")
    page.wait_for_selector("#btn_create_scenario", state="visible")
    assert _click_for_note(
        page, "#btn_create_scenario", "Scenario created successfully", 90
    ), _notes(page)
    _to_ecospace(page)
    assert _wait_text(
        page, "#ecospace_config_note", "habitat capacity and dispersal rates of", 30
    ), _text(page, "#wizard_config_note")
    assert _patches(page) == 1576
    assert _click_for_note(
        page, "#run_spatial_sim", "Running with the Ecospace scenario", 60
    ), _notes(page)
    assert _wait_text(page, "#spatial_run_status", "complete", 500), _text(
        page, "#spatial_run_status"
    )

    # another model ends it: its rows belong to the groups it was read for
    page.click("a[data-value='Ecosim']")
    page.select_option("#ecosim_example_model", "Central_Baltic")
    assert _click_for_note(
        page, "#btn_ecosim_open_example", "is no longer used", 120
    ), _notes(page)
    _to_ecospace(page)
    # the note is redrawn once the page is shown again
    page.wait_for_selector("#ecospace_config_note", state="detached", timeout=30000)
    assert _wait_text(page, "#stored_ecospace_hint", "has no spatial grid", 20)

    assert errors == []
    assert server.tracebacks() == 0
