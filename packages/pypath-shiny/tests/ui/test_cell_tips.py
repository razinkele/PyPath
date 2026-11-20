"""Headless drive of the table tooltips and fixed headings.

Every table cell has a tooltip (row label, column, value; the unrounded
value where the table rounds), tables scroll under their headings, and an
input's hint shows on hover and on keyboard focus.

CI ignores ``tests/ui``. Run by hand:

    pytest packages/pypath-shiny/tests/ui/test_cell_tips.py -v

Needs Playwright with Chromium, as ``test_ecosim_async_runs.py``, whose
server and browser fixtures this file uses.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("playwright.sync_api")

from tests.ui.test_ecosim_async_runs import (  # noqa: E402
    _new_page,
    _notes,
    _wait_note,
    browser,  # noqa: F401  (fixture)
    pytestmark,  # noqa: F401  (slow; needs /proc)
    server,  # noqa: F401  (fixture)
)

# The heading's distance from the top of its scrolling box, before and after
# the box is scrolled; a fixed heading does not move.
HEADING_GAPS = """(box) => {
    // a heading cell: the cells are what stay put in a plain table (the
    // row itself scrolls away), the whole heading in a great_tables table
    const cell = box.querySelector('thead th');
    const gap = () => Math.round(
        cell.getBoundingClientRect().top - box.getBoundingClientRect().top);
    const before = gap();
    box.scrollTop = 120;
    return {scrolled: box.scrollTop, before: before, after: gap(),
            background: getComputedStyle(cell).backgroundColor};
}"""


def _open(page, value) -> None:
    """Show the tab (or sub-tab) whose nav link has this value."""
    page.evaluate(
        """(v) => [...document.querySelectorAll('a[data-value]')]
            .filter(a => a.getAttribute('data-value') === v)
            .forEach(a => a.click())""",
        value,
    )


def _tip(page, cell, timeout=5) -> list[str]:
    """The lines of the tooltip shown for ``cell``."""
    page.mouse.move(3, 3)
    time.sleep(0.3)
    cell.scroll_into_view_if_needed()
    cell.hover()
    shown = page.locator(".tooltip.show .tooltip-inner")
    end = time.monotonic() + timeout
    while shown.count() == 0 and time.monotonic() < end:
        time.sleep(0.1)
    if shown.count() == 0:
        return []
    return [
        line.strip() for line in shown.first.inner_text().split("\n") if line.strip()
    ]


def _fixed_heading(page, box_selector, min_rows_height=200) -> dict:
    box = page.locator(box_selector).first
    box.evaluate(f"(b) => {{ b.style.maxHeight = '{min_rows_height}px'; }}")
    return box.evaluate(HEADING_GAPS)


def test_cells_have_tooltips_and_headings_stay_put(server, browser):  # noqa: F811
    errors: list[str] = []
    page = _new_page(browser, errors)
    page.goto(server.url)
    page.wait_for_function(
        "() => window.Shiny && Shiny.shinyapp && Shiny.shinyapp.isConnected()",
        timeout=60000,
    )
    page.select_option("#example_model", "Central_Baltic")
    page.click("#btn_load_example_model")
    assert _wait_note(page, "opened", 60), _notes(page)
    page.wait_for_selector("#model_params_table tbody tr", timeout=30000)

    # The editable parameter grid: row label, column and value.
    row = page.locator("#model_params_table tbody tr").nth(2)
    group = row.locator("td").nth(0).inner_text()
    biomass = _tip(page, row.locator("td").nth(2))
    assert biomass[0] == f"{group} · Biomass" and float(biomass[1]) > 0
    # What the server puts beside the grid reaches the tooltip: the block
    # is replaced here by one with a full value and a remark for this cell
    # (the example model has no remarks of its own).
    page.evaluate(
        """([group]) => {
            const old = document.querySelector(
                'script.cell-tips-data[data-for="model_params_table"]');
            const block = old.cloneNode(false);
            block.textContent = JSON.stringify(
                {[group]: {Biomass: {full: "1.3500001234", remark: "From <the> survey"}}});
            old.replaceWith(block);
        }""",
        [group],
    )
    assert _tip(page, row.locator("td").nth(2)) == [
        f"{group} · Biomass",
        "1.3500001234",
        "From <the> survey",
    ]
    # An edited cell shows its new value at once; the full value sent for
    # the old one must not be shown for it.
    target = row.locator("td").nth(2)
    target.dblclick()
    editor = page.locator(
        "#model_params_table tbody textarea, #model_params_table tbody input"
    ).first
    editor.wait_for(state="visible", timeout=5000)
    editor.fill("7.5")
    editor.press("Enter")
    page.wait_for_function(
        """() => document.querySelectorAll('#model_params_table tbody tr')[2]
            .cells[2].textContent.trim().startsWith('7.5')""",
        timeout=10000,
    )
    edited = _tip(page, row.locator("td").nth(2))
    assert edited[0] == f"{group} · Biomass" and float(edited[1]) == 7.5
    assert "From <the> survey" not in edited or len(edited) == 3
    # A second edit the grid rounds to the same text: the tooltip must
    # follow the model, not keep the value from before the edit.
    target.dblclick()
    editor.wait_for(state="visible", timeout=5000)
    editor.fill("7.5004")
    editor.press("Enter")
    end = time.monotonic() + 10
    while (again := _tip(page, row.locator("td").nth(2)))[1:2] != ["7.5004"]:
        assert time.monotonic() < end, again
        time.sleep(0.5)
    assert row.locator("td").nth(2).inner_text().strip() == "7.5"
    grid = _fixed_heading(page, "#model_params_table .shiny-data-grid")
    assert grid["scrolled"] > 0 and grid["after"] == grid["before"]

    page.click("#btn_balance")
    assert _wait_note(page, "Model balanced successfully", 120), _notes(page)

    # A great_tables table: the same tooltip, and a heading that stays.
    _open(page, "Model Results")
    page.wait_for_selector("table.gt_table td.gt_row", state="visible", timeout=30000)
    cell = page.locator("#model_results_table td.gt_row").nth(14)
    lines = _tip(page, cell)
    assert len(lines) == 2 and " · " in lines[0]
    # the value is the cell's own, or the unrounded number behind it
    assert float(lines[1]) == pytest.approx(float(cell.inner_text()), abs=5e-4)
    # at least one cell of the table tells more than it shows (EE, GE, TL)
    cells = page.locator("#model_results_table td.gt_row")
    longer = False
    for i in range(2, min(cells.count(), 40)):
        shown = cells.nth(i).inner_text()
        tip = _tip(page, cells.nth(i))
        if len(tip) == 2 and len(tip[1]) > len(shown):
            longer = True
            break
    assert longer
    gt = page.locator("#model_results_table table.gt_table").evaluate(
        "(t) => { const b = t.parentElement; b.style.maxHeight = '200px'; return null; }"
    )
    assert gt is None
    gaps = page.locator("#model_results_table table.gt_table").evaluate(
        "(t) => (" + HEADING_GAPS + ")(t.parentElement)"
    )
    assert gaps["scrolled"] > 0 and abs(gaps["after"] - gaps["before"]) <= 1
    assert gaps["background"] not in ("rgba(0, 0, 0, 0)", "transparent")

    # A plain table: rounded on the page, in full in the tooltip.
    _open(page, "Analysis")
    _open(page, "Balance Check")
    page.wait_for_selector(
        "#balance_details_table .cell-table", state="visible", timeout=30000
    )
    row = page.locator("#balance_details_table tbody tr").nth(1)
    shown = row.locator("td").nth(1).inner_text()
    lines = _tip(page, row.locator("td").nth(1))
    assert lines[0] == f"{row.locator('td').nth(0).inner_text()} · EE"
    assert len(lines[1]) > len(shown) and float(lines[1]) == pytest.approx(
        float(shown), abs=5e-4
    )
    plain = _fixed_heading(page, "#balance_details_table .cell-table-wrap", 160)
    assert plain["scrolled"] > 0 and plain["after"] == plain["before"]
    assert plain["background"] not in ("rgba(0, 0, 0, 0)", "transparent")

    # Every group is listed (the tables were once cut to 10 to 20 rows), and
    # a wide table keeps its row names in view when scrolled sideways.
    _open(page, "Export Data")
    page.wait_for_selector(
        "#export_diet_table .cell-table", state="visible", timeout=30000
    )
    wide = page.locator("#export_diet_table .cell-table-wrap").evaluate(
        """(box) => {
            const table = box.querySelector('table');
            const name = table.tBodies[0].rows[0].cells[0];
            const gap = () => Math.round(
                name.getBoundingClientRect().left - box.getBoundingClientRect().left);
            const before = gap();
            box.scrollLeft = 400;
            return {rows: table.tBodies[0].rows.length, scrolled: box.scrollLeft,
                    before: before, after: gap(),
                    heading: table.tHead.rows[0].cells[0].textContent};
        }"""
    )
    assert wide["rows"] > 20 and wide["heading"] == "Prey"
    assert wide["scrolled"] > 0 and wide["after"] == wide["before"]

    # An input's hint: on hover and on keyboard focus.
    _open(page, "Ecosim")
    icon = page.locator(".info-tip").first
    icon.wait_for(state="visible", timeout=30000)
    assert len(" ".join(_tip(page, icon))) > 20
    page.mouse.move(3, 3)
    time.sleep(0.5)
    icon.focus()
    page.wait_for_selector(".tooltip.show", timeout=5000)

    assert errors == []
    assert server.tracebacks() == 0
