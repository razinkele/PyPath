"""Headless drive of the Ecospace page's Cancel button.

The spatial run reports each month and can be stopped between months: the
page shows ``Month m of n (p%)``, Cancel ends the run at its next month
boundary, and a new run is accepted soon after.

CI ignores ``tests/ui``. Run by hand (a few minutes):

    pytest packages/pypath-shiny/tests/ui/test_ecospace_cancel.py -v

Needs Playwright with Chromium and Linux, as ``test_ecosim_async_runs.py``,
whose server and page helpers this file uses.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("playwright.sync_api")

from tests.ui.test_ecosim_async_runs import (  # noqa: E402
    MONTH,
    _clear_notes,
    _click_for_note,
    _create_scenario,
    _new_page,
    _notes,
    _setup,
    _text,
    _wait_note,
    _wait_text,
    browser,  # noqa: F401  (fixture)
    pytestmark,  # noqa: F401  (slow; needs /proc)
    server,  # noqa: F401  (fixture)
)

CANCELLED = "Cancelled: the run stops at the end of its current month"
FINISHING = "The cancelled run is finishing its current month."
ACCEPTED = "Running spatial simulation..."
# A month of the 25-patch Central Baltic run takes well under this.
RESTART_LIMIT = 20.0


def _month(page):
    match = MONTH.search(_text(page, "#spatial_month_progress"))
    return None if match is None else tuple(int(g) for g in match.groups())


def _wait_month(page, after=0, timeout=90):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        month = _month(page)
        if month is not None and month[0] > after:
            return month
        time.sleep(0.25)
    return None


def test_cancel_stops_a_spatial_run_and_a_new_one_starts(server, browser):  # noqa: F811
    errors: list[str] = []
    page = _new_page(browser, errors)
    _setup(page, server.url)
    _create_scenario(page)

    page.click("a.dropdown-toggle:has-text('Advanced')")  # the tab's menu
    page.click("a[data-value='Ecospace']")
    page.wait_for_selector("#create_grid", state="visible", timeout=30000)
    assert _click_for_note(page, "#create_grid", "regular grid", 30), _notes(page)

    # Run: the Cancel button and the month progress appear, and months advance.
    assert _click_for_note(page, "#run_spatial_sim", ACCEPTED, 60), _notes(page)
    page.wait_for_selector("#btn_cancel_spatial", state="visible", timeout=30000)
    first = _wait_month(page)
    assert first is not None, _text(page, "#spatial_run_status")
    month, n_months, percent = first
    assert n_months == 600 and percent == round(100 * month / n_months)
    later = _wait_month(page, after=month)
    assert later is not None and later[0] > month

    # Cancel: notified, shown as cancelled, and the server goes quiet long
    # before the 50-year run could have ended on its own.
    assert _click_for_note(page, "#btn_cancel_spatial", CANCELLED, 15), _notes(page)
    cancelled_at = time.monotonic()
    assert _wait_text(
        page, "#spatial_run_status", "Spatial simulation cancelled.", 10
    ), _text(page, "#spatial_run_status")
    assert page.locator("#btn_cancel_spatial").count() == 0
    assert server.wait_idle(limit=RESTART_LIMIT) is not None

    # A new run is accepted (a click during the last month is refused politely).
    accepted = False
    while not accepted and time.monotonic() - cancelled_at < RESTART_LIMIT + 30:
        _clear_notes(page)
        page.click("#run_spatial_sim")
        if _wait_note(page, ACCEPTED, 5):
            accepted = True
        else:
            assert any(FINISHING in n for n in _notes(page)), _notes(page)
    assert accepted
    page.wait_for_selector("#btn_cancel_spatial", state="visible", timeout=30000)
    assert _wait_month(page) is not None

    # Leave nothing computing for the next test's server.
    assert _click_for_note(page, "#btn_cancel_spatial", CANCELLED, 15), _notes(page)
    assert server.wait_idle(limit=RESTART_LIMIT) is not None
    assert errors == []
    assert server.tracebacks() == 0
