"""Headless drive of the Ecospace wizard's study-area map.

The map is a shiny-deckgl map with its drawing tools: a polygon drawn on it
becomes the study area, a second one replaces it, and the area survives
leaving step 1 and coming back.

CI ignores ``tests/ui``. Run by hand:

    pytest packages/pypath-shiny/tests/ui/test_wizard_area_map.py -v

Needs Playwright with Chromium, as ``test_ecosim_async_runs.py``, whose
server and browser fixtures this file uses. The map draws with WebGL
(software-rendered in a headless browser) on a basemap from the network.
"""

from __future__ import annotations

import re
import time

import pytest

pytest.importorskip("playwright.sync_api")

from tests.ui.test_ecosim_async_runs import (  # noqa: E402
    _new_page,
    _text,
    browser,  # noqa: F401  (fixture)
    pytestmark,  # noqa: F401  (slow; needs /proc)
    server,  # noqa: F401  (fixture)
)

MAP = "#wizard_area_map"
POLYGON_TOOL = f"{MAP} .mapbox-gl-draw_polygon"
AREA = re.compile(r"about ([\d,]+) km² \((\d+) corners\)")


def _draw(page, corners) -> None:
    """Click the polygon tool, each corner, and double-click the last one.

    ``corners`` are fractions of the map's width and height.
    """
    page.click(POLYGON_TOOL)
    box = page.locator(MAP).bounding_box()
    points = [
        (box["x"] + fx * box["width"], box["y"] + fy * box["height"])
        for fx, fy in corners
    ]
    for x, y in points[:-1]:
        page.mouse.click(x, y)
        time.sleep(0.3)
    page.mouse.dblclick(*points[-1])


def _area(page, timeout=15):
    """``(km2, corners)`` once the page reports a study area, else None."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        match = AREA.search(_text(page, "#wizard_area_info"))
        if match:
            return int(match.group(1).replace(",", "")), int(match.group(2))
        time.sleep(0.25)
    return None


def test_a_polygon_drawn_on_the_map_is_the_study_area(server, browser):  # noqa: F811
    errors: list[str] = []
    page = _new_page(browser, errors)
    page.goto(server.url)
    page.wait_for_function(
        "() => window.Shiny && Shiny.shinyapp && Shiny.shinyapp.isConnected()",
        timeout=60000,
    )
    page.click("a.dropdown-toggle:has-text('Advanced')")
    page.click("a[data-value='Ecospace Wizard']")
    try:
        page.wait_for_selector(f"{MAP} canvas", timeout=30000)
        page.wait_for_selector(POLYGON_TOOL, state="visible", timeout=30000)
    except Exception as e:  # no WebGL, or the map libraries did not load
        pytest.skip(f"the map did not start in this browser: {e}")
    assert "No study area drawn yet." in _text(page, "#wizard_area_info")

    _draw(page, [(0.45, 0.40), (0.60, 0.40), (0.60, 0.60), (0.45, 0.60)])
    first = _area(page)
    assert first is not None and first[1] == 4 and first[0] > 0

    # A second polygon replaces the first: one study area.
    _draw(page, [(0.30, 0.30), (0.40, 0.30), (0.35, 0.45)])
    end = time.monotonic() + 15
    while (second := _area(page)) == first and time.monotonic() < end:
        time.sleep(0.25)
    assert second is not None and second[1] == 3 and second[0] < first[0]

    # The map is shown on step 1 only, and keeps the area while away.
    page.click("#wizard_next")
    page.wait_for_selector("#wizard_cell_size", state="visible", timeout=15000)
    assert not page.locator(MAP).is_visible()
    page.click("#wizard_back")
    page.wait_for_selector(POLYGON_TOOL, state="visible", timeout=15000)
    assert _area(page) == second

    # The button clears the area, and a new one can be drawn afterwards.
    page.click("#wizard_clear_area")
    end = time.monotonic() + 10
    while "No study area" not in _text(page, "#wizard_area_info"):
        assert time.monotonic() < end, _text(page, "#wizard_area_info")
        time.sleep(0.25)
    time.sleep(1)  # let the map apply the clear before drawing on it again
    _draw(page, [(0.45, 0.40), (0.60, 0.40), (0.60, 0.60), (0.45, 0.60)])
    again = _area(page)
    # The same corners on the screen: the same area, give or take the layout.
    assert again is not None and again[1] == 4
    assert again[0] == pytest.approx(first[0], rel=0.05)
    assert errors == []
    assert server.tracebacks() == 0
