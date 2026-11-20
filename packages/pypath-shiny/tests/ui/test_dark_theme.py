"""Headless drive of the dark theme.

The header's toggle switches the page between light and dark, the choice
survives a reload, and light mode looks as it did.

CI ignores ``tests/ui``. Run by hand:

    pytest packages/pypath-shiny/tests/ui/test_dark_theme.py -v
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

CONNECTED = "() => window.Shiny && Shiny.shinyapp && Shiny.shinyapp.isConnected()"
LOOK = """() => {
    const css = (el, prop) => el ? getComputedStyle(el)[prop] : null;
    const probe = document.createElement('div');
    probe.className = 'js-plotly-plot';
    document.body.appendChild(probe);
    const chartFilter = getComputedStyle(probe).filter;
    probe.remove();
    const stanza = [...document.querySelectorAll('#model_params_table td')]
        .find(td => (td.getAttribute('style') || '').includes('--pp-cell-stanza-bg'));
    return {
        theme: document.documentElement.getAttribute('data-bs-theme'),
        page: css(document.body, 'backgroundColor'),
        text: css(document.body, 'color'),
        heading: css(document.querySelector('h2'), 'color'),
        card: css(document.querySelector('#model_params_table').closest('.card'), 'backgroundColor'),
        sidebar: css(document.querySelector('.well'), 'backgroundImage'),
        navLink: css(document.querySelector('.well .nav-link:not(.active)'), 'color'),
        stanzaCell: css(stanza, 'backgroundColor'),
        chartFilter: chartFilter,
    };
}"""

LIGHT_PAGE, DARK_PAGE = "rgb(241, 245, 249)", "rgb(15, 23, 42)"
NAVY = "rgb(13, 27, 42)"


def _luminance(colour: str) -> float:
    r, g, b = (
        int(v) for v in colour[colour.index("(") + 1 :].split(")")[0].split(",")[:3]
    )
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255


def _look(page, theme) -> dict:
    end = time.monotonic() + 5
    while (look := page.evaluate(LOOK))["theme"] != theme:
        assert time.monotonic() < end, look["theme"]
        time.sleep(0.1)
    time.sleep(0.4)  # colour transitions
    return page.evaluate(LOOK)


def test_the_toggle_switches_the_whole_page(server, browser):  # noqa: F811
    errors: list[str] = []
    page = _new_page(browser, errors)  # a new context: light scheme, nothing saved
    page.goto(server.url)
    page.wait_for_function(CONNECTED, timeout=60000)
    page.select_option("#example_model", "Central_Baltic")
    page.click("#btn_load_example_model")
    assert _wait_note(page, "opened", 60), _notes(page)
    page.wait_for_selector("#model_params_table tbody tr", timeout=30000)

    light = _look(page, "light")
    assert light["page"] == LIGHT_PAGE and light["card"] == "rgb(255, 255, 255)"
    assert light["heading"] == NAVY  # headings are navy on the light page
    assert NAVY in light["sidebar"]
    assert light["stanzaCell"] == "rgb(230, 243, 255)"
    assert light["chartFilter"] == "none"

    page.click("[data-theme-toggle='true']")
    dark = _look(page, "dark")
    assert dark["page"] == DARK_PAGE
    assert _luminance(dark["card"]) < 0.15 and _luminance(dark["text"]) > 0.7
    assert _luminance(dark["heading"]) > 0.7  # readable on the dark page
    # the sidebar keeps its navy and its light links
    assert NAVY in dark["sidebar"] and _luminance(dark["navLink"]) > 0.6
    assert _luminance(dark["stanzaCell"]) < 0.3
    assert "invert" in dark["chartFilter"]

    page.reload()
    page.wait_for_function(CONNECTED, timeout=60000)
    assert _look(page, "dark")["page"] == DARK_PAGE

    page.click("[data-theme-toggle='true']")
    back = _look(page, "light")
    assert back["page"] == LIGHT_PAGE and back["chartFilter"] == "none"
    assert errors == []
    assert server.tracebacks() == 0
