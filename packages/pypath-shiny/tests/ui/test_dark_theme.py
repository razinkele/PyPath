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
    // a matplotlib image, as a plot output holds it
    const probe = document.createElement('div');
    probe.className = 'shiny-plot-output';
    probe.appendChild(document.createElement('img'));
    document.body.appendChild(probe);
    const chartFilter = getComputedStyle(probe.firstChild).filter;
    probe.remove();
    // a stanza-group cell, as the grids style it (pages/utils.py)
    const stanza = document.createElement('div');
    stanza.style.backgroundColor = 'var(--pp-cell-stanza-bg, #e6f3ff)';
    document.body.appendChild(stanza);
    const stanzaCell = getComputedStyle(stanza).backgroundColor;
    stanza.remove();
    return {
        theme: document.documentElement.getAttribute('data-bs-theme'),
        page: css(document.body, 'backgroundColor'),
        text: css(document.body, 'color'),
        heading: css(document.querySelector('h2'), 'color'),
        card: css(document.querySelector('#model_params_table').closest('.card'), 'backgroundColor'),
        sidebar: css(document.querySelector('.well'), 'backgroundImage'),
        navLink: css(document.querySelector('.well .nav-link:not(.active)'), 'color'),
        stanzaCell: stanzaCell,
        chartFilter: chartFilter,
    };
}"""

CHART = "#mti_heatmap_plot .js-plotly-plot"
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

    # A Plotly chart on screen: the impact heatmap of the balanced model.
    page.click("#btn_balance")
    assert _wait_note(page, "Model balanced successfully", 120), _notes(page)
    for tab in ("Analysis", "Trophic Impacts"):
        page.evaluate(
            """(v) => [...document.querySelectorAll('a[data-value]')]
                .filter(a => a.getAttribute('data-value') === v)
                .forEach(a => a.click())""",
            tab,
        )
        time.sleep(1)
    page.wait_for_selector(f"{CHART} .main-svg", state="visible", timeout=60000)

    def chart_sheet(expected, timeout=20):
        """The chart's background once the server has redrawn it for the theme."""
        end = time.monotonic() + timeout
        read = (
            "() => { const s = document.querySelector("
            "'#mti_heatmap_plot .js-plotly-plot .main-svg');"
            " return s ? getComputedStyle(s).backgroundColor : null; }"
        )
        while (colour := page.evaluate(read)) != expected:
            assert time.monotonic() < end, colour
            time.sleep(0.25)
        return colour

    assert chart_sheet("rgb(255, 255, 255)")
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
    # the chart is redrawn for the dark page: a transparent sheet on the card
    assert chart_sheet("rgba(0, 0, 0, 0)")

    # ...and back: light page, and the chart redrawn white
    page.click("[data-theme-toggle='true']")
    back = _look(page, "light")
    assert back["page"] == LIGHT_PAGE and back["chartFilter"] == "none"
    assert chart_sheet("rgb(255, 255, 255)")

    # The choice survives a reload (which starts a new session: the model and
    # its charts are gone, the theme is not).
    page.click("[data-theme-toggle='true']")
    _look(page, "dark")
    page.reload()
    page.wait_for_function(CONNECTED, timeout=60000)
    assert _look(page, "dark")["page"] == DARK_PAGE
    page.wait_for_function(
        "() => Shiny.shinyapp.$inputValues.pypath_theme === 'dark'", timeout=10000
    )
    assert errors == []
    assert server.tracebacks() == 0
