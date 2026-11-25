"""Pill tab sets inside pages take the page's colours.

The side navigation is a pill list on a dark panel (``.well``), dark in both
themes, and its light-on-dark link colours were written for every
``.nav-pills`` link. The pill tab sets inside pages took them too: Data
Import's sources, the Results page's two food web views and its data tables.
On a light page their unselected tabs were white on white, so the
"Interactive network" tab beside "Diagram" could not be seen.
"""

import re
from pathlib import Path

import pypath_shiny

SRC = Path(pypath_shiny.__file__).resolve().parent
CSS = (SRC / "static" / "custom.css").read_text(encoding="utf-8")

# "color:" itself, not background-color or border-color
_LIGHT_INK = re.compile(
    r"(?<![-\w])color:\s*(#fff\b|#ffffff\b|rgba\(255,\s*255,\s*255)", re.IGNORECASE
)


def _rules():
    """(selectors, body) of every rule, rules inside ``@media`` included."""
    css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        selectors = [s.strip() for s in match.group(1).split(",") if s.strip()]
        yield selectors, match.group(2)


def _block(selector: str) -> str:
    """The body of the rule that starts a line with exactly *selector*."""
    start = CSS.index("\n" + selector + " {")
    return CSS[start : CSS.index("}", start)]


def test_light_on_dark_pill_links_belong_to_the_side_navigation():
    """White ink on an unselected pill is right on the dark side panel, and
    on a dark page; nowhere else. (A selected pill is filled, so its white
    label is right anywhere.)"""
    loose = [
        selector
        for selectors, body in _rules()
        if _LIGHT_INK.search(body)
        for selector in selectors
        if ".nav-pills" in selector
        and ".well" not in selector
        and ".active" not in selector
        and '[data-bs-theme="dark"]' not in selector
    ]
    assert not loose, loose


def test_pill_tabs_inside_pages_take_the_pages_colours():
    idle = _block(".nav-pills .nav-link")
    # the ink of the unselected in-page tabs (.navset-card-tab), a token that
    # is dark on the light page and light on the dark one
    assert "color: var(--pp-text-secondary)" in idle
    assert "color: var(--pp-text-secondary)" in _block(
        ".navset-card-tab .nav-tabs .nav-link"
    )
    selected = _block(".nav-pills .nav-link.active")
    assert "background: var(--pp-teal)" in selected
    assert "color: #fff" in selected


def test_the_side_navigation_keeps_its_own_colours():
    assert "color: rgba(255, 255, 255, 0.7)" in _block(".well .nav-pills .nav-link")
    assert "background: var(--pp-teal)" in _block(".well .nav-pills .nav-link.active")
    # a dark page must not hand the side panel's links to the in-page rule
    assert '[data-bs-theme="dark"] .well .nav-pills .nav-link' in CSS
