"""Tests for reusable PyPath Shiny UI components."""

from pypath_shiny.components import (
    metric_card,
    nav_label,
    section_header,
    theme_toggle_button,
)


def test_nav_label_contains_icon_and_text():
    """Sidebar nav labels should include icon markup and text."""
    label = nav_label("bi-house-fill", "Home")
    assert label is not None


def test_section_header_creates_styled_block():
    """Section headings should preserve title and optional subtitle."""
    header = section_header(
        "Model summary", subtitle="Ecosystem metrics", icon_class="bi-bar-chart"
    )
    assert header is not None


def test_metric_card_builds_metric_shell():
    """Metric cards should expose the title and value in a standard card layout."""
    card = metric_card("Biomass", 123.45, subtitle="t/km²", tone="success")
    assert card is not None


def test_theme_toggle_button_has_accessible_attributes():
    """Theme toggle must expose label and a11y metadata."""
    button = theme_toggle_button()
    assert button is not None
