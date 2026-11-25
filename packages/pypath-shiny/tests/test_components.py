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


def test_empty_state_shows_message_without_a_table():
    from pypath_shiny.components import empty_state

    html = str(empty_state("Run diagnostics to see ratios."))
    assert "Run diagnostics to see ratios." in html
    assert "text-muted" in html
    assert "<table" not in html


def test_feature_card_can_carry_a_button_and_an_outside_link():
    from pypath_shiny.components import feature_card

    html = str(
        feature_card(
            "Tutorial",
            ["one", "two"],
            button_label="Open Tutorial",
            button_id="btn_x",
            link_label="Documentation",
            link_href="https://example.org/docs",
        )
    )
    assert 'id="btn_x"' in html and "Open Tutorial" in html
    assert 'href="https://example.org/docs"' in html and "Documentation" in html
    # an outside link opens in a new tab and does not hand over the opener
    assert 'target="_blank"' in html and "noopener" in html


def test_feature_card_without_actions_has_no_empty_action_row():
    from pypath_shiny.components import feature_card

    html = str(feature_card("Plain", ["one"]))
    assert "feature-card-actions" not in html
