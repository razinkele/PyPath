"""Reusable UI components for the PyPath Shiny app.

This module centralizes small, reusable interface primitives so the app shell and
individual pages follow the same design system and remain easier to maintain.
"""

from __future__ import annotations

from shiny import ui


def nav_label(icon_class: str, text: str) -> ui.TagList:
    """Create a sidebar or nav label with an icon and text."""
    return ui.TagList(
        ui.tags.i(class_=f"bi {icon_class}", style="margin-right: 8px;"),
        text,
    )


def section_header(
    title: str,
    subtitle: str | None = None,
    icon_class: str | None = None,
) -> ui.Tag:
    """Create a consistent section heading block for cards and dashboards."""
    children = [ui.h4(title, class_="mb-0")]
    if subtitle:
        children.append(ui.p(subtitle, class_="text-secondary mt-1 mb-0"))
    if icon_class:
        children.insert(
            0,
            ui.tags.i(class_=f"bi {icon_class} me-2", style="color: var(--pp-teal);"),
        )
    return ui.div(*children, class_="d-flex flex-column gap-1")


def metric_card(
    title: str,
    value: str | int | float,
    subtitle: str | None = None,
    tone: str = "primary",
) -> ui.Tag:
    """Create a compact metric/value card used across dashboard sections."""
    tone_class = {
        "primary": "border-primary-subtle",
        "success": "border-success-subtle",
        "warning": "border-warning-subtle",
        "danger": "border-danger-subtle",
        "info": "border-info-subtle",
    }.get(tone, "border-primary-subtle")

    body = [
        ui.div(title, class_="text-uppercase small fw-semibold text-secondary mb-2"),
        ui.div(str(value), class_="h3 mb-0 fw-bold"),
    ]
    if subtitle:
        body.append(ui.div(subtitle, class_="small text-secondary mt-2 mb-0"))

    return ui.div(
        ui.div(*body, class_="card-body"),
        class_=f"card h-100 shadow-sm border {tone_class}",
    )


def feature_card(
    title: str,
    items: list[str],
    *,
    icon_class: str | None = None,
    button_label: str | None = None,
    button_id: str | None = None,
    button_class: str = "btn-primary",
    badge: str | None = None,
) -> ui.Tag:
    """Create a consistent feature card with list items and optional CTA."""
    header_children = []
    if icon_class:
        header_children.append(ui.tags.i(class_=f"bi {icon_class} me-2"))
    header_children.append(title)
    if badge:
        header_children.append(ui.tags.span(badge, class_="badge ms-2"))

    body = [ui.tags.ul(*(ui.tags.li(item) for item in items), class_="mb-0 ps-3")]
    if button_label and button_id:
        body.append(
            ui.input_action_button(
                button_id,
                button_label,
                class_=f"mt-3 {button_class}",
            )
        )

    return ui.card(
        ui.card_header(*header_children),
        ui.card_body(*body),
        class_="h-100 border-0 shadow-sm",
    )


def quick_start_step(
    title: str,
    description: str,
    snippet: str,
    *,
    badge: str | None = None,
) -> ui.Tag:
    """Create a compact step card for onboarding and quick-start guidance."""
    header = [title]
    if badge:
        header.append(ui.tags.span(badge, class_="badge bg-info ms-2"))
    return ui.card(
        ui.card_header(*header),
        ui.card_body(
            ui.p(description),
            ui.tags.code(snippet),
        ),
        class_="h-100 border-0 shadow-sm",
    )


def theme_toggle_button(
    button_id: str = "theme_toggle_button",
    label_id: str = "theme_toggle_label",
    label_text: str = "Dark mode",
) -> ui.Tag:
    """Create the dark-mode toggle used in the app header."""
    return ui.tags.button(
        ui.TagList(
            ui.tags.i(class_="bi bi-moon-fill me-1"),
            ui.tags.span(label_text, id=label_id),
        ),
        type="button",
        id=button_id,
        class_="btn btn-outline-secondary btn-sm topbar-btn",
        **{
            "data-theme-toggle": "true",
            "aria-label": "Toggle light and dark theme",
            "aria-pressed": "false",
        },
    )
