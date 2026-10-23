"""Feedback reporting and dashboard page."""

from __future__ import annotations

import sqlite3
import time
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui

from pypath_shiny import __version__
from pypath_shiny.services.feedback_store import (
    add_feedback,
    feedback_stats,
    get_feedback,
    list_feedback,
    resolve_feedback,
    weekly_feedback_counts,
)
from pypath_shiny.services.notifications import (
    notify_error,
    notify_success,
    notify_warning,
)

CATEGORIES = {
    "bug": "Bug report",
    "feature": "Improvement suggestion",
    "general": "General feedback",
    "data_quality": "Data quality",
}

STATUS_FILTERS = {
    "open": ("open",),
    "resolved": ("addressed", "wontfix", "duplicate"),
    "all": ("open", "addressed", "wontfix", "duplicate"),
}

MAX_TITLE_LEN = 200
MAX_MESSAGE_LEN = 5000
MAX_STEPS_LEN = 2000
RATE_LIMIT_SECONDS = 30.0


def system_info_text(app_version: str, page: str, browser: str, session_id: str) -> str:
    return (
        f"App version: {app_version}\n"
        f"Page: {page or 'Unknown'}\n"
        f"Browser: {browser or 'Unknown'}\n"
        f"Session id: {session_id}"
    )


def validate_feedback_submission(
    title: str, message: str, steps: str | None
) -> str | None:
    if not title.strip():
        return "Title is required."
    if not message.strip():
        return "Description is required."
    if len(title.strip()) > MAX_TITLE_LEN:
        return f"Title must be <= {MAX_TITLE_LEN} characters."
    if len(message.strip()) > MAX_MESSAGE_LEN:
        return f"Description must be <= {MAX_MESSAGE_LEN} characters."
    if steps and len(steps.strip()) > MAX_STEPS_LEN:
        return f"Steps to reproduce must be <= {MAX_STEPS_LEN} characters."
    return None


def rate_limited(last_submit: float | None, now: float) -> bool:
    if last_submit is None:
        return False
    return (now - last_submit) < RATE_LIMIT_SECONDS


def _display_title(entry: dict) -> str:
    title = (entry.get("title") or "").strip()
    return title if title else "(untitled)"


def build_feedback_modal(info_text: str) -> ui.Tag:
    return ui.modal(
        ui.input_select(
            "fb_type",
            "Type",
            choices=CATEGORIES,
            selected="general",
        ),
        ui.input_text("fb_title", "Title", placeholder="Short summary"),
        ui.input_text_area("fb_message", "Description", rows=5),
        ui.input_text_area(
            "fb_steps",
            "Steps to reproduce (for bug reports)",
            rows=3,
            placeholder="Optional unless this is a bug report",
        ),
        ui.input_select(
            "fb_rating",
            "Rating (optional)",
            choices={"": "No rating", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5"},
            selected="",
        ),
        ui.tags.details(
            ui.tags.summary("System information"),
            ui.tags.pre(info_text, style="font-size: 11px;"),
            style="background:#f8f9fa;border-radius:5px;padding:.5rem",
        ),
        title="Send Feedback",
        size="l",
        easy_close=False,
        footer=ui.div(
            ui.input_action_button("fb_cancel", "Cancel", class_="btn-secondary"),
            ui.input_action_button(
                "fb_submit", "Submit Feedback", class_="btn-primary"
            ),
        ),
    )


def feedback_ui() -> ui.Tag:
    """Feedback dashboard UI."""
    return ui.div(
        ui.layout_columns(
            ui.h2("Feedback Dashboard", class_="mb-3"),
            ui.tags.button(
                "Send feedback",
                id="feedback_send_btn",
                class_="btn btn-primary",
                type="button",
                onclick="Shiny.setInputValue('feedback_open', Math.random()); return false;",
            ),
            col_widths=[9, 3],
        ),
        ui.layout_columns(
            ui.output_ui("feedback_tiles"),
            col_widths=[12],
        ),
        ui.card(
            ui.card_header("Weekly trend"),
            ui.output_plot("feedback_trend", height="280px"),
            class_="mb-3",
        ),
        ui.layout_columns(
            ui.input_select(
                "fb_status_filter",
                "Status",
                choices={"open": "Open", "resolved": "Resolved", "all": "All"},
                selected="open",
            ),
            ui.input_select(
                "fb_type_filter",
                "Type",
                choices={"all": "All types", **CATEGORIES},
                selected="all",
            ),
            col_widths=[4, 4],
        ),
        ui.output_data_frame("feedback_grid"),
        ui.input_selectize(
            "fb_selected_id", "Selected feedback", choices={}, selected=None
        ),
        ui.output_ui("feedback_detail"),
    )


def feedback_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    *,
    current_page: Callable[[], str] | None = None,
) -> None:
    """Feedback server logic."""
    dirty = reactive.Value(0)
    last_submit = [None]
    modal_open = [False]
    snapshot = [None]

    def _browser() -> str:
        try:
            return str(input.fb_browser_info() or "")
        except Exception:
            return ""

    def _page() -> str:
        if current_page is None:
            return "Unknown"
        try:
            value = current_page()
            return str(value) if value else "Unknown"
        except Exception:
            return "Unknown"

    def _close_modal() -> None:
        if not modal_open[0]:
            return
        modal_open[0] = False
        ui.modal_remove()

    def _entry_choices(entries: list[dict]) -> dict[str, str]:
        return {str(e["id"]): f"#{e['id']} - {_display_title(e)}" for e in entries}

    @reactive.effect
    @reactive.event(input.feedback_open)
    def _open_modal() -> None:
        if modal_open[0]:
            return
        snapshot[0] = {
            "app_version": __version__,
            "page": _page(),
            "browser": _browser(),
            "session_id": session.id,
        }
        modal_open[0] = True
        snap = snapshot[0]
        ui.modal_show(
            build_feedback_modal(
                system_info_text(
                    snap["app_version"],
                    snap["page"],
                    snap["browser"],
                    snap["session_id"],
                )
            )
        )

    @reactive.effect
    @reactive.event(input.fb_cancel)
    def _cancel_feedback() -> None:
        _close_modal()

    @reactive.effect
    @reactive.event(input.fb_submit)
    def _submit_feedback() -> None:
        if not modal_open[0]:
            return

        now = time.monotonic()
        if rate_limited(last_submit[0], now):
            notify_warning("Please wait before submitting feedback again.")
            return

        title = str(input.fb_title() or "").strip()
        message = str(input.fb_message() or "").strip()
        category = str(input.fb_type() or "general")
        steps_raw = str(input.fb_steps() or "").strip()
        steps = steps_raw if category == "bug" else None

        err = validate_feedback_submission(title, message, steps)
        if err:
            notify_warning(err)
            return

        snap = snapshot[0] or {}
        rating_raw = str(input.fb_rating() or "").strip()
        rating = int(rating_raw) if rating_raw.isdigit() else None

        try:
            add_feedback(
                category=category,
                title=title,
                message=message,
                steps=steps,
                rating=rating,
                page=snap.get("page"),
                app_version=snap.get("app_version"),
                browser=snap.get("browser"),
                session_id=snap.get("session_id"),
            )
        except (sqlite3.Error, OSError) as exc:
            notify_error(f"Could not save feedback: {exc}")
            _close_modal()
            return

        last_submit[0] = now
        _close_modal()
        notify_success("Thank you! Your feedback has been saved.", duration=5)
        dirty.set(dirty() + 1)

    @reactive.calc
    def _entries() -> list[dict]:
        dirty()
        status_key = str(input.fb_status_filter() or "open")
        status_values = STATUS_FILTERS.get(status_key, STATUS_FILTERS["open"])
        category = str(input.fb_type_filter() or "all")
        return list_feedback(statuses=status_values, category=category)

    @output
    @render.ui
    def feedback_tiles():
        dirty()
        stats = feedback_stats()
        return ui.layout_columns(
            ui.value_box(
                "Total",
                str(stats["total"]),
                showcase=ui.tags.i(class_="bi bi-chat-left-text"),
            ),
            ui.value_box(
                "Open", str(stats["open"]), showcase=ui.tags.i(class_="bi bi-inbox")
            ),
            ui.value_box(
                "Resolved",
                str(stats["resolved"]),
                showcase=ui.tags.i(class_="bi bi-check2-circle"),
            ),
            ui.value_box(
                "Last 7 days",
                str(stats["last_7_days"]),
                showcase=ui.tags.i(class_="bi bi-calendar-week"),
            ),
            col_widths=[3, 3, 3, 3],
        )

    @output
    @render.plot
    def feedback_trend():
        dirty()
        points = weekly_feedback_counts(weeks=10)
        fig, ax = plt.subplots(figsize=(8, 3))
        if not points:
            ax.text(0.5, 0.5, "No feedback yet", ha="center", va="center")
            ax.set_axis_off()
            return fig

        labels = [p[0] for p in points]
        values = [p[1] for p in points]
        ax.bar(labels, values, color="#2d7ff9")
        ax.set_ylabel("Entries")
        ax.set_xlabel("ISO week")
        ax.set_title("Feedback entries per week")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        return fig

    @output
    @render.data_frame
    def feedback_grid():
        entries = _entries()
        if not entries:
            return render.DataGrid(
                pd.DataFrame(
                    [
                        {
                            "Date": "",
                            "Type": "",
                            "Title": "No feedback found",
                            "Status": "",
                        }
                    ]
                ),
                width="100%",
            )

        df = pd.DataFrame(
            [
                {
                    "ID": int(e["id"]),
                    "Date": str(e.get("created_at", ""))[:10],
                    "Type": CATEGORIES.get(
                        e.get("category", "general"), e.get("category", "")
                    ),
                    "Title": _display_title(e),
                    "Status": e.get("status", "open"),
                }
                for e in entries
            ]
        )
        return render.DataGrid(df, width="100%", filters=True)

    @reactive.effect
    def _sync_selected_feedback() -> None:
        entries = _entries()
        choices = _entry_choices(entries)
        current = str(input.fb_selected_id() or "")
        selected = (
            current
            if current in choices
            else (next(iter(choices)) if choices else None)
        )
        ui.update_selectize("fb_selected_id", choices=choices, selected=selected)

    @output
    @render.ui
    def feedback_detail():
        selected = str(input.fb_selected_id() or "")
        if not selected:
            return ui.div(
                "Select feedback to inspect details.", class_="alert alert-info"
            )

        try:
            entry_id = int(selected)
        except ValueError:
            return ui.div("Invalid feedback selection.", class_="alert alert-warning")

        entry = get_feedback(entry_id)
        if entry is None:
            return ui.div("Feedback entry not found.", class_="alert alert-warning")

        details = [
            ui.tags.p(ui.tags.b("Date: "), str(entry.get("created_at", ""))),
            ui.tags.p(
                ui.tags.b("Type: "),
                CATEGORIES.get(
                    entry.get("category", "general"), entry.get("category", "")
                ),
            ),
            ui.tags.p(ui.tags.b("Title: "), _display_title(entry)),
            ui.tags.p(ui.tags.b("Status: "), str(entry.get("status", "open"))),
            ui.tags.p(ui.tags.b("Description: "), str(entry.get("message", ""))),
        ]

        if entry.get("steps"):
            details.append(ui.tags.p(ui.tags.b("Steps: "), str(entry.get("steps"))))
        if entry.get("resolved_note"):
            details.append(
                ui.tags.p(
                    ui.tags.b("Resolution note: "), str(entry.get("resolved_note"))
                )
            )

        if entry.get("status") == "open":
            details.extend(
                [
                    ui.tags.hr(),
                    ui.h6("Resolve feedback"),
                    ui.input_select(
                        "fb_resolve_status",
                        "Resolution",
                        choices={
                            "addressed": "Addressed",
                            "wontfix": "Won't fix",
                            "duplicate": "Duplicate",
                        },
                        selected="addressed",
                    ),
                    ui.input_text_area("fb_resolve_note", "Resolution note", rows=3),
                    ui.input_text(
                        "fb_fix_reference",
                        "Fix reference (optional)",
                        placeholder="Commit SHA / PR URL",
                    ),
                    ui.input_numeric(
                        "fb_duplicate_of", "Duplicate of feedback ID", value=None, min=1
                    ),
                    ui.input_action_button(
                        "fb_resolve_btn", "Save resolution", class_="btn-primary"
                    ),
                ]
            )

        return ui.card(ui.card_body(*details))

    @reactive.effect
    @reactive.event(input.fb_resolve_btn)
    def _resolve_feedback() -> None:
        selected = str(input.fb_selected_id() or "")
        if not selected:
            notify_warning("Select feedback before resolving.")
            return

        try:
            feedback_id = int(selected)
        except ValueError:
            notify_warning("Invalid feedback selection.")
            return

        status = str(input.fb_resolve_status() or "addressed")
        note = str(input.fb_resolve_note() or "").strip()
        if not note:
            notify_warning("Resolution note is required.")
            return

        duplicate_raw = input.fb_duplicate_of()
        duplicate_of = int(duplicate_raw) if duplicate_raw not in (None, "") else None

        ok = resolve_feedback(
            feedback_id=feedback_id,
            status=status,
            note=note,
            fix_reference=str(input.fb_fix_reference() or "").strip() or None,
            duplicate_of=duplicate_of,
        )
        if not ok:
            notify_warning(
                "Could not resolve feedback. Check status and duplicate target."
            )
            return

        notify_success("Feedback updated.")
        dirty.set(dirty() + 1)


