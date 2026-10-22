"""Notification wrappers for consistent message types and durations."""

from __future__ import annotations

from shiny import ui

DEFAULT_DURATIONS = {
    "success": 3,
    "warning": 5,
    "error": 8,
    "info": 4,
}


def _notify(message: str, *, level: str, duration: int | None = None) -> None:
    notification_type = {
        "success": "message",
        "warning": "warning",
        "error": "error",
        "info": "default",
    }[level]
    ui.notification_show(
        message,
        type=notification_type,
        duration=DEFAULT_DURATIONS[level] if duration is None else duration,
    )


def notify_success(message: str, *, duration: int | None = None) -> None:
    _notify(message, level="success", duration=duration)


def notify_warning(message: str, *, duration: int | None = None) -> None:
    _notify(message, level="warning", duration=duration)


def notify_error(message: str, *, duration: int | None = None) -> None:
    _notify(message, level="error", duration=duration)


def notify_info(message: str, *, duration: int | None = None) -> None:
    _notify(message, level="info", duration=duration)
