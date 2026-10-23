"""Validation primitives shared across page modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ValidationSeverity(IntEnum):
    """Issue severity; lower numbers sort first."""

    ERROR = 0
    WARNING = 1
    INFO = 2


@dataclass(frozen=True)
class ValidationIssue:
    """One validation issue tied to a field and section."""

    field: str
    message: str
    severity: ValidationSeverity
    section: str


def sort_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    """Sort issues deterministically by section, severity, then field."""
    return sorted(issues, key=lambda i: (i.section, i.severity, i.field, i.message))


def summarize_issues(issues: list[ValidationIssue]) -> str:
    """Build a short summary string by severity counts."""
    n_error = sum(1 for i in issues if i.severity is ValidationSeverity.ERROR)
    n_warning = sum(1 for i in issues if i.severity is ValidationSeverity.WARNING)
    n_info = sum(1 for i in issues if i.severity is ValidationSeverity.INFO)

    parts = []
    if n_error:
        parts.append(f"{n_error} error{'s' if n_error != 1 else ''}")
    if n_warning:
        parts.append(f"{n_warning} warning{'s' if n_warning != 1 else ''}")
    if n_info:
        parts.append(f"{n_info} info")

    return ", ".join(parts) if parts else "No issues"
