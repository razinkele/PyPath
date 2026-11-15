"""Diagnostics for a continuous Ecosim run, from ``RsimOutput.solver_info``.

Pure functions with no Shiny import. Every value comes from ``solver_info``
(never from logs), and every function accepts None, which is what the parity
methods (RK4, AB) leave there.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pypath.core.ecosim_continuous import KERNEL_CHECK_TOL

logger = logging.getLogger(__name__)

# Mirrors the core's literal: a quasi-steady separation below this warns.
QS_SEPARATION_MIN = 10.0

_METHOD_LABELS = {
    "lsoda": "LSODA",
    "bdf": "BDF",
    "dop853": "DOP853",
    "radau": "Radau",
}
_MAX_LISTED = 5


@dataclass(frozen=True)
class Badge:
    """One diagnostic badge: a Bootstrap colour level and its text."""

    level: Literal["danger", "warning", "info", "secondary"]
    text: str


def _method_label(info: dict) -> str:
    method = str(info.get("method", ""))
    return _METHOD_LABELS.get(method, method)


def _finite(value) -> float | None:
    """Return ``value`` as a float, or None for None, NaN or non-numbers."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _months(info: dict) -> list[dict]:
    return list(info.get("months") or [])


def _month_number(month: dict, index: int) -> int:
    value = month.get("month")
    return int(value) if isinstance(value, int) else index + 1


def _month_list(numbers: list[int]) -> str:
    shown = ", ".join(str(n) for n in numbers[:_MAX_LISTED])
    extra = len(numbers) - _MAX_LISTED
    return f"{shown} +{extra} more" if extra > 0 else shown


def _group_name(index: int, group_names: Sequence[str] | None) -> str:
    if group_names is not None and 0 <= index < len(group_names):
        return str(group_names[index])
    return f"group {index}"


def _qs_group_names(info: dict, group_names: Sequence[str] | None) -> list[str]:
    return [_group_name(int(g), group_names) for g in info.get("qs_groups") or []]


def solver_badges(
    info: dict | None, group_names: Sequence[str] | None = None
) -> list[Badge]:
    """Badges for a run's ``solver_info``; ``[]`` when there is none."""
    if info is None:
        return []
    months = _months(info)
    badges = [
        Badge("secondary", f"Continuous path (not EwE parity): {_method_label(info)}")
    ]

    failed = [
        _month_number(m, i) for i, m in enumerate(months) if (m.get("status") or 0) != 0
    ]
    if failed:
        badges.append(
            Badge(
                "danger",
                f"Solver did not finish month {_month_list(failed)}; "
                "the trajectory is unreliable there",
            )
        )

    degraded = [_month_number(m, i) for i, m in enumerate(months) if m.get("degraded")]
    if degraded:
        badges.append(
            Badge(
                "warning",
                f"Degraded month {_month_list(degraded)}: "
                "the solver fell back to a cruder step",
            )
        )

    kernel = [
        k
        for k in (_finite(m.get("kernel_check")) for m in months)
        if k is not None and k > KERNEL_CHECK_TOL
    ]
    if kernel:
        badges.append(
            Badge(
                "warning",
                f"Kernel check above {KERNEL_CHECK_TOL:.0e} in {len(kernel)} "
                f"month(s), maximum {max(kernel):.2e}",
            )
        )

    separation = _finite(info.get("qs_separation"))
    if (
        separation is not None
        and math.isfinite(separation)
        and separation < QS_SEPARATION_MIN
    ):
        badges.append(
            Badge(
                "warning",
                f"qs_separation {separation:.3g} < {QS_SEPARATION_MIN:g}: "
                "raise the quasi-steady rate or integrate the fast groups",
            )
        )

    quasi_steady = info.get("fast_groups") == "quasi-steady"
    qs_names = _qs_group_names(info, group_names)
    if quasi_steady and not qs_names:
        rate = _finite(info.get("qs_rate"))
        at = f" at {rate:g} per year" if rate is not None else ""
        badges.append(
            Badge(
                "info",
                f"No group qualified as quasi-steady{at}; every group was integrated",
            )
        )

    changes = []
    interpolation = info.get("forcing_interpolation")
    if interpolation in ("linear", "smooth"):
        changes.append(f"{interpolation} forcing")
    if quasi_steady and qs_names:
        changes.append("quasi-steady " + ", ".join(qs_names))
    if changes:
        badges.append(Badge("info", "Changes the model: " + "; ".join(changes)))

    clipped = [
        _month_number(m, i)
        for i, m in enumerate(months)
        if (_finite(m.get("clips")) or 0) > 0
    ]
    if clipped:
        total = sum(int(_finite(m.get("clips")) or 0) for m in months)
        badges.append(
            Badge(
                "info",
                f"{total} clipped evaluation(s) in month {_month_list(clipped)}; "
                "log states avoid them",
            )
        )
    return badges


def solver_summary(
    info: dict | None, group_names: Sequence[str] | None = None
) -> list[tuple[str, str]]:
    """Label and value rows summarising the solver run; ``[]`` for None."""
    if info is None:
        return []
    n_months = len(_months(info))
    rhs = info.get("rhs_evals")
    if rhs is None:
        rhs_text = "not recorded"
    elif n_months:
        rhs_text = f"{int(rhs)} in total, {rhs / n_months:.0f} per month"
    else:
        rhs_text = f"{int(rhs)} in total"

    rtol = _finite(info.get("rtol"))
    tolerance = (
        f"rtol {rtol:g}; atol: default, follows rtol"
        if rtol is not None
        else "atol: default, follows rtol"
    )

    options = [
        f"positivity: {info.get('positivity', 'clip')}",
        f"forcing interpolation: {info.get('forcing_interpolation', 'constant')}",
        f"fast groups: {info.get('fast_groups', 'states')}",
    ]
    if info.get("fast_groups") == "quasi-steady":
        names = _qs_group_names(info, group_names)
        options.append("quasi-steady groups: " + (", ".join(names) or "none"))
        separation = _finite(info.get("qs_separation"))
        if separation is not None:
            options.append(f"qs_separation {separation:.3g}")

    corrections = info.get("corrections") or []
    ignored = info.get("ignored_switches") or []
    ftime = info.get("ftime_groups") or []
    return [
        ("Method", f"{_method_label(info)} (continuous, not EwE parity)"),
        ("Tolerance", tolerance),
        ("Right-hand-side evaluations", rhs_text),
        ("Model options", "; ".join(options)),
        ("Corrections applied", ", ".join(corrections) or "none"),
        ("EwE switches ignored", ", ".join(ignored) or "none"),
        ("Feeding-time states", f"{len(ftime)} integrated"),
        ("Months", str(n_months)),
    ]


_BASE_COLUMNS = [
    ("Month", None),
    ("nfev", "nfev"),
    ("njev", "njev"),
    ("Jacobians (fresh)", "jac_fresh"),
    ("Jacobians (reused)", "jac_reused"),
    ("steps", "steps"),
    ("status", "status"),
    ("degraded", "degraded"),
    ("clips", "clips"),
    ("hden_passes_max", "hden_passes_max"),
    ("kernel_check", "kernel_check"),
    ("budget_residual", "budget_residual"),
    ("pool_budget_residual", "pool_budget_residual"),
    ("min_state", "min_state"),
]


def _cell(value):
    """None becomes NaN; booleans stay booleans."""
    return float("nan") if value is None else value


def solver_month_table(info: dict | None) -> pd.DataFrame:
    """One row per month; an empty frame for None."""
    columns = [name for name, _ in _BASE_COLUMNS]
    quasi_steady = info is not None and info.get("fast_groups") == "quasi-steady"
    if quasi_steady:
        columns.append("qs_inner_max")
    if info is None:
        return pd.DataFrame(columns=columns)
    rows = []
    for i, m in enumerate(_months(info)):
        row = {"Month": _month_number(m, i)}
        for name, key in _BASE_COLUMNS[1:]:
            row[name] = _cell(m.get(key))
        if quasi_steady:
            row["qs_inner_max"] = _cell(m.get("qs_inner_max"))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def flagged_months(info: dict | None) -> pd.DataFrame:
    """The month-table rows that are failed, degraded, unchecked or clipped."""
    table = solver_month_table(info)
    if table.empty:
        return table
    status = pd.to_numeric(table["status"], errors="coerce").fillna(0) != 0
    degraded = table["degraded"].fillna(False).astype(bool)
    kernel = pd.to_numeric(table["kernel_check"], errors="coerce") > KERNEL_CHECK_TOL
    clips = pd.to_numeric(table["clips"], errors="coerce").fillna(0) > 0
    return table[status | degraded | kernel | clips].reset_index(drop=True)
