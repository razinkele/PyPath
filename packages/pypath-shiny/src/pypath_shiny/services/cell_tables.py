"""Tables with fixed headings and a tooltip on every cell; hover tips.

Three kinds of table are on the pages: plain tables (built here, by
:func:`cell_table`), Shiny data grids and great_tables tables. The browser
script ``static/cell-tips.js`` gives the cells of all three a tooltip with the
row's label, the column's name and the value. What the page cannot know, the
unrounded value behind a rounded one and the model's remark on a cell, comes
from here: as attributes on a plain table's cells, or for a grid as the data
:func:`cell_tips_data` puts beside it.
"""

from __future__ import annotations

import functools
import json
import numbers
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from htmltools import Tag, tags
from shiny import ui

__all__ = [
    "as_cell_table",
    "cell_table",
    "cell_tips_data",
    "info_tip",
    "remarks_by_row",
    "with_cell_tips",
]

DEFAULT_DECIMALS = 3


def info_tip(text: str, placement: str = "right") -> Tag:
    """An information icon whose tooltip shows *text*.

    For an input's label: ``ui.span("Simulation Years ", info_tip("..."))``.
    Unlike a ``title`` attribute the tooltip shows at once, on keyboard focus
    and on a touch screen.
    """
    icon = tags.i(
        class_="bi bi-info-circle info-tip",
        tabindex="0",
        role="img",
        aria_label="More information",
    )
    return ui.tooltip(icon, text, placement=placement)


def _is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # a list or an array in a cell
        return False


def _is_fraction(value) -> bool:
    """A real number that is not whole (bools and integers are not)."""
    if isinstance(value, (bool, np.bool_, numbers.Integral)):
        return False
    if not isinstance(value, numbers.Real) or not np.isfinite(value):
        return False
    return float(value) != int(value)


def _shown_and_full(value, decimals: int) -> tuple[str, str | None]:
    """The cell's text and, when rounding hid digits, the full value."""
    if _is_missing(value):
        return "", None
    if isinstance(value, (bool, np.bool_, numbers.Integral)):
        return str(value), None
    if isinstance(value, numbers.Real):
        if not np.isfinite(value):
            return str(value), None
        shown = f"{float(value):.{decimals}f}"
        full = repr(float(value))
        return shown, (full if float(shown) != float(value) else None)
    return str(value), None


def _all_whole(values: pd.Series) -> bool:
    """Whether a column holds numbers and every one present is whole."""
    if not pd.api.types.is_float_dtype(values):
        return False
    present = values[np.isfinite(values)]
    return len(present) > 0 and bool((present == np.floor(present)).all())


def _label(value) -> str:
    """A row label or column name as the page reads it from a cell: runs of
    whitespace are one space and the ends are trimmed (cell-tips.js)."""
    return "" if _is_missing(value) else " ".join(str(value).split())


def _decimals_for(decimals, column) -> int:
    if isinstance(decimals, Mapping):
        return int(decimals.get(column, DEFAULT_DECIMALS))
    return int(decimals)


def _remark(remarks: pd.DataFrame | None, row, column) -> str:
    if remarks is None or column not in remarks.columns or row not in remarks.index:
        return ""
    text = remarks.at[row, column]
    return "" if _is_missing(text) else str(text).strip()


def cell_table(
    df: pd.DataFrame,
    *,
    decimals: int | Mapping[str, int] = DEFAULT_DECIMALS,
    remarks: pd.DataFrame | None = None,
    max_height: str = "60vh",
    empty: str = "No data to show.",
) -> Tag:
    """A read-only table that scrolls under fixed headings.

    Parameters
    ----------
    df : pd.DataFrame
        The data, at full precision: numbers are rounded here for display and
        a cell whose rounding hid digits carries the full value for its
        tooltip.
    decimals : int or mapping
        Decimal places for numbers that are not whole; a mapping gives them
        per column (others get 3).
    remarks : pd.DataFrame, optional
        Remark text by the same index and columns; a cell with one carries it
        for its tooltip and is marked.
    max_height : str
        CSS height above which the table scrolls.
    empty : str
        Shown in place of a table with no rows or no columns.
    """
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        return tags.p(empty, class_="text-muted")

    head = tags.tr(*(tags.th(str(column), scope="col") for column in df.columns))
    # A column whose numbers are all whole (a group type or a count stored
    # as a float) is not a measurement: it gets no decimals.
    places = {
        column: 0 if _all_whole(df[column]) else _decimals_for(decimals, column)
        for column in df.columns
    }
    rows = []
    for index, record in zip(df.index, df.itertuples(index=False, name=None)):
        cells = []
        for column, value in zip(df.columns, record):
            shown, full = _shown_and_full(value, places[column])
            attrs = {}
            if full is not None:
                attrs["data-full"] = full
            remark = _remark(remarks, index, column)
            if remark:
                attrs["data-remark"] = remark
                attrs["class"] = "has-remark"
            cells.append(tags.td(shown, **attrs))
        rows.append(tags.tr(*cells))

    return tags.div(
        tags.table(
            tags.thead(head),
            tags.tbody(*rows),
            class_="table table-sm cell-table",
        ),
        class_="cell-table-wrap",
        style=f"max-height: {max_height};",
        # reachable and scrollable from the keyboard
        tabindex="0",
        role="region",
        aria_label="Table; scrolls",
    )


def as_cell_table(**options) -> Callable:
    """Decorator: the function's DataFrame becomes a :func:`cell_table`.

    Put it under ``@render.ui``; *options* go to :func:`cell_table`. The
    function keeps its name, which is the output's id.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def table():
            return cell_table(fn(), **options)

        return table

    return decorate


def remarks_by_row(
    df: pd.DataFrame, remarks: pd.DataFrame | None, key: str = "Group"
) -> pd.DataFrame | None:
    """A model's remarks, indexed like *df*.

    A model keeps its remarks in a frame of its own with a *key* column
    (``RpathParams.remarks``: one row per group). This lines them up with the
    rows of *df* by that column, so a table that shows part of the model, or
    shows it in another order, still finds each cell's remark.
    """
    if remarks is None or key not in remarks.columns or key not in df.columns:
        return None
    by_key = remarks.assign(**{key: remarks[key].astype(str)}).drop_duplicates(key)
    aligned = by_key.set_index(key).reindex(df[key].astype(str))
    aligned.index = df.index
    return aligned


def cell_tips_data(
    output_id: str | None,
    df: pd.DataFrame,
    *,
    remarks: pd.DataFrame | None = None,
    key: str | None = None,
) -> Tag:
    """What the tooltips of a data grid (or a great_tables table) look up.

    A grid draws its own cells, so the full values and remarks cannot ride
    on them; they go beside the grid, by row label and column name, in a
    JSON block the browser script reads for the output *output_id*.

    Parameters
    ----------
    output_id : str or None
        The id of the output that shows the table, when this block is
        rendered by another output (a grid's own output holds only the
        grid). None when the block is returned together with the table, in
        the table's output (:func:`with_cell_tips`).
    df : pd.DataFrame
        The data at full precision.
    remarks : pd.DataFrame, optional
        Remark text by the same index and columns.
    key : str, optional
        The column whose text labels a row (the first one by default). Rows
        that share a label get no entry: the page could not tell them apart.
        Labels are compared as the page reads them from the cells, with runs
        of whitespace as one space.
    """
    data: dict[str, dict[str, dict[str, str]]] = {}
    if df is not None and df.shape[0] and df.shape[1]:
        key = df.columns[0] if key is None else key
        # Labels as the page reads them; rows it could not tell apart, and
        # columns likewise, get no entry.
        labels = df[key].map(_label)
        unique = ~labels.duplicated(keep=False)
        names = [_label(column) for column in df.columns]
        clear = {name for name in names if names.count(name) == 1}
        for index, label in zip(df.index[unique], labels[unique]):
            entry = {}
            for column, name in zip(df.columns, names):
                if name not in clear:
                    continue
                value = df.at[index, column]
                cell = {}
                if _is_fraction(value):
                    cell["full"] = repr(float(value))
                remark = _remark(remarks, index, column)
                if remark:
                    cell["remark"] = remark
                if cell:
                    entry[name] = cell
            if entry:
                data[label] = entry
    # "<" escaped: the text must not be able to close the script element.
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    attrs = {} if output_id is None else {"data-for": output_id}
    return tags.script(
        ui.HTML(payload),
        type="application/json",
        class_="cell-tips-data",
        **attrs,
    )


def with_cell_tips(
    table,
    df: pd.DataFrame,
    *,
    remarks: pd.DataFrame | None = None,
    key: str | None = None,
):
    """A rendered table (a great_tables table, say) with its tooltip data.

    For a ``@render.ui`` function: ``return with_cell_tips(format_x(df), df)``.
    *df* is the data at full precision, as for :func:`cell_tips_data`.
    """
    return ui.TagList(table, cell_tips_data(None, df, remarks=remarks, key=key))
