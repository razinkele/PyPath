"""Great Tables formatters for PyPath data types.

Provides reusable great_tables (GT) formatters for displaying PyPath
model parameters, simulation results, and diagnostic data.

The returned GT objects are static HTML tables, rendered with ``@render.ui``
and ``ui.output_ui``. Shiny's ``@render.data_frame`` rejects GT objects, and
``opt_interactive()`` is avoided: its import map is ignored when the table is
inserted after page load. Tables needing row selection, cell editing or
filters stay as ``render.DataGrid``.
"""

import logging
from typing import Optional

import pandas as pd
from great_tables import GT, loc, style

logger = logging.getLogger(__name__)

# Numeric columns that hold category codes, not measurements: the Ecopath group
# type (0 consumer, 1 producer, 2 detritus, 3 fleet) is stored as a float, but
# "1.000" in a table reads as a quantity.
_CODE_COLUMNS = ("Type",)


def _numeric_cols(df: pd.DataFrame, include=("float64", "float32")) -> list:
    """Numeric columns to format as decimals (category-code columns excluded)."""
    cols = df.select_dtypes(include=list(include)).columns.tolist()
    return [c for c in cols if c not in _CODE_COLUMNS]


def _fmt_codes(gt: GT, df: pd.DataFrame) -> GT:
    """Show numeric category-code columns as whole numbers."""
    codes = [
        c
        for c in _CODE_COLUMNS
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if codes:
        gt = gt.fmt_integer(columns=codes)
    return gt


def format_params_table(
    df: pd.DataFrame,
    title: str = "Model Parameters",
    subtitle: Optional[str] = None,
) -> GT:
    """Format Ecopath model parameters table with professional styling."""
    gt = _fmt_codes(GT(df), df)

    if subtitle:
        gt = gt.tab_header(title=title, subtitle=subtitle)
    else:
        gt = gt.tab_header(title=title)

    # Format numeric columns
    numeric_cols = _numeric_cols(df)

    # Biomass/Abundance: 4 decimals
    biomass_cols = [c for c in numeric_cols if "Biomass" in c or "Abundance" in c]
    if biomass_cols:
        gt = gt.fmt_number(columns=biomass_cols, decimals=4)

    # Ratios: 3 decimals
    ratio_cols = [c for c in numeric_cols if "/" in c or "Ratio" in c]
    if ratio_cols:
        gt = gt.fmt_number(columns=ratio_cols, decimals=3)

    # Other: 2 decimals
    other_cols = [c for c in numeric_cols if c not in biomass_cols + ratio_cols]
    if other_cols:
        gt = gt.fmt_number(columns=other_cols, decimals=2)

    # Header styling: navy fill needs light text (the default text is dark)
    gt = gt.tab_style(
        style=[style.fill(color="#1b3a5c"), style.text(color="#ffffff")],
        locations=loc.header(),
    )

    return gt


def format_results_summary(
    df: pd.DataFrame,
    title: str = "Simulation Results",
    decimals: int = 3,
) -> GT:
    """Format Ecosim results summary table."""
    gt = _fmt_codes(GT(df), df)
    gt = gt.tab_header(title=title)

    numeric_cols = _numeric_cols(df)
    if numeric_cols:
        gt = gt.fmt_number(columns=numeric_cols, decimals=decimals)

    return gt


def format_diagnostics_table(
    df: pd.DataFrame,
    title: str = "Diagnostics",
    highlight_problems: bool = True,
) -> GT:
    """Format diagnostic results with visual indicators."""
    gt = _fmt_codes(GT(df), df)
    gt = gt.tab_header(title=title)

    numeric_cols = _numeric_cols(df)
    if numeric_cols:
        gt = gt.fmt_number(columns=numeric_cols, decimals=3)

    return gt


def format_diet_matrix(
    df: pd.DataFrame,
    title: str = "Diet Composition Matrix",
) -> GT:
    """Format diet composition matrix with color gradient."""
    gt = _fmt_codes(GT(df), df)
    gt = gt.tab_header(title=title)

    numeric_cols = _numeric_cols(df)
    if numeric_cols:
        gt = gt.fmt_percent(columns=numeric_cols, decimals=1)

    return gt


def format_time_series_summary(
    df: pd.DataFrame,
    title: str = "Time Series Data",
    subtitle: Optional[str] = None,
    decimals: int = 4,
) -> GT:
    """Format time series summary data with professional styling."""
    gt = _fmt_codes(GT(df), df)

    if subtitle:
        gt = gt.tab_header(title=title, subtitle=subtitle)
    else:
        gt = gt.tab_header(title=title)

    numeric_cols = _numeric_cols(df, ("float64", "float32", "int64", "int32"))
    # Skip time-related columns like Year, Month
    numeric_cols = [c for c in numeric_cols if c not in ["Year", "Month"]]
    if numeric_cols:
        gt = gt.fmt_number(columns=numeric_cols, decimals=decimals)

    return gt


def format_dataframe(
    df: pd.DataFrame,
    title: str = "Data Table",
    subtitle: Optional[str] = None,
    decimals: int = 3,
) -> GT:
    """Format any DataFrame as a professional GT table."""
    gt = _fmt_codes(GT(df), df)

    if subtitle:
        gt = gt.tab_header(title=title, subtitle=subtitle)
    else:
        gt = gt.tab_header(title=title)

    numeric_cols = _numeric_cols(df)
    if numeric_cols:
        gt = gt.fmt_number(columns=numeric_cols, decimals=decimals)

    return gt
