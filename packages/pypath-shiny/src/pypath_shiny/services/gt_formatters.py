"""Great Tables formatters for PyPath data types.

Provides reusable great_tables (GT) formatters for displaying PyPath
model parameters, simulation results, and diagnostic data.
"""

from typing import Optional
import logging

import pandas as pd
from great_tables import GT, style, loc

logger = logging.getLogger(__name__)


def format_params_table(
    df: pd.DataFrame,
    title: str = "Model Parameters",
    subtitle: Optional[str] = None,
) -> GT:
    """Format Ecopath model parameters table with professional styling."""
    gt = GT(df)
    
    if subtitle:
        gt = gt.tab_header(title=title, subtitle=subtitle)
    else:
        gt = gt.tab_header(title=title)
    
    # Format numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns.tolist()
    
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
    
    # Style and interactivity
    gt = gt.tab_style(style=style.fill(color="#1b3a5c"), locations=loc.header())
    gt = gt.opt_interactive()
        
    return gt


def format_results_summary(
    df: pd.DataFrame,
    title: str = "Simulation Results",
    decimals: int = 3,
) -> GT:
    """Format Ecosim results summary table."""
    gt = GT(df)
    gt = gt.tab_header(title=title)
    
    numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns.tolist()
    if numeric_cols:
        gt = gt.fmt_number(columns=numeric_cols, decimals=decimals)
    
    gt = gt.opt_interactive()
        
    return gt


def format_diagnostics_table(
    df: pd.DataFrame,
    title: str = "Diagnostics",
    highlight_problems: bool = True,
) -> GT:
    """Format diagnostic results with visual indicators."""
    gt = GT(df)
    gt = gt.tab_header(title=title)
    
    numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns.tolist()
    if numeric_cols:
        gt = gt.fmt_number(columns=numeric_cols, decimals=3)
    
    gt = gt.opt_interactive()
    return gt


def format_diet_matrix(
    df: pd.DataFrame,
    title: str = "Diet Composition Matrix",
) -> GT:
    """Format diet composition matrix with color gradient."""
    gt = GT(df)
    gt = gt.tab_header(title=title)
    
    numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns.tolist()
    if numeric_cols:
        gt = gt.fmt_percent(columns=numeric_cols, decimals=1)
    
    gt = gt.opt_interactive()
    return gt


def format_dataframe(
    df: pd.DataFrame,
    title: str = "Data Table",
    subtitle: Optional[str] = None,
    decimals: int = 3,
) -> GT:
    """Format any DataFrame as a professional GT table."""
    gt = GT(df)
    
    if subtitle:
        gt = gt.tab_header(title=title, subtitle=subtitle)
    else:
        gt = gt.tab_header(title=title)
    
    numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns.tolist()
    if numeric_cols:
        gt = gt.fmt_number(columns=numeric_cols, decimals=decimals)
    
    gt = gt.opt_interactive()
        
    return gt

