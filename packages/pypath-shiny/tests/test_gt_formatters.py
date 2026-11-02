"""Test GT formatters module."""

import pandas as pd
import pytest

from pypath_shiny.services.gt_formatters import (
    format_dataframe,
    format_diagnostics_table,
    format_diet_matrix,
    format_params_table,
    format_results_summary,
)


class TestGTFormatters:
    """Test suite for great_tables formatters."""

    def test_format_params_table(self):
        """Test parameter table formatting."""
        df = pd.DataFrame(
            {
                "Group": ["Fish1", "Fish2"],
                "Biomass": [1.234567, 2.345678],
                "Production/Biomass": [0.5123, 0.4567],
            }
        )

        gt = format_params_table(df, title="Test Params")
        assert gt is not None
        # GT object should have tab_header called

    def test_format_results_summary(self):
        """Test results summary formatting."""
        df = pd.DataFrame(
            {
                "Year": [2020, 2021, 2022],
                "Biomass": [100.1, 105.2, 110.3],
                "Catch": [10.5, 11.2, 12.1],
            }
        )

        gt = format_results_summary(df, title="Results", decimals=2)
        assert gt is not None

    def test_format_diagnostics_table(self):
        """Test diagnostic table formatting."""
        df = pd.DataFrame(
            {
                "Metric": ["MSY Check", "Q Sum Check"],
                "Value": [0.95, 0.98],
                "Status": ["OK", "OK"],
            }
        )

        gt = format_diagnostics_table(df)
        assert gt is not None

    def test_format_diet_matrix(self):
        """Test diet matrix formatting."""
        df = pd.DataFrame(
            {
                "Prey1": [0.5, 0.3],
                "Prey2": [0.4, 0.6],
                "Prey3": [0.1, 0.1],
            },
            index=["Pred1", "Pred2"],
        )

        gt = format_diet_matrix(df)
        assert gt is not None

    def test_format_dataframe_generic(self):
        """Test generic dataframe formatting."""
        df = pd.DataFrame(
            {
                "Name": ["A", "B", "C"],
                "Value": [1.111, 2.222, 3.333],
            }
        )

        gt = format_dataframe(df, title="Test Table", decimals=2)
        assert gt is not None

    def test_handles_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        gt = format_dataframe(df)
        assert gt is not None

    def test_handles_non_numeric_data(self):
        """Test handling of non-numeric data."""
        df = pd.DataFrame(
            {
                "Name": ["Group1", "Group2"],
                "Status": ["Active", "Inactive"],
            }
        )
        gt = format_dataframe(df)
        assert gt is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_params_table_header_text_is_readable_on_navy():
    """The navy header fill needs light text; dark default text is unreadable."""
    import re

    import pandas as pd

    from pypath_shiny.services.gt_formatters import format_params_table

    html = format_params_table(
        pd.DataFrame({"Group": ["Fish"], "Biomass": [1.0]}), subtitle="sub"
    ).as_raw_html()
    header_styles = re.findall(
        r'<td[^>]*class="gt_heading[^"]*"[^>]*style="([^"]*)"', html
    )
    assert header_styles, "header cells carry no inline style"
    for css in header_styles:
        assert "#1b3a5c" in css.lower()
        # (?<![-\w]) so that "background-color: #fff" does not count as text
        assert re.search(r"(?<![-\w])color:\s*(#fff(fff)?|white)", css.lower()), css


def test_untrusted_text_is_html_escaped_by_every_formatter():
    """Group names come from uploaded EwE files, EcoBase and biodata lookups.

    The GT tables are inserted with @render.ui, so the formatters must keep
    great_tables' HTML escaping for cells, column labels, titles and
    subtitles (security review on #34).
    """
    import pandas as pd

    from pypath_shiny.services import gt_formatters as g

    evil = "<img src=x onerror=alert(1)>"
    df = pd.DataFrame({"Group": [evil], evil: [1.5], "Biomass": [2.0]})
    builders = [
        lambda: g.format_dataframe(df, title=evil, subtitle=evil),
        lambda: g.format_params_table(df, title=evil, subtitle=evil),
        lambda: g.format_diagnostics_table(df, title=evil),
        lambda: g.format_results_summary(df, title=evil),
        lambda: g.format_diet_matrix(df, title=evil),
        lambda: g.format_time_series_summary(df, title=evil, subtitle=evil),
    ]
    for build in builders:
        html = build().as_raw_html()
        assert "<img src=x" not in html
        assert "&lt;img src=x" in html
