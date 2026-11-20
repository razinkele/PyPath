"""Charts drawn for the page's theme.

The header's toggle switches the page between light and dark in the browser
(``data-bs-theme``). The page head reports the theme to the server as
``input.pypath_theme`` ("light" or "dark"), at the start of the session and
whenever it changes, so a chart can be drawn to match: a render function
that reads it is run again when the user switches.

The theme is per session. Plotly's process-wide default template
(``plotly.io.templates.default``) must not be used for this: it would switch
every user's charts.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from shiny.types import SilentException

__all__ = ["apply_page_theme", "page_is_dark", "page_themed"]

# The page's own text colour in dark mode (--pp-text in custom.css).
_DARK_TEXT = "#e2e8f0"
_TRANSPARENT = "rgba(0,0,0,0)"
# Red below the centre, blue above (as RdBu), through a dark surface colour.
_DARK_DIVERGING = [[0, "#ef4444"], [0.5, "#1e293b"], [1, "#3b82f6"]]


def page_is_dark(input) -> bool:
    """Whether this session's page is in dark mode.

    Reading it inside a render function makes that function run again when
    the theme changes. Light until the page has reported its theme.
    """
    try:
        return input.pypath_theme() == "dark"
    except SilentException:  # not reported yet
        return False


def apply_page_theme(fig, dark: bool):
    """Give a Plotly figure the look of a dark page; a light page changes nothing.

    The figure gets Plotly's dark template (axis, grid and legend colours) on
    a transparent sheet, so the card it sits on shows through, with the
    page's text colour. Returns the same figure.
    """
    if not dark or fig is None:
        return fig
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        font_color=_DARK_TEXT,
    )
    for trace in fig.data:
        # A heatmap centred on a value (zmid) uses a scale that is white
        # there: most of a sparse matrix would be a white slab on the dark
        # page. Keep the sense of the scale, with a dark middle.
        if trace.type == "heatmap" and trace.zmid is not None:
            trace.colorscale = _DARK_DIVERGING
        # Markers coloured along a scale: its dark end would vanish into
        # the page (the top predators of a food web), so outline them.
        marker = getattr(trace, "marker", None)
        if trace.type == "scatter" and marker is not None and marker.colorscale:
            marker.line.color = _DARK_TEXT
            marker.line.width = 1
    return fig


def page_themed(input) -> Callable:
    """Decorator for a chart's render function: theme the figure it returns.

    Put it under ``@render_widget``. The function keeps its name, which is
    the output's id::

        @render_widget
        @page_themed(input)
        def biomass_plot():
            return build_biomass_figure(...)
    """

    def decorate(fn):
        @functools.wraps(fn)
        def themed():
            return apply_page_theme(fn(), page_is_dark(input))

        return themed

    return decorate
