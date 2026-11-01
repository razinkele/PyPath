"""Plotly visualization helpers for PyPath Shiny pages."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go


def _safe_palette(name: str | None, n: int) -> list[str]:
    """Return a palette of color values for Plotly traces."""
    if name in {None, "", "default"}:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    else:
        try:
            import plotly.express as px

            options = getattr(px.colors.qualitative, name, None)
            if options is not None:
                colors = list(options)
            else:
                colors = px.colors.qualitative.Plotly
        except Exception:
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
            ]

    if not isinstance(colors, Sequence):
        colors = list(colors)

    if len(colors) < n:
        colors = list(colors) + [f"#{i:06x}" for i in range(len(colors), n)]
    return list(colors[:n])


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert a hex color to an rgba string for Plotly fills."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(99, 110, 250, {alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def build_biomass_figure(
    sim,
    groups: Sequence[str],
    selected: Sequence[str],
    log_scale: bool = False,
    palette: str | None = "tab10",
    title: str = "Biomass Time Series",
) -> go.Figure:
    """Build a Plotly biomass timeseries figure for selected groups."""
    fig = go.Figure()

    if sim is None or not groups or not selected:
        fig.update_layout(
            title=title,
            annotations=[
                {
                    "text": "No simulation data available",
                    "x": 0.5,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 16},
                }
            ],
        )
        return fig

    biomass_matrix = np.asarray(sim.out_Biomass, dtype=float)
    time = np.arange(biomass_matrix.shape[0]) / 12
    colors = _safe_palette(palette, len(selected))

    for i, group in enumerate(selected):
        if group not in groups:
            continue
        idx = groups.index(group)
        if biomass_matrix.shape[1] > len(groups):
            idx += 1
        biomass = np.asarray(biomass_matrix[:, idx], dtype=float)
        if log_scale:
            biomass = np.log10(np.maximum(biomass, 1e-10))

        fig.add_trace(
            go.Scatter(
                x=time,
                y=biomass,
                mode="lines",
                line={"width": 2, "color": colors[i % len(colors)]},
                name=group,
                hovertemplate="%{x:.2f} years<br>%{y:.4g}<extra>%{fullData.name}</extra>",
            )
        )

    if not fig.data:
        fig.update_layout(
            title=f"{title} (no data)",
            annotations=[
                {
                    "text": "Select groups to display",
                    "x": 0.5,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 16},
                }
            ],
        )
        return fig

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Year",
        yaxis_title="Log10(Biomass)" if log_scale else "Biomass",
        legend_title="Group",
        hovermode="x unified",
        margin={"l": 60, "r": 130, "t": 50, "b": 50},
    )
    return fig


def build_catch_figure(
    sim,
    groups: Sequence[str] | None = None,
    palette: str | None = "tab10",
    title: str = "Annual Catch",
) -> go.Figure:
    """Build a Plotly annual catch figure."""
    fig = go.Figure()

    if sim is None:
        fig.update_layout(
            title=title,
            annotations=[
                {
                    "text": "No simulation data available",
                    "x": 0.5,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 16},
                }
            ],
        )
        return fig

    catch_matrix = np.asarray(sim.annual_Catch, dtype=float)
    years = np.arange(catch_matrix.shape[0]) + 1
    total_catch = np.sum(catch_matrix[:, 1:], axis=1)
    color = _safe_palette(palette, 1)[0]

    fig.add_trace(
        go.Scatter(
            x=years,
            y=total_catch,
            mode="lines+markers",
            line={"width": 2, "color": color},
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, alpha=0.2),
            name="Total catch",
            hovertemplate="Year %{x}<br>Catch %{y:.4g}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Year",
        yaxis_title="Total Catch",
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return fig
