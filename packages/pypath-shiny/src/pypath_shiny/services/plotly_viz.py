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
    relative: bool = False,
    empty_message: str = "No simulation data available",
) -> go.Figure:
    """Build a Plotly biomass timeseries figure for selected groups.

    ``relative=True`` divides each series by its first value (all zeros if
    that is 0) and draws a dashed reference line at 1. ``sim.out_Biomass``
    may carry a leading "Outside" column (Ecosim output); it is skipped when
    the matrix has one more column than ``groups``.
    """
    fig = go.Figure()

    if sim is None or not groups or not selected:
        fig.update_layout(
            title=title,
            annotations=[
                {
                    "text": empty_message,
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
        if relative:
            start = biomass[0] if biomass.size else 0.0
            biomass = biomass / start if start > 0 else np.zeros_like(biomass)
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

    if relative:
        y_title = "Log10(Relative Biomass)" if log_scale else "Relative Biomass"
        fig.add_hline(
            y=0.0 if log_scale else 1.0,
            line={"dash": "dash", "color": "gray", "width": 1},
            opacity=0.6,
        )
    else:
        y_title = "Log10(Biomass)" if log_scale else "Biomass"

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Year",
        yaxis_title=y_title,
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


# --- Analysis page -----------------------------------------------------------

_GOOD = "#2ca02c"
_BAD = "#d62728"


def _empty_figure(title: str, message: str) -> go.Figure:
    """A blank figure whose centre text explains why there is nothing to show."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
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


def build_mti_heatmap(
    mti, group_names: Sequence[str], title: str = "Mixed Trophic Impacts"
) -> go.Figure:
    """Heatmap of the MTI matrix (rows impact columns), centred on zero."""
    if mti is None or len(group_names) == 0:
        return _empty_figure(title, "Balance a model to see mixed trophic impacts")
    z = np.asarray(mti, dtype=float)
    names = [str(g) for g in group_names][: z.shape[0]]
    fig = go.Figure(
        go.Heatmap(
            z=z[: len(names), : len(names)],
            x=names,
            y=names,
            zmid=0,
            colorscale="RdBu",
            colorbar={"title": "Impact"},
            hovertemplate="%{y} → %{x}<br>Impact %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis={"title": "Impacted group", "tickangle": -45},
        yaxis={"title": "Impacting group", "autorange": "reversed"},
        margin={"l": 120, "r": 20, "t": 50, "b": 120},
    )
    return fig


def build_keystoneness_figure(
    groups: Sequence[str],
    biomass,
    keystoneness,
    title: str = "Keystoneness vs Biomass",
) -> go.Figure:
    """Keystoneness against log10 biomass; hover shows the group name."""
    b = np.asarray(biomass, dtype=float)
    k = np.asarray(keystoneness, dtype=float)
    n = min(len(groups), b.size, k.size)
    valid = np.zeros(n, dtype=bool)
    if n:
        valid = (b[:n] > 0) & np.isfinite(k[:n])
    if not valid.any():
        return _empty_figure(title, "No keystoneness values to plot")
    names = np.asarray([str(g) for g in groups[:n]])[valid]
    fig = go.Figure(
        go.Scatter(
            x=np.log10(b[:n][valid]),
            y=k[:n][valid],
            text=list(names),
            mode="markers",
            marker={"size": 12, "color": "#1f77b4", "opacity": 0.75},
            hovertemplate="%{text}<br>log10 B %{x:.2f}<br>Keystoneness %{y:.3f}"
            "<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Log10(Biomass)",
        yaxis_title="Keystoneness Index",
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return fig


def build_ee_figure(
    groups: Sequence[str], ee, title: str = "Ecotrophic Efficiency by Group"
) -> go.Figure:
    """Horizontal EE bars, red above 1, with a dashed line at EE = 1.

    Groups without an EE (detritus, fleets) are left out.
    """
    e = np.asarray(ee, dtype=float)
    n = min(len(groups), e.size)
    keep = np.isfinite(e[:n]) if n else np.zeros(0, dtype=bool)
    if not keep.any():
        return _empty_figure(title, "No EE values to plot")
    names = [str(g) for g, ok in zip(groups[:n], keep) if ok]
    values = e[:n][keep]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": [_BAD if v > 1 else _GOOD for v in values]},
            hovertemplate="%{y}<br>EE %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=1, line={"dash": "dash", "color": _BAD, "width": 2})
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Ecotrophic Efficiency (EE)",
        yaxis={"autorange": "reversed"},
        height=max(350, 22 * len(names) + 120),
        margin={"l": 140, "r": 20, "t": 50, "b": 50},
    )
    return fig


# --- Ecosim Summary tab ------------------------------------------------------


def build_initial_final_figure(
    groups: Sequence[str],
    initial,
    final,
    title: str = "Initial vs Final Biomass",
) -> go.Figure:
    """Grouped bars of biomass at the start and end of a simulation."""
    if len(groups) == 0:
        return _empty_figure(title, "Run simulation first")
    names = [str(g) for g in groups]
    fig = go.Figure(
        [
            go.Bar(
                x=names,
                y=list(np.asarray(initial, dtype=float)),
                name="Initial",
                marker_color="#1f77b4",
            ),
            go.Bar(
                x=names,
                y=list(np.asarray(final, dtype=float)),
                name="Final",
                marker_color=_GOOD,
            ),
        ]
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        barmode="group",
        xaxis={"title": "Group", "tickangle": -45},
        yaxis_title="Biomass",
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 50, "b": 100},
    )
    return fig


def build_biomass_change_figure(
    groups: Sequence[str],
    initial,
    final,
    title: str = "Biomass Change by Group",
) -> go.Figure:
    """Horizontal bars of percent biomass change, green up / red down.

    Groups that start at zero biomass show 0 % (no division warning).
    """
    if len(groups) == 0:
        return _empty_figure(title, "Run simulation first")
    start = np.asarray(initial, dtype=float)
    end = np.asarray(final, dtype=float)
    pct = np.zeros_like(start)
    ok = start > 0
    pct[ok] = (end[ok] - start[ok]) / start[ok] * 100.0
    names = [str(g) for g in groups]
    fig = go.Figure(
        go.Bar(
            x=list(pct),
            y=names,
            orientation="h",
            marker={"color": [_GOOD if v >= 0 else _BAD for v in pct]},
            hovertemplate="%{y}<br>%{x:+.1f} %<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": "gray", "width": 1})
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Percent Change (%)",
        yaxis={"autorange": "reversed"},
        margin={"l": 140, "r": 20, "t": 50, "b": 50},
    )
    return fig
