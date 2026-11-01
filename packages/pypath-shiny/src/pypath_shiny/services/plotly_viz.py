"""Plotly visualization helpers for PyPath Shiny pages."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

_TAB10 = [
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


def _matplotlib_palette(name: str, n: int) -> list[str] | None:
    """Colors from a matplotlib colormap (tab10, Paired, viridis, ...).

    Qualitative maps (few listed colors) are used in order; continuous maps
    are sampled evenly. Returns None if matplotlib has no such map.
    """
    try:
        from matplotlib import colormaps
        from matplotlib.colors import ListedColormap, to_hex
    except ImportError:  # pragma: no cover
        return None
    if name not in colormaps:
        return None
    cmap = colormaps[name]
    if isinstance(cmap, ListedColormap) and cmap.N <= 20:
        return [to_hex(c) for c in cmap.colors]
    return [to_hex(cmap(x)) for x in np.linspace(0, 1, max(n, 1))]


def _safe_palette(name: str | None, n: int) -> list[str]:
    """``n`` colors for Plotly traces from a Plotly or matplotlib palette name.

    The Results page offers matplotlib names (tab10, Set2, Paired, viridis);
    they used to fall back to Plotly's default colors. Short palettes cycle.
    """
    colors: list[str] | None = None
    if name in {None, "", "default"}:
        colors = list(_TAB10)
    else:
        try:
            import plotly.express as px

            options = getattr(px.colors.qualitative, name, None)
            if options is not None:
                colors = list(options)
        except ImportError:  # pragma: no cover
            pass
        if colors is None:
            colors = _matplotlib_palette(name, n)
    if not colors:
        colors = list(_TAB10)
    return [colors[i % len(colors)] for i in range(n)]


_PLOTLY_TEMPLATES = {
    "default": "plotly_white",
    "seaborn-v0_8": "seaborn",
    "ggplot": "ggplot2",
    "dark_background": "plotly_dark",
}


def plotly_template_for(style: str | None) -> str:
    """Plotly template for a Results-page "Plot Style" (matplotlib) name."""
    return _PLOTLY_TEMPLATES.get(style or "default", "plotly_white")


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


# --- Ecosim effort preview ---------------------------------------------------


def build_effort_figure(
    forced_effort,
    fleet_names: Sequence[str],
    palette: str | None = "tab10",
    title: str = "Fishing Effort Trajectory",
) -> go.Figure:
    """One line per fleet of the monthly effort multiplier.

    ``forced_effort`` is Ecosim's ``scenario.fishing.ForcedEffort``: one row
    per month, column 0 is the "Outside" slot and columns 1.. are the fleets.
    """
    if forced_effort is None:
        return _empty_figure(title, "Create scenario to preview effort")
    effort = np.asarray(forced_effort, dtype=float)
    n_fleets = effort.shape[1] - 1 if effort.ndim == 2 else 0
    if n_fleets <= 0:
        return _empty_figure(title, "No fishing fleets in model")
    names = [str(n) for n in fleet_names][:n_fleets]
    names += [f"Fleet {i + 1}" for i in range(len(names), n_fleets)]
    years = np.arange(effort.shape[0]) / 12
    colors = _safe_palette(palette, n_fleets)
    fig = go.Figure()
    for i, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=years,
                y=effort[:, i + 1],
                mode="lines",
                name=name,
                line={"width": 2, "color": colors[i]},
                hovertemplate="%{x:.2f} years<br>effort ×%{y:.2f}<extra>"
                "%{fullData.name}</extra>",
            )
        )
    fig.add_hline(y=1, line={"dash": "dash", "color": "gray", "width": 1})
    top = max(1.5, float(np.nanmax(effort[:, 1:])) * 1.1)
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Year",
        yaxis={"title": "Effort Multiplier", "range": [0, top]},
        legend_title="Fleet",
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return fig


# --- Results: trophic structure and food web ---------------------------------


def build_tl_bar_figure(
    groups: Sequence[str],
    trophic_levels,
    palette: str | None = "tab10",
    title: str = "Trophic Levels",
) -> go.Figure:
    """Horizontal trophic-level bars, highest TL on top, dashed line at TL 1."""
    tl = np.asarray(trophic_levels, dtype=float)
    if len(groups) == 0 or tl.size == 0:
        return _empty_figure(title, "Balance model first to see trophic levels")
    order = np.argsort(tl)[::-1]
    names = [str(groups[i]) for i in order]
    values = tl[order]
    fig = go.Figure(
        go.Bar(
            x=list(values),
            y=names,
            orientation="h",
            marker={"color": _safe_palette(palette, len(names))},
            hovertemplate="%{y}<br>TL %{x:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=1, line={"dash": "dash", "color": "gray", "width": 1})
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Trophic Level",
        yaxis={"autorange": "reversed"},
        margin={"l": 140, "r": 20, "t": 50, "b": 50},
    )
    return fig


_TL_BINS = [(1, 2, "TL 1-2"), (2, 3, "TL 2-3"), (3, 4, "TL 3-4"), (4, 5, "TL 4-5")]


def build_trophic_pyramid_figure(
    trophic_levels, production, title: str = "Trophic Pyramid (Production)"
) -> go.Figure:
    """Production summed by trophic-level band, TL 1-2 at the bottom.

    The last band is open-ended ("TL 5+"), so apex groups above TL 6 count.
    """
    tl = np.asarray(trophic_levels, dtype=float)
    prod = np.asarray(production, dtype=float)
    if tl.size == 0:
        return _empty_figure(title, "Balance model first to see trophic flows")
    totals = [float(np.nansum(prod[(tl >= lo) & (tl < hi)])) for lo, hi, _ in _TL_BINS]
    totals.append(float(np.nansum(prod[tl >= 5])))
    labels = [label for _, _, label in _TL_BINS] + ["TL 5+"]
    greens = ["#c7e9c0", "#a1d99b", "#74c476", "#31a354", "#006d2c"]
    fig = go.Figure(
        go.Bar(
            x=totals,
            y=labels,
            orientation="h",
            marker={"color": greens},
            hovertemplate="%{y}<br>Production %{x:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Production",
        margin={"l": 80, "r": 20, "t": 50, "b": 50},
    )
    return fig


def build_foodweb_figure(
    groups: Sequence[str],
    trophic_levels,
    biomass,
    diet_matrix,
    n_living: int,
    min_flow: float = 0.0,
    size_by_biomass: bool = True,
    width_by_flow: bool = True,
    title: str = "Food Web Structure",
) -> go.Figure:
    """Food web: groups at their trophic level, links prey -> predator.

    ``diet_matrix`` is Rpath's ``DC`` (prey rows x living-predator columns).
    Groups in the same whole-number TL band are spread across the x axis;
    y is the actual trophic level. Node sizes scale with sqrt(biomass) into
    12-45 px; links weaker than ``min_flow`` (diet share) are hidden.
    """
    names = [str(g) for g in groups]
    tl = np.asarray(trophic_levels, dtype=float)[: len(names)]
    bio = np.asarray(biomass, dtype=float)[: len(names)]
    dc = np.asarray(diet_matrix, dtype=float)
    if not names:
        return _empty_figure(title, "Balance model first to see food web")

    # x: spread within each whole-number TL band (as before), y: real TL
    x = np.zeros(len(names))
    bands: dict[int, list[int]] = {}
    for i, level in enumerate(tl):
        bands.setdefault(int(level) if np.isfinite(level) else 0, []).append(i)
    for members in bands.values():
        for k, i in enumerate(members):
            x[i] = (k + 1) / (len(members) + 1)

    links = []
    for pred in range(min(n_living, dc.shape[1] if dc.ndim == 2 else 0)):
        for prey in range(min(len(names), dc.shape[0])):
            share = dc[prey, pred]
            if np.isfinite(share) and share > min_flow and prey != pred:
                links.append((prey, pred, float(share)))
    if not links:
        return _empty_figure(title, "No diet connections found. Check diet matrix.")

    fig = go.Figure()
    for prey, pred, share in links:
        fig.add_trace(
            go.Scatter(
                x=[x[prey], x[pred]],
                y=[tl[prey], tl[pred]],
                mode="lines",
                line={
                    "width": 1 + 5 * share if width_by_flow else 1.5,
                    "color": "rgba(120,120,120,0.45)",
                },
                hoverinfo="text",
                text=f"{names[prey]} → {names[pred]}: {share:.0%} of diet",
                showlegend=False,
            )
        )

    if size_by_biomass:
        root = np.sqrt(np.clip(np.nan_to_num(bio), 0, None))
        span = root.max() - root.min()
        sizes = 12 + 33 * ((root - root.min()) / span if span > 0 else 0 * root)
    else:
        sizes = np.full(len(names), 20.0)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=tl,
            mode="markers+text",
            text=names,
            textposition="top center",
            marker={
                "size": list(sizes),
                "color": tl,
                "colorscale": "YlGnBu",
                "line": {"width": 1, "color": "#333"},
                "showscale": False,
            },
            customdata=np.stack([bio], axis=-1),
            hovertemplate="%{text}<br>TL %{y:.2f}<br>Biomass %{customdata[0]:.4g}"
            "<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis={"visible": False, "range": [-0.05, 1.05]},
        yaxis_title="Trophic Level",
        margin={"l": 60, "r": 20, "t": 50, "b": 30},
    )
    return fig
