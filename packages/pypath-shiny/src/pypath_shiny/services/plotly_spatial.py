"""Plotly maps for the Ecospace page.

Plotly has no "polygon patch" primitive outside its geo subplots, so each grid
cell is drawn as a filled ``go.Scatter`` outline coloured by sampling a named
colorscale. An invisible marker trace carries the colour bar.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# Each polygon is an (xs, ys) pair: the exterior ring of one grid cell part.
PolygonRing = tuple[Sequence[float], Sequence[float]]


def _colorbar_trace(colorscale: str, cmin: float, cmax: float, title: str):
    """An invisible trace whose only job is to show the colour bar."""
    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker={
            "colorscale": colorscale,
            "cmin": cmin,
            "cmax": cmax,
            "color": [cmin],
            "showscale": True,
            "colorbar": {"title": {"text": title, "side": "right"}},
        },
        hoverinfo="skip",
        showlegend=False,
    )


def _map_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> None:
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis={"title": x_title, "showgrid": True},
        # Equal aspect, like matplotlib's set_aspect("equal"), so cells keep
        # their shape.
        yaxis={"title": y_title, "scaleanchor": "x", "scaleratio": 1},
        showlegend=False,
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )


def build_patch_map_figure(
    values: Sequence[float],
    centroids: np.ndarray,
    polygons: Sequence[Iterable[PolygonRing]] | None,
    *,
    title: str,
    colorbar_title: str,
    colorscale: str,
    vmin: float,
    vmax: float,
    outline_color: str = "gray",
    label_values: bool = False,
    value_format: str = ".3g",
) -> go.Figure:
    """Colour each patch by its value.

    ``polygons[i]`` holds the rings of patch ``i`` (several for a MultiPolygon).
    Without polygons (a regular grid), patches are drawn as markers at their
    centroids.
    """
    values = np.asarray(values, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    span = vmax - vmin if vmax > vmin else 1.0
    fig = go.Figure()

    if polygons is not None:
        norm = np.clip((values - vmin) / span, 0.0, 1.0)
        colors = sample_colorscale(colorscale, list(norm))
        for i, rings in enumerate(polygons):
            for xs, ys in rings:
                fig.add_trace(
                    go.Scatter(
                        x=list(xs),
                        y=list(ys),
                        mode="lines",
                        fill="toself",
                        fillcolor=colors[i],
                        line={"color": outline_color, "width": 1},
                        opacity=0.9,
                        hoveron="fills",
                        hoverinfo="text",
                        text=f"Patch {i}: {values[i]:{value_format}}",
                    )
                )
        fig.add_trace(_colorbar_trace(colorscale, vmin, vmax, colorbar_title))
    else:
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                marker={
                    "color": values,
                    "colorscale": colorscale,
                    "cmin": vmin,
                    "cmax": vmax,
                    "size": 14,
                    "line": {"color": "black", "width": 0.5},
                    "showscale": True,
                    "colorbar": {"title": {"text": colorbar_title, "side": "right"}},
                },
                customdata=np.arange(len(values)),
                hovertemplate="Patch %{customdata}: %{marker.color:"
                + value_format
                + "}<extra></extra>",
            )
        )

    if label_values and len(values):
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="text",
                text=[f"{v:.2f}" for v in values],
                textfont={"size": 10, "color": "black"},
                hoverinfo="skip",
            )
        )

    _map_layout(fig, title, "Longitude (degrees)", "Latitude (degrees)")
    return fig


def build_effort_map_figure(
    effort: Sequence[float], centroids: np.ndarray, method: str
) -> go.Figure:
    """Fishing effort per patch: colour and marker area both scale with effort."""
    effort = np.asarray(effort, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    peak = float(effort.max()) if effort.size else 0.0
    # sizemode="area" keeps marker *area* proportional to effort, as matplotlib's
    # scatter(s=...) did; sizeref makes the busiest patch about 30 px across.
    sizeref = 2.0 * peak / (30.0**2) if peak > 0 else 1.0
    fig = go.Figure(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers",
            marker={
                "color": effort,
                "colorscale": "Reds",
                "cmin": 0,
                "size": effort,
                "sizemode": "area",
                "sizeref": sizeref,
                "sizemin": 2,
                "opacity": 0.75,
                "line": {"color": "black", "width": 0.5},
                "showscale": True,
                "colorbar": {"title": {"text": "Fishing Effort", "side": "right"}},
            },
            customdata=np.arange(len(effort)),
            hovertemplate="Patch %{customdata}: effort %{marker.color:.3g}"
            "<extra></extra>",
        )
    )
    _map_layout(
        fig,
        f"Spatial Fishing Effort ({method})",
        "X (degrees longitude)",
        "Y (degrees latitude)",
    )
    return fig
