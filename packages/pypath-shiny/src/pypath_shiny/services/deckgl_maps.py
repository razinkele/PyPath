"""deck.gl layer builders for the Ecospace maps (shiny-deckgl).

Everything here is pure: functions take an Ecospace grid (and values) and
return shiny-deckgl layer / widget dicts. The page pushes them with
``MapWidget.update``. Keeping the session out makes the builders testable.

Grids with polygon geometry (uploaded shapefiles, hexagons in a boundary)
are drawn as GeoJSON cells. Regular and transect grids have no geometry, so
their patches are drawn as points at the centroids, as the folium map did.
Coordinates are used as WGS 84 longitude/latitude for every grid.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from shiny_deckgl import (
    PALETTE_GREENS,
    PALETTE_REDS,
    PALETTE_YELLOW_RED,
    color_range,
    geojson_layer,
    layer_legend_widget,
    line_layer,
    scatterplot_layer,
    text_layer,
)

__all__ = [
    "HABITAT_PALETTE",
    "BIOMASS_PALETTE",
    "EFFORT_PALETTE",
    "values_to_colors",
    "patch_value_layers",
    "effort_layers",
    "grid_overview_layers",
    "gradient_legend",
    "grid_bounds",
    "MAP_TOOLTIP",
]

HABITAT_PALETTE = PALETTE_GREENS
BIOMASS_PALETTE = PALETTE_YELLOW_RED
EFFORT_PALETTE = PALETTE_REDS

# Tooltip template shared by the maps; fields missing from a feature render
# empty, so one template fits cells, points and the boundary.
# {label} and {detail} can carry group names from uploaded EwE files. They are
# safe here because shiny-deckgl's interpolateTemplate() HTML-escapes every
# interpolated value (and the legend sets title/labels with textContent). Do
# not pre-escape them in Python: that would show "&amp;" in the tooltip.
MAP_TOOLTIP = {
    "html": "<b>{label}</b><br/>{detail}",
    "style": {"backgroundColor": "white", "color": "black", "fontSize": "12px"},
}

_LIGHT_BLUE = [173, 216, 230, 110]
_STEEL_BLUE = [70, 130, 180, 255]
_BOUNDARY_RED = [220, 30, 30, 255]
_EDGE_GREY = [128, 128, 128, 80]
_LABEL_LIMIT = 200  # patches; above this, labels only clutter the map


def values_to_colors(
    values: Sequence[float],
    palette: list[list[int]],
    vmin: float,
    vmax: float,
    alpha: int = 220,
) -> list[list[int]]:
    """RGBA colours by linear interpolation along *palette*, clipped to [vmin, vmax]."""
    v = np.asarray(values, dtype=float)
    span = vmax - vmin if vmax > vmin else 1.0
    t = np.clip((v - vmin) / span, 0.0, 1.0)
    t = np.where(np.isfinite(t), t, 0.0)
    stops = np.asarray([c[:3] for c in palette], dtype=float)
    pos = np.linspace(0.0, 1.0, len(stops))
    rgb = np.column_stack([np.interp(t, pos, stops[:, k]) for k in range(3)])
    return [[int(round(r)), int(round(g)), int(round(b)), alpha] for r, g, b in rgb]


def _iter_parts(geom):
    """Polygon parts of a (Multi)Polygon; other geometry types are skipped."""
    if geom is None:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


def _cell_features(grid, props_for) -> dict:
    """GeoJSON FeatureCollection with one feature per polygon part.

    ``props_for(i)`` returns the properties of patch ``i`` (its position in the
    grid, which is how every per-patch array is indexed).
    """
    features = []
    for i, geom in enumerate(grid.geometry.geometry):
        props = props_for(i)
        for part in _iter_parts(geom):
            features.append(
                {
                    "type": "Feature",
                    "geometry": part.__geo_interface__,
                    "properties": props,
                }
            )
    return {"type": "FeatureCollection", "features": features}


def _centroid_points(grid, props_for) -> list[dict]:
    return [
        {"position": [float(x), float(y)], **props_for(i)}
        for i, (x, y) in enumerate(np.asarray(grid.patch_centroids, dtype=float))
    ]


def _label_layer(layer_id: str, grid, texts: Sequence[str]) -> dict:
    data = [
        {"position": [float(x), float(y)], "text": t}
        for (x, y), t in zip(np.asarray(grid.patch_centroids, dtype=float), texts)
    ]
    return text_layer(
        layer_id,
        data,
        getPosition="@@d.position",
        getText="@@d.text",
        getSize=11,
        getColor=[20, 20, 20, 255],
        fontWeight="bold",
        outlineWidth=2,
        outlineColor=[255, 255, 255, 255],
        fontSettings={"sdf": True},
        pickable=False,
    )


def _area(grid, i: int) -> float:
    areas = getattr(grid, "patch_areas", None)
    return float(areas[i]) if areas is not None else float("nan")


def patch_value_layers(
    layer_id: str,
    grid,
    values: Sequence[float],
    *,
    palette: list[list[int]],
    vmin: float,
    vmax: float,
    value_name: str,
    value_format: str = ".3g",
    labels: bool = False,
    outline: list[int] | None = None,
) -> list[dict]:
    """Patches coloured by *values* (habitat quality, biomass, ...)."""
    values = np.asarray(values, dtype=float)
    colors = values_to_colors(values, palette, vmin, vmax)

    def props(i):
        return {
            "patch_id": i,
            "label": f"Patch {i}",
            "detail": f"{value_name}: {values[i]:{value_format}}",
            "fill": colors[i],
        }

    if grid.geometry is not None:
        layers = [
            geojson_layer(
                layer_id,
                _cell_features(grid, props),
                # "@@=" expression form: the "@@d.x" shorthand reads a single
                # key, so "@@d.properties.fill" would be d["properties.fill"]
                # (undefined -> deck.gl draws the cells black).
                getFillColor="@@=d.properties.fill",
                getLineColor=outline or [90, 90, 90, 200],
                lineWidthMinPixels=1,
                stroked=True,
                filled=True,
                pickable=True,
                updateTriggers={"getFillColor": [vmin, vmax, len(values)]},
            )
        ]
    else:
        layers = [
            scatterplot_layer(
                layer_id,
                _centroid_points(grid, props),
                getPosition="@@d.position",
                getFillColor="@@d.fill",
                getLineColor=[0, 0, 0, 160],
                radiusUnits="pixels",
                getRadius=9,
                stroked=True,
                lineWidthMinPixels=1,
                pickable=True,
            )
        ]
    if labels and len(values) <= _LABEL_LIMIT:
        layers.append(
            _label_layer(f"{layer_id}-labels", grid, [f"{v:.2f}" for v in values])
        )
    return layers


def effort_layers(layer_id: str, grid, effort: Sequence[float]) -> list[dict]:
    """Fishing effort per patch: colour and marker area both follow effort."""
    effort = np.asarray(effort, dtype=float)
    peak = float(effort.max()) if effort.size else 0.0
    colors = values_to_colors(effort, EFFORT_PALETTE, 0.0, peak if peak > 0 else 1.0)
    # Radius in pixels proportional to sqrt(effort), so marker *area* follows
    # effort (as the matplotlib scatter(s=...) did); the busiest patch is 12 px.
    radii = 12.0 * np.sqrt(effort / peak) if peak > 0 else np.zeros_like(effort)
    data = [
        {
            "position": [float(x), float(y)],
            "patch_id": i,
            "label": f"Patch {i}",
            "detail": f"Effort: {effort[i]:.3g}",
            "fill": colors[i],
            "radius": float(radii[i]),
        }
        for i, (x, y) in enumerate(np.asarray(grid.patch_centroids, dtype=float))
    ]
    return [
        scatterplot_layer(
            layer_id,
            data,
            getPosition="@@d.position",
            getFillColor="@@d.fill",
            getRadius="@@d.radius",
            getLineColor=[0, 0, 0, 160],
            radiusUnits="pixels",
            radiusMinPixels=2,
            stroked=True,
            lineWidthMinPixels=1,
            pickable=True,
        )
    ]


def grid_overview_layers(grid, boundary: dict | None = None) -> list[dict]:
    """The grid itself: cells (or centroids + connections) and the study boundary.

    *boundary* is a GeoJSON dict (``GeoDataFrame.__geo_interface__``).
    """
    layers: list[dict] = []
    if grid is not None:
        n = grid.n_patches

        def props(i):
            return {
                "patch_id": i,
                "label": f"Patch {i}",
                "detail": f"Area: {_area(grid, i):.2f} km²",
            }

        if grid.geometry is not None:
            layers.append(
                geojson_layer(
                    "grid-cells",
                    _cell_features(grid, props),
                    getFillColor=_LIGHT_BLUE,
                    getLineColor=_STEEL_BLUE,
                    lineWidthMinPixels=1,
                    stroked=True,
                    filled=True,
                    pickable=True,
                )
            )
        else:
            adjacency = getattr(grid, "adjacency_matrix", None)
            if adjacency is not None:
                rows, cols = adjacency.nonzero()
                c = np.asarray(grid.patch_centroids, dtype=float)
                edges = [
                    {
                        "sourcePosition": [float(c[i, 0]), float(c[i, 1])],
                        "targetPosition": [float(c[j, 0]), float(c[j, 1])],
                    }
                    for i, j in zip(rows, cols)
                    if i < j
                ]
                layers.append(
                    line_layer(
                        "grid-connections",
                        edges,
                        getSourcePosition="@@d.sourcePosition",
                        getTargetPosition="@@d.targetPosition",
                        getColor=_EDGE_GREY,
                        getWidth=1,
                        widthUnits="pixels",
                        pickable=False,
                    )
                )
            layers.append(
                scatterplot_layer(
                    "grid-cells",
                    _centroid_points(grid, props),
                    getPosition="@@d.position",
                    getFillColor=_STEEL_BLUE,
                    radiusUnits="pixels",
                    getRadius=7,
                    pickable=True,
                )
            )
        if n <= _LABEL_LIMIT:
            layers.append(_label_layer("grid-labels", grid, [str(i) for i in range(n)]))
    if boundary is not None:
        layers.append(
            geojson_layer(
                "study-boundary",
                boundary,
                filled=False,
                stroked=True,
                getLineColor=_BOUNDARY_RED,
                lineWidthMinPixels=2,
                pickable=False,
            )
        )
    return layers


def gradient_legend(
    title: str, palette: list[list[int]], vmin: float, vmax: float, layer_id: str
) -> dict:
    """A deck.gl legend with a colour ramp for one value layer."""
    return layer_legend_widget(
        entries=[
            {
                "layer_id": layer_id,
                "label": f"{vmin:.3g} – {vmax:.3g}",
                "shape": "gradient",
                "colors": color_range(6, palette),
            }
        ],
        placement="bottom-left",
        title=title,
        show_checkbox=False,
    )


def grid_bounds(grid, boundary_bounds: Sequence[float] | None = None):
    """``[[west, south], [east, north]]`` around the grid and the boundary."""
    boxes = []
    if grid is not None:
        if grid.geometry is not None:
            boxes.append(np.asarray(grid.geometry.total_bounds, dtype=float))
        else:
            c = np.asarray(grid.patch_centroids, dtype=float)
            boxes.append(
                np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()])
            )
    if boundary_bounds is not None:
        boxes.append(np.asarray(boundary_bounds, dtype=float))
    if not boxes:
        return None
    b = np.vstack(boxes)
    west, south = b[:, 0].min(), b[:, 1].min()
    east, north = b[:, 2].max(), b[:, 3].max()
    # A single point or a transect would give a zero-area box
    pad = 0.01 if (east - west) < 1e-9 or (north - south) < 1e-9 else 0.0
    return [
        [float(west - pad), float(south - pad)],
        [float(east + pad), float(north + pad)],
    ]
