"""Tests for the deck.gl (shiny-deckgl) Ecospace map builders."""

from types import SimpleNamespace

import numpy as np
import pytest

from pypath_shiny.services import deckgl_maps as d

gpd = pytest.importorskip("geopandas")
shapely_geometry = pytest.importorskip("shapely.geometry")
Polygon = shapely_geometry.Polygon
MultiPolygon = shapely_geometry.MultiPolygon


def _regular_grid():
    from pypath.spatial import create_regular_grid

    return create_regular_grid(bounds=(0, 0, 3, 2), nx=3, ny=2)


def _polygon_grid():
    """Two cells; the second is a MultiPolygon (cell plus an island)."""
    a = Polygon([(21, 55), (21.5, 55), (21.5, 55.5), (21, 55.5)])
    b = MultiPolygon(
        [
            Polygon([(21.5, 55), (22, 55), (22, 55.5), (21.5, 55.5)]),
            Polygon([(22.2, 55.8), (22.3, 55.8), (22.3, 55.9), (22.2, 55.9)]),
        ]
    )
    gdf = gpd.GeoDataFrame(geometry=[a, b], crs="EPSG:4326")
    return SimpleNamespace(
        geometry=gdf,
        n_patches=2,
        patch_centroids=np.array([[21.25, 55.25], [21.75, 55.25]]),
        patch_areas=np.array([1.5, 2.5]),
        adjacency_matrix=None,
    )


def _layer_types(layers):
    return [layer["type"] for layer in layers]


def test_values_to_colors_runs_along_the_palette_and_clips():
    palette = [[0, 0, 0], [200, 100, 50]]
    c = d.values_to_colors([-1.0, 0.0, 0.5, 1.0, 9.0], palette, 0.0, 1.0, alpha=255)
    assert c[0] == c[1] == [0, 0, 0, 255]
    assert c[2] == [100, 50, 25, 255]
    assert c[3] == c[4] == [200, 100, 50, 255]


def test_values_to_colors_survives_a_flat_range_and_nan():
    c = d.values_to_colors([float("nan"), 2.0], [[0, 0, 0], [255, 255, 255]], 2, 2)
    assert len(c) == 2 and all(len(x) == 4 for x in c)


def test_polygon_grid_is_a_geojson_layer_with_one_feature_per_part():
    g = _polygon_grid()
    layers = d.patch_value_layers(
        "habitat",
        g,
        [0.0, 1.0],
        palette=d.HABITAT_PALETTE,
        vmin=0,
        vmax=1,
        value_name="Habitat quality",
        labels=True,
    )
    assert _layer_types(layers) == ["GeoJsonLayer", "TextLayer"]
    features = layers[0]["data"]["features"]
    assert len(features) == 3  # MultiPolygon -> two parts
    assert [f["properties"]["patch_id"] for f in features] == [0, 1, 1]
    # Both parts of patch 1 share its colour; patch 0 differs
    assert features[1]["properties"]["fill"] == features[2]["properties"]["fill"]
    assert features[0]["properties"]["fill"] != features[1]["properties"]["fill"]
    assert layers[0]["getFillColor"] == "@@d.properties.fill"
    assert "Habitat quality: 1" in features[1]["properties"]["detail"]


def test_regular_grid_values_are_centroid_points():
    g = _regular_grid()
    layers = d.patch_value_layers(
        "biomass",
        g,
        np.arange(g.n_patches, dtype=float),
        palette=d.BIOMASS_PALETTE,
        vmin=0,
        vmax=g.n_patches - 1,
        value_name="Biomass",
    )
    assert _layer_types(layers) == ["ScatterplotLayer"]  # labels off by default
    data = layers[0]["data"]
    assert len(data) == g.n_patches
    assert data[0]["position"] == [0.5, 0.5]
    assert layers[0]["getFillColor"] == "@@d.fill"


def test_effort_marker_area_follows_effort():
    g = _regular_grid()
    effort = np.array([0.0, 1.0, 4.0, 4.0, 1.0, 0.0])
    (layer,) = d.effort_layers("effort", g, effort)
    radii = [p["radius"] for p in layer["data"]]
    # Area (radius squared) is proportional to effort; busiest patch is 12 px
    assert radii[2] == pytest.approx(12.0)
    assert radii[1] ** 2 / radii[2] ** 2 == pytest.approx(0.25)
    assert radii[0] == 0.0
    assert layer["radiusUnits"] == "pixels"


def test_effort_with_no_effort_does_not_divide_by_zero():
    (layer,) = d.effort_layers("effort", _regular_grid(), np.zeros(6))
    assert all(p["radius"] == 0.0 for p in layer["data"])


def test_regular_grid_overview_has_connections_points_and_labels():
    g = _regular_grid()
    layers = d.grid_overview_layers(g)
    assert [layer["id"] for layer in layers] == [
        "grid-connections",
        "grid-cells",
        "grid-labels",
    ]
    rows, cols = g.adjacency_matrix.nonzero()
    n_edges = int(sum(1 for i, j in zip(rows, cols) if i < j))
    assert len(layers[0]["data"]) == n_edges


def test_polygon_grid_overview_with_boundary():
    g = _polygon_grid()
    boundary = gpd.GeoDataFrame(
        geometry=[Polygon([(20.9, 54.9), (22.4, 54.9), (22.4, 56), (20.9, 56)])],
        crs="EPSG:4326",
    ).__geo_interface__
    layers = d.grid_overview_layers(g, boundary)
    assert [layer["id"] for layer in layers] == [
        "grid-cells",
        "grid-labels",
        "study-boundary",
    ]
    assert "Area: 2.50 km²" in layers[0]["data"]["features"][1]["properties"]["detail"]
    assert layers[2]["filled"] is False


def test_boundary_alone_is_drawn():
    boundary = {"type": "FeatureCollection", "features": []}
    assert [layer["id"] for layer in d.grid_overview_layers(None, boundary)] == [
        "study-boundary"
    ]


def test_grid_bounds_cover_grid_and_boundary():
    g = _polygon_grid()
    west_south, east_north = d.grid_bounds(g)
    assert west_south == [21.0, 55.0] and east_north == [22.3, 55.9]
    west_south, east_north = d.grid_bounds(g, [20.0, 54.0, 21.0, 55.0])
    assert west_south == [20.0, 54.0] and east_north == [22.3, 55.9]
    assert d.grid_bounds(None) is None


def test_grid_bounds_pad_a_transect():
    from pypath.spatial import create_1d_grid

    (w, s), (e, n) = d.grid_bounds(create_1d_grid(n_patches=4, spacing=1.0))
    assert e > w and n > s


def test_gradient_legend_lists_the_range():
    legend = d.gradient_legend("Biomass", d.BIOMASS_PALETTE, 0.0, 12.5, "biomass")
    (entry,) = legend["entries"]
    assert entry["shape"] == "gradient"
    assert entry["layer_id"] == "biomass"
    assert entry["label"] == "0 – 12.5"
    assert len(entry["colors"]) == 6
