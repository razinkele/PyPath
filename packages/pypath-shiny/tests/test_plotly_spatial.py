"""Tests for the Ecospace Plotly maps."""

import numpy as np

from pypath_shiny.services.plotly_spatial import (
    build_effort_map_figure,
    build_patch_map_figure,
)

CENTROIDS = np.array([[0.5, 0.5], [1.5, 0.5]])
SQUARE_A = ([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
SQUARE_B = ([1, 2, 2, 1, 1], [0, 0, 1, 1, 0])


def _kw(**over):
    kw = dict(
        title="Map",
        colorbar_title="Value",
        colorscale="YlGn",
        vmin=0.0,
        vmax=1.0,
    )
    kw.update(over)
    return kw


def test_polygon_map_draws_one_filled_outline_per_ring_plus_a_colorbar():
    # Patch 1 is a MultiPolygon with two parts
    polygons = [[SQUARE_A], [SQUARE_B, SQUARE_B]]
    fig = build_patch_map_figure([0.0, 1.0], CENTROIDS, polygons, **_kw())
    fills = [t for t in fig.data if t.fill == "toself"]
    assert len(fills) == 3
    # Low value gets the light end of YlGn, high value the dark end
    assert fills[0].fillcolor != fills[1].fillcolor
    assert fills[1].fillcolor == fills[2].fillcolor
    colorbars = [t for t in fig.data if t.marker and t.marker.showscale]
    assert len(colorbars) == 1
    assert colorbars[0].marker.colorbar.title.text == "Value"


def test_polygon_map_values_outside_the_range_are_clipped():
    fig = build_patch_map_figure(
        [-5.0, 9.0], CENTROIDS, [[SQUARE_A], [SQUARE_B]], **_kw()
    )
    fills = [t.fillcolor for t in fig.data if t.fill == "toself"]
    lo = build_patch_map_figure(
        [0.0, 1.0], CENTROIDS, [[SQUARE_A], [SQUARE_B]], **_kw()
    )
    assert fills == [t.fillcolor for t in lo.data if t.fill == "toself"]


def test_regular_grid_uses_centroid_markers_with_a_colorbar():
    fig = build_patch_map_figure([0.2, 0.8], CENTROIDS, None, **_kw())
    assert len(fig.data) == 1
    marker = fig.data[0].marker
    assert list(marker.color) == [0.2, 0.8]
    assert marker.showscale


def test_labels_are_added_as_a_text_trace():
    fig = build_patch_map_figure(
        [0.25, 0.75], CENTROIDS, [[SQUARE_A], [SQUARE_B]], **_kw(label_values=True)
    )
    text = [t for t in fig.data if t.mode == "text"]
    assert len(text) == 1
    assert list(text[0].text) == ["0.25", "0.75"]


def test_maps_keep_an_equal_aspect():
    fig = build_patch_map_figure([0.2, 0.8], CENTROIDS, None, **_kw())
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


def test_effort_marker_area_is_proportional_to_effort():
    fig = build_effort_map_figure([10.0, 40.0], CENTROIDS, "gravity")
    marker = fig.data[0].marker
    assert marker.sizemode == "area"
    assert list(marker.size) == [10.0, 40.0]
    assert "gravity" in fig.layout.title.text


def test_effort_map_with_no_effort_does_not_divide_by_zero():
    fig = build_effort_map_figure([0.0, 0.0], CENTROIDS, "uniform")
    assert fig.data[0].marker.sizeref == 1.0


def test_effort_colour_range_starts_at_zero_even_when_effort_is_uniform():
    """Only cmin left Plotly autoscaling: uniform effort showed a 3.5-4.5 bar."""
    marker = build_effort_map_figure([4.0, 4.0], CENTROIDS, "uniform").data[0].marker
    assert marker.cmin == 0
    assert marker.cmax == 4.0
