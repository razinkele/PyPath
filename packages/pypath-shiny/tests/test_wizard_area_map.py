"""The Ecospace wizard's study-area map is a shiny-deckgl map with its
drawing tools; the helpers that read what was drawn are plain functions."""

import ast
from pathlib import Path

import pytest
from shiny import ui

from pypath_shiny.pages import ecospace_wizard as wizard
from pypath_shiny.services.deckgl_maps import (
    drawn_polygon_geometry,
    polygon_area_km2,
    study_area_layers,
)

SQUARE = {
    "type": "Polygon",
    "coordinates": [
        [[20.0, 55.0], [21.0, 55.0], [21.0, 56.0], [20.0, 56.0], [20.0, 55.0]]
    ],
}


def _feature(geometry):
    return {"type": "Feature", "id": "a", "properties": {}, "geometry": geometry}


def _collection(*geometries):
    return {"type": "FeatureCollection", "features": [_feature(g) for g in geometries]}


@pytest.mark.parametrize(
    "drawn",
    [None, {}, _collection(), _collection({"type": "Point", "coordinates": [20, 55]})],
)
def test_nothing_drawn_gives_no_polygon(drawn):
    assert drawn_polygon_geometry(drawn) is None


def test_the_drawn_polygon_is_its_geometry():
    assert drawn_polygon_geometry(_collection(SQUARE)) == SQUARE


def test_the_last_polygon_drawn_is_the_study_area():
    other = {**SQUARE, "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    assert drawn_polygon_geometry(_collection(other, SQUARE)) == SQUARE


def test_a_polygon_with_too_few_points_is_not_a_study_area():
    line = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [0, 0]]]}
    assert drawn_polygon_geometry(_collection(line)) is None


def test_area_of_a_one_degree_cell_at_55_north():
    # 1 deg x 1 deg between 55 N and 56 N is about 7,030 km2 on the sphere
    assert polygon_area_km2(SQUARE) == pytest.approx(7030, rel=0.01)


def test_area_does_not_depend_on_the_ring_direction():
    reverse = {**SQUARE, "coordinates": [SQUARE["coordinates"][0][::-1]]}
    assert polygon_area_km2(reverse) == pytest.approx(polygon_area_km2(SQUARE))


def test_area_across_the_antimeridian():
    """179 E to 179 W is 2 degrees wide, not 358."""
    across = {
        "type": "Polygon",
        "coordinates": [
            [
                [179.0, 55.0],
                [-179.0, 55.0],
                [-179.0, 56.0],
                [179.0, 56.0],
                [179.0, 55.0],
            ]
        ],
    }
    assert polygon_area_km2(across) == pytest.approx(2 * polygon_area_km2(SQUARE))


def test_the_area_can_be_cleared_with_a_button_whatever_its_origin():
    """The map's bin only removes a drawn polygon, not a loaded draft's outline."""
    html = str(ui.TagList(wizard._step1_select_area_ui()).tagify())
    assert 'id="wizard_clear_area"' in html
    source = Path(wizard.__file__).read_text(encoding="utf-8")
    handler = source[source.index("def _clear_area") :][:900]
    for needed in (
        "drawn_polygon.set(None)",
        "delete_drawn_features(session)",
        "AREA_MAP.update(session, [])",
    ):
        assert needed in handler, needed


def test_study_area_layers_draw_the_polygon_or_nothing():
    assert study_area_layers(None) == []
    (layer,) = study_area_layers(SQUARE)
    assert layer["id"] == "study-area"


def test_step_one_is_a_deckgl_map_without_leaflet():
    html = str(ui.TagList(wizard._step1_select_area_ui()).tagify())
    assert 'id="wizard_area_map"' in html and "deckgl-map" in html
    assert "leaflet" not in html.lower() and "unpkg.com" not in html


def test_the_wizard_page_has_no_inline_script():
    source = Path(wizard.__file__).read_text(encoding="utf-8")
    assert "leaflet" not in source.lower()
    assert "Shiny.setInputValue" not in source
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "AREA_MAP" in names


def test_the_map_is_in_the_page_layout_not_in_the_step_content():
    """A map built with the step would miss the server's drawing message."""
    html = str(ui.TagList(wizard.ecospace_wizard_ui()).tagify())
    assert html.count('id="wizard_area_map"') == 1
    assert 'data-display-if="input.wizard_on_step1"' in html
