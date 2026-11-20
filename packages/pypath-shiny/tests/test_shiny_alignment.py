"""The app is written for the Shiny and shiny-deckgl it declares.

Shiny 1.9 (ARIA landmark sidebars, concurrent effects, uvicorn >= 0.50) and
shiny-deckgl 1.16 are the floors; these tests keep the declared floors, the
deploy's pinned tag and the code's use of the newer API in step.
"""

import ast
import re
import tomllib
from pathlib import Path

import pytest
from shiny import ui

import pypath_shiny
from pypath_shiny import app as app_module

SRC = Path(pypath_shiny.__file__).resolve().parent
REPO = SRC.parents[3]
PAGES = sorted((SRC / "pages").glob("*.py"))

FLOORS = {"shiny": "1.9.0", "shiny-deckgl": "1.16.1", "uvicorn": "0.50"}


def _floor(requirements, name):
    pattern = re.compile(rf"^{re.escape(name)}\s*>=\s*([\d.]+)")
    for req in requirements:
        match = pattern.match(req.strip().lstrip("- ").strip())
        if match:
            return match.group(1)
    raise AssertionError(f"{name} is not required")


def test_the_package_requires_the_versions_it_is_written_for():
    with (SRC.parents[1] / "pyproject.toml").open("rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    assert {name: _floor(deps, name) for name in FLOORS} == FLOORS


@pytest.mark.skipif(
    not (REPO / "conda-recipes").exists(), reason="not run from the repository"
)
def test_the_conda_recipe_and_the_workflows_agree_with_the_package():
    recipe = (REPO / "conda-recipes/pypath-shiny/meta.yaml").read_text().splitlines()
    assert {name: _floor(recipe, name) for name in FLOORS} == FLOORS
    deploy = (REPO / ".github/workflows/deploy.yml").read_text()
    assert f"shiny_deckgl@v{FLOORS['shiny-deckgl']}'" in deploy
    smoke = (REPO / ".github/workflows/ci-shiny-smoke.yml").read_text()
    assert f"shiny-deckgl>={FLOORS['shiny-deckgl']}" in smoke


def _calls(tree, attr):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
        ):
            yield node


def test_every_sidebar_is_a_named_landmark():
    """Screen readers can jump between a page's settings and its content."""
    found = 0
    for path in PAGES:
        for call in _calls(ast.parse(path.read_text(encoding="utf-8")), "sidebar"):
            found += 1
            kwargs = {k.arg: k.value for k in call.keywords}
            where = f"{path.name}:{call.lineno}"
            assert ast.literal_eval(kwargs["role"]) == "complementary", where
            assert ast.literal_eval(kwargs["aria_label"]).endswith(" settings"), where
    assert found == 11


def test_the_built_page_has_the_landmarks():
    html = str(ui.TagList(app_module.app_ui).tagify())
    labels = re.findall(r'<aside[^>]*aria-label="([^"]+)"', html)
    assert len(labels) == 11 and len(set(labels)) == 11


def test_no_page_uses_the_legacy_output_decorator():
    """``@render.*`` registers an output by itself; ``@output`` is the old form."""
    for path in [*PAGES, SRC / "app.py"]:
        lines = path.read_text(encoding="utf-8").splitlines()
        legacy = [i + 1 for i, line in enumerate(lines) if line.strip() == "@output"]
        assert legacy == [], f"{path.name}: {legacy[:5]}"


def test_the_stylesheet_no_longer_works_around_the_draw_control():
    """shiny-deckgl 1.16.1 gives mapbox-gl-draw MapLibre's class names itself.

    Before that the draw buttons could not be clicked, and the stylesheet
    carried rules for ``.mapboxgl-ctrl``; they match nothing now.
    """
    css = (SRC / "static" / "custom.css").read_text(encoding="utf-8")
    assert "mapboxgl-ctrl" not in css
