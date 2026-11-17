"""Prey switching is described the same way across the app (module source)."""

import re
from pathlib import Path

import pytest

import pypath_shiny.pages as pages_pkg

PAGES = Path(pages_pkg.__file__).parent
SRC = PAGES.parent


def _text(name: str) -> str:
    """A module's source with adjacent string literals joined, whitespace collapsed."""
    src = (PAGES / name).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", re.sub(r'"\s+"', "", src))


@pytest.mark.parametrize(
    "name", ["home.py", "about.py", "forcing_demo.py", "ecosim.py"]
)
def test_no_planned_or_rewiring_wording(name):
    text = _text(name)
    assert not re.search(r"(?i)prey switching[^.]{0,60}planned", text)
    assert "diet rewiring (prey switching)" not in text.lower()


@pytest.mark.parametrize("name", ["home.py", "about.py"])
def test_feature_line(name):
    assert "Prey switching (EwE's switching power)" in _text(name)


def test_home_card():
    text = _text("home.py")
    assert '"Prey Switching"' in text
    assert (
        "Predators turn towards prey that become more abundant: EwE's switching "
        "power, set per predator on the Ecosim page." in text
    )
    assert (
        "The Prey Switching demo page shows how the switching power shapes a "
        "predator's diet." in text
    )


def test_autofix_modal_dd_is_handling_time():
    text = _text("ecosim.py")
    assert "DD (handling time)" in text
    assert "DD (Prey Switching)" not in text


def test_nothing_calls_dd_prey_switching():
    for path in SRC.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not re.search(r"(?i)DD.*switch", line), (path, line)


def test_forcing_demo_points_to_per_predator_switching():
    text = _text("forcing_demo.py")
    assert "Diet rewiring is deprecated; prey switching is set per predator" in text
    assert text.count("scenario.params.SwitchPower") >= 2


def test_nav_label():
    app = (SRC / "app.py").read_text(encoding="utf-8")
    assert '_icon_label("bi-arrow-repeat", "Prey Switching")' in app
    assert 'value="Diet Rewiring"' in app
