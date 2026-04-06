"""Tests for demo page modules — UI renders and server signatures."""

import inspect

from pypath_shiny.pages.diet_rewiring_demo import (
    diet_rewiring_demo_server,
    diet_rewiring_demo_ui,
)
from pypath_shiny.pages.forcing_demo import forcing_demo_server, forcing_demo_ui
from pypath_shiny.pages.optimization_demo import (
    optimization_demo_server,
    optimization_demo_ui,
)


def test_forcing_demo_ui_renders():
    assert forcing_demo_ui() is not None


def test_forcing_demo_server_exact_signature():
    p = list(inspect.signature(forcing_demo_server).parameters)
    assert p == ["input", "output", "session"]


def test_diet_rewiring_demo_ui_renders():
    assert diet_rewiring_demo_ui() is not None


def test_diet_rewiring_demo_server_exact_signature():
    p = list(inspect.signature(diet_rewiring_demo_server).parameters)
    assert p == ["input", "output", "session"]


def test_optimization_demo_ui_renders():
    assert optimization_demo_ui() is not None


def test_optimization_demo_server_exact_signature():
    p = list(inspect.signature(optimization_demo_server).parameters)
    assert p == ["input", "output", "session"]
