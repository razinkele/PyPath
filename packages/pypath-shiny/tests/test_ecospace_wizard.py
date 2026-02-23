"""Tests for the ecospace wizard page."""


def test_wizard_page_imports():
    from pypath_shiny.pages import ecospace_wizard

    assert hasattr(ecospace_wizard, "ecospace_wizard_ui")
    assert hasattr(ecospace_wizard, "ecospace_wizard_server")


def test_wizard_step_names():
    from pypath_shiny.pages.ecospace_wizard import _STEPS

    assert len(_STEPS) == 7
    assert _STEPS[0] == "Select Area"
    assert _STEPS[-1] == "Review & Launch"


def test_wizard_ui_renders():
    from pypath_shiny.pages.ecospace_wizard import ecospace_wizard_ui

    ui_result = ecospace_wizard_ui()
    assert ui_result is not None
