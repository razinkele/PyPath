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


def test_wizard_all_step_uis_render():
    """Test that all step UI functions return non-None."""
    from pypath_shiny.pages.ecospace_wizard import (
        _step1_select_area_ui,
        _step2_configure_grid_ui,
        _step3_download_data_ui,
        _step4_review_habitats_ui,
        _step5_assign_preferences_ui,
        _step6_set_dispersal_ui,
        _step7_review_launch_ui,
    )

    assert _step1_select_area_ui() is not None
    assert _step2_configure_grid_ui() is not None
    assert _step3_download_data_ui() is not None
    assert _step4_review_habitats_ui() is not None
    assert _step5_assign_preferences_ui() is not None
    assert _step6_set_dispersal_ui() is not None
    assert _step7_review_launch_ui() is not None


def test_wizard_server_function_exists():
    """Test that the server function is callable."""
    from pypath_shiny.pages.ecospace_wizard import ecospace_wizard_server

    assert callable(ecospace_wizard_server)
