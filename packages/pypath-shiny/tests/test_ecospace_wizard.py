from pathlib import Path

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


def test_ecospace_wizard_page_uses_notification_service_not_direct_ui_calls():
    """Guardrail: migrate ecospace wizard notifications to service wrappers."""
    from pypath_shiny.pages import ecospace_wizard

    source = Path(ecospace_wizard.__file__).read_text(encoding="utf-8")
    assert "ui.notification_show(" not in source


def test_wizard_draft_round_trip_includes_versioned_payload():
    from pypath_shiny.pages.ecospace_wizard import (
        deserialize_wizard_draft_payload,
        serialize_wizard_draft_payload,
    )

    payload = serialize_wizard_draft_payload(
        step=4,
        polygon={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 1], [0, 0]]]},
        grid_type="hexagonal",
        cell_size_km=7.5,
        dispersal_default=12.0,
        gravity_strength=0.45,
    )
    parsed = deserialize_wizard_draft_payload(payload)

    assert parsed["draft_version"] == 1
    assert parsed["step"] == 4
    assert parsed["grid_type"] == "hexagonal"
    assert parsed["cell_size_km"] == 7.5
    assert parsed["dispersal_default"] == 12.0
    assert parsed["gravity_strength"] == 0.45


def test_wizard_draft_parser_rejects_non_object_json():
    import pytest

    from pypath_shiny.pages.ecospace_wizard import deserialize_wizard_draft_payload

    with pytest.raises(ValueError):
        deserialize_wizard_draft_payload('["not-an-object"]')


def test_collect_wizard_validation_issues_detects_missing_grid_and_habitats():
    from pypath_shiny.pages.ecospace_wizard import collect_wizard_validation_issues

    issues = collect_wizard_validation_issues(
        grid=None,
        habitat_types=None,
        preference_matrix=None,
        group_names=["Cod", "Herring"],
        salinity_layer=None,
    )

    fields = [i.field for i in issues]
    assert "grid" in fields
    assert "habitats" in fields
    assert "preferences" in fields


def test_collect_wizard_validation_issues_flags_preference_row_mismatch():
    import numpy as np

    from pypath_shiny.pages.ecospace_wizard import collect_wizard_validation_issues

    class Grid:
        n_patches = 3

    issues = collect_wizard_validation_issues(
        grid=Grid(),
        habitat_types=np.array(["A", "A", "B"]),
        preference_matrix=np.ones((1, 2)),
        group_names=["Cod", "Herring"],
        salinity_layer=object(),
    )

    messages = [i.message for i in issues]
    assert any("rows do not match" in m for m in messages)


def test_summarize_wizard_validation_issues_includes_error_counts():
    from pypath_shiny.pages.ecospace_wizard import (
        collect_wizard_validation_issues,
        summarize_wizard_validation_issues,
    )

    issues = collect_wizard_validation_issues(
        grid=None,
        habitat_types=None,
        preference_matrix=None,
        group_names=["Cod"],
        salinity_layer=None,
    )
    summary = summarize_wizard_validation_issues(issues)

    assert "error" in summary


def test_wizard_download_task_alert_running_contains_progress():
    from pypath_shiny.pages.ecospace_wizard import wizard_download_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="w1",
        status=TaskStatus.RUNNING,
        message="Downloading habitats",
        progress=20.0,
    )
    css_class, _icon, message = wizard_download_task_alert(state)

    assert css_class == "text-info mt-2"
    assert "20%" in message


def test_wizard_download_task_alert_error_uses_error_field():
    from pypath_shiny.pages.ecospace_wizard import wizard_download_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="w2",
        status=TaskStatus.ERROR,
        message="Download failed",
        error="Timeout from EMODnet",
    )
    css_class, _icon, message = wizard_download_task_alert(state)

    assert css_class == "text-danger mt-2"
    assert "Timeout from EMODnet" in message


def test_wizard_download_task_alert_success_prefers_message():
    from pypath_shiny.pages.ecospace_wizard import wizard_download_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="w3",
        status=TaskStatus.SUCCESS,
        message="Done! 5 habitat features",
    )
    css_class, _icon, message = wizard_download_task_alert(state)

    assert css_class == "text-success mt-2"
    assert message == "Done! 5 habitat features"
