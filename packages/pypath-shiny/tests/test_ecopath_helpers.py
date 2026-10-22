"""Tests for ecopath.py private helpers."""

from pathlib import Path

import pytest

from pypath_shiny.pages.ecopath import (
    _convert_input_to_numeric,
    _get_groups_from_model,
    _recreate_params_from_model,
)


class TestConvertInputToNumeric:
    def test_int_string(self):
        assert _convert_input_to_numeric("42") == 42.0

    def test_float_string(self):
        assert _convert_input_to_numeric("3.14") == pytest.approx(3.14)

    def test_numeric_passthrough(self):
        assert _convert_input_to_numeric(7.5) == pytest.approx(7.5)

    def test_empty_string_returns_none_or_nan(self):
        result = _convert_input_to_numeric("")
        # Either None or NaN is acceptable for empty input
        import math

        assert result is None or (isinstance(result, float) and math.isnan(result))

    def test_non_numeric_string_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _convert_input_to_numeric("abc")


class TestGetGroupsFromModel:
    def test_rpath_params_returns_group_list(self, rpath_params):
        groups = _get_groups_from_model(rpath_params)
        assert groups == ["Fish", "Plankton", "Detritus"]

    def test_balanced_model_returns_group_list(self, balanced_rpath_model):
        groups = _get_groups_from_model(balanced_rpath_model)
        assert groups == ["Fish", "Plankton", "Detritus"]

    def test_invalid_object_raises_value_error(self):
        class Fake:
            pass

        with pytest.raises(ValueError):
            _get_groups_from_model(Fake())

    def test_returns_list_type(self, rpath_params):
        result = _get_groups_from_model(rpath_params)
        assert isinstance(result, list)


class TestRecreateParamsFromModel:
    def test_returns_rpath_params(self, balanced_rpath_model):
        from pypath.core.params import RpathParams

        recreated = _recreate_params_from_model(balanced_rpath_model)
        assert isinstance(recreated, RpathParams)

    def test_same_groups(self, balanced_rpath_model):
        recreated = _recreate_params_from_model(balanced_rpath_model)
        assert list(recreated.model["Group"]) == ["Fish", "Plankton", "Detritus"]

    def test_biomass_preserved(self, balanced_rpath_model):
        recreated = _recreate_params_from_model(balanced_rpath_model)
        assert recreated.model["Biomass"].iloc[0] == pytest.approx(10.0)

    def test_pb_preserved(self, balanced_rpath_model):
        recreated = _recreate_params_from_model(balanced_rpath_model)
        assert recreated.model["PB"].iloc[0] == pytest.approx(1.0)

    def test_types_preserved(self, balanced_rpath_model):
        recreated = _recreate_params_from_model(balanced_rpath_model)
        types = list(recreated.model["Type"])
        assert types[0] == pytest.approx(0)  # Fish = Consumer
        assert types[1] == pytest.approx(1)  # Plankton = Producer

    def test_diet_reconstructed(self, balanced_rpath_model):
        # Plankton (row 1) eaten by Fish: DC[1,0]=1.0 → diet.iloc[1,1]=1.0
        recreated = _recreate_params_from_model(balanced_rpath_model)
        assert recreated.diet.iloc[1, 1] == pytest.approx(1.0)


def test_ecopath_page_uses_notification_service_not_direct_ui_calls():
    """Guardrail: migrate ecopath page notifications to service wrappers."""
    from pypath_shiny.pages import ecopath

    source = Path(ecopath.__file__).read_text(encoding="utf-8")
    assert "ui.notification_show(" not in source


def test_ecopath_balance_task_alert_running_contains_progress():
    from pypath_shiny.pages.ecopath import ecopath_balance_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="e1",
        status=TaskStatus.RUNNING,
        message="Running pre-balance validation",
        progress=40.0,
    )
    alert_class, _icon, message = ecopath_balance_task_alert(state)

    assert alert_class == "alert alert-info"
    assert "40%" in message


def test_ecopath_balance_task_alert_error_prefers_error_message():
    from pypath_shiny.pages.ecopath import ecopath_balance_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="e2",
        status=TaskStatus.ERROR,
        message="failed",
        error="Input validation failed",
    )
    alert_class, _icon, message = ecopath_balance_task_alert(state)

    assert alert_class == "alert alert-danger"
    assert message == "Input validation failed"


def test_ecopath_balance_task_alert_success_uses_task_message():
    from pypath_shiny.pages.ecopath import ecopath_balance_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="e3",
        status=TaskStatus.SUCCESS,
        message="Model 'Demo' balanced successfully!",
    )
    alert_class, _icon, message = ecopath_balance_task_alert(state)

    assert alert_class == "alert alert-success"
    assert "Demo" in message


def test_collect_ecopath_validation_issues_reports_missing_model():
    from pypath_shiny.pages.ecopath import collect_ecopath_validation_issues

    issues = collect_ecopath_validation_issues(None)
    assert any(i.field == "model" for i in issues)


def test_collect_ecopath_validation_issues_reports_diet_sum_warning(rpath_params):
    from pypath_shiny.pages.ecopath import collect_ecopath_validation_issues
    from pypath_shiny.services.validation import ValidationSeverity

    params = rpath_params
    # break one predator diet column sum deliberately
    predator_cols = [c for c in params.diet.columns if c != "Group"]
    if predator_cols:
        col = predator_cols[0]
        params.diet[col] = 0.0

    issues = collect_ecopath_validation_issues(params)
    assert any(
        i.field == "diet" and i.severity is ValidationSeverity.WARNING for i in issues
    )


def test_summarize_ecopath_validation_issues_counts_errors():
    from pypath_shiny.pages.ecopath import (
        collect_ecopath_validation_issues,
        summarize_ecopath_validation_issues,
    )

    issues = collect_ecopath_validation_issues(None)
    summary = summarize_ecopath_validation_issues(issues)
    assert "error" in summary
