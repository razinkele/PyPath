"""Tests for pages/ecospace.py — UI renders, signature, and notification guardrail."""

import inspect
from pathlib import Path

from pypath_shiny.pages.ecospace import ecospace_server, ecospace_ui


def test_ecospace_ui_renders():
    assert ecospace_ui() is not None


def test_ecospace_server_signature_contains_expected_params():
    params = list(inspect.signature(ecospace_server).parameters)
    assert "input" in params and any(p.lstrip("_") == "output" for p in params)
    assert any(p.lstrip("_") == "session" for p in params)
    assert any(p.lstrip("_") == "model_data" for p in params)
    assert any(p.lstrip("_") == "sim_results" for p in params)


def test_ecospace_page_uses_notification_service_not_direct_ui_calls():
    from pypath_shiny.pages import ecospace

    source = Path(ecospace.__file__).read_text(encoding="utf-8")
    assert "ui.notification_show(" not in source


def test_spatial_task_alert_running_contains_progress():
    from pypath_shiny.pages.ecospace import spatial_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="s1",
        status=TaskStatus.RUNNING,
        message="Running spatial integration",
        progress=70.0,
    )
    alert_class, _icon, message = spatial_task_alert(state)

    assert alert_class == "alert alert-info mt-2"
    assert "70%" in message


def test_spatial_task_alert_error_prefers_error_field():
    from pypath_shiny.pages.ecospace import spatial_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="s2",
        status=TaskStatus.ERROR,
        message="failed",
        error="Spatial simulation failed: numerical issue",
    )
    alert_class, _icon, message = spatial_task_alert(state)

    assert alert_class == "alert alert-danger mt-2"
    assert "numerical issue" in message


def test_spatial_task_alert_success_uses_message():
    from pypath_shiny.pages.ecospace import spatial_task_alert
    from pypath_shiny.services.tasks import TaskState, TaskStatus

    state = TaskState(
        run_id="s3",
        status=TaskStatus.SUCCESS,
        message="Spatial simulation complete (121 time steps).",
    )
    alert_class, _icon, message = spatial_task_alert(state)

    assert alert_class == "alert alert-success mt-2"
    assert "121 time steps" in message


def test_ecospace_ui_warns_runs_are_not_ewe_equivalent():
    """Audit 2026-11-10 B1-B4: imported Ecospace runs are not EwE output."""
    html = str(ecospace_ui())
    assert 'id="ecospace_parity_warning"' in html
    assert "not equivalent to EwE Ecospace" in html


# --- the per-group arrays the page's run hands to EcospaceParams ---


def _fished_scenario():
    """A 4-group model with one fleet, balanced and turned into a scenario."""
    from pypath.core.ecopath import rpath
    from pypath.core.ecosim import rsim_scenario
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(
        ["Fish", "Plankton", "Detritus", "Fleet"], [0, 1, 2, 3]
    )
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [10.0, 1.0, 5.0, 0.8]
    params.model.loc[1, ["Biomass", "PB", "EE"]] = [5.0, 50.0, 0.6]
    params.model.loc[2, "Biomass"] = 1.0
    params.diet.iloc[1, 1] = 1.0
    params.model.loc[0, "Fleet"] = 0.5
    model = rpath(params)
    return rsim_scenario(model, params, years=range(1, 3))


def test_page_ecospace_arrays_give_every_group_the_page_settings():
    # Ecospace arrays are indexed from 0 by group (row g - 1 is Ecosim group
    # g), so there is no "Outside" row: group 1 is row 0.
    import numpy as np

    from pypath_shiny.pages.ecospace import page_ecospace_arrays

    habitat = np.array([0.2, 0.6, 1.0])
    arrays = page_ecospace_arrays(
        habitat, 4, dispersal_rate=5.0, advection=True, gravity=0.3
    )
    np.testing.assert_array_equal(
        arrays["habitat_preference"], np.tile(habitat, (4, 1))
    )
    np.testing.assert_array_equal(arrays["habitat_capacity"], np.tile(habitat, (4, 1)))
    np.testing.assert_array_equal(arrays["dispersal_rate"], [5.0] * 4)
    np.testing.assert_array_equal(arrays["advection_enabled"], [True] * 4)
    np.testing.assert_array_equal(arrays["gravity_strength"], [0.3] * 4)


def test_group_one_starts_on_the_page_habitat_in_a_run():
    # A run starts each group in proportion to its habitat capacity. With an
    # "Outside" row 0, group 1 read that row: a constant (all-zero) capacity
    # row, so it started uniform whatever the pattern, and no dispersal.
    import numpy as np
    from pypath.spatial import EcospaceParams, create_regular_grid, rsim_run_spatial

    from pypath_shiny.pages.ecospace import page_ecospace_arrays

    scenario = _fished_scenario()
    grid = create_regular_grid(bounds=(0, 0, 3, 1), nx=3, ny=1)
    habitat = np.array([0.2, 0.6, 1.0])
    eco = EcospaceParams(
        grid=grid,
        **page_ecospace_arrays(
            habitat,
            scenario.params.NUM_GROUPS,
            dispersal_rate=5.0,
            advection=True,
            gravity=0.3,
        ),
    )
    result = rsim_run_spatial(scenario, ecospace=eco)
    assert np.all(np.isfinite(result.out_Biomass_spatial))
    fish = result.out_Biomass_spatial[0, 1, :]  # Ecosim group 1 at the start
    np.testing.assert_allclose(fish / fish.sum(), habitat / habitat.sum())
