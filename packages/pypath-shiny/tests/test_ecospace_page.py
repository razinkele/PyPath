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


# --- the Ecospace Wizard's configuration in a page run ---


def _grid():
    from pypath.spatial import create_regular_grid

    return create_regular_grid(bounds=(0, 0, 3, 1), nx=3, ny=1)


def _wizard_params(grid, n_rows, drivers=None):
    """EcospaceParams as the wizard builds them: one row per group, by name."""
    import numpy as np
    from pypath.io.marine_data import HabitatPreferenceBuilder
    from pypath.spatial import EcospaceParams

    prefs = np.array([[0.9, 0.2], [0.3, 0.8], [0.5, 0.5], [0.1, 0.7], [0.4, 0.6]])
    habitat_map = np.array(["A5.1", "A5.2", "A5.2"], dtype=object)
    pref = HabitatPreferenceBuilder().build_preference_matrix(
        prefs[:n_rows], ["A5.1", "A5.2"], habitat_map, grid
    )
    return EcospaceParams(
        grid=grid,
        habitat_preference=pref,
        habitat_capacity=pref.copy(),
        dispersal_rate=np.full(n_rows, 7.0),
        advection_enabled=np.ones(n_rows, dtype=bool),
        gravity_strength=np.full(n_rows, 0.5),
        environmental_drivers=drivers,
    )


def _page_arrays(n_groups):
    import numpy as np

    from pypath_shiny.pages.ecospace import page_ecospace_arrays

    return page_ecospace_arrays(
        np.array([0.2, 0.6, 1.0]),
        n_groups,
        dispersal_rate=5.0,
        advection=True,
        gravity=0.3,
    )


def test_the_wizard_configuration_runs_on_its_grid():
    from pypath_shiny.pages.ecospace import run_ecospace_params

    grid = _grid()
    wizard = _wizard_params(grid, 4)
    params, note = run_ecospace_params(grid, 4, _page_arrays(4), wizard_params=wizard)
    assert params is wizard
    assert "Ecospace Wizard" in note


def test_a_new_grid_goes_back_to_the_page_controls():
    import numpy as np

    from pypath_shiny.pages.ecospace import run_ecospace_params

    wizard = _wizard_params(_grid(), 4)
    new_grid = _grid()  # equal, but a grid built on this page
    params, note = run_ecospace_params(
        new_grid, 4, _page_arrays(4), wizard_params=wizard
    )
    assert params.grid is new_grid
    np.testing.assert_array_equal(
        params.habitat_preference, _page_arrays(4)["habitat_preference"]
    )
    assert note is None


def test_a_wizard_configuration_for_another_group_count_is_not_used():
    # e.g. built before a model was loaded (5 placeholder groups)
    import numpy as np

    from pypath_shiny.pages.ecospace import run_ecospace_params

    grid = _grid()
    params, note = run_ecospace_params(
        grid, 4, _page_arrays(4), wizard_params=_wizard_params(grid, 5)
    )
    np.testing.assert_array_equal(params.dispersal_rate, [5.0] * 4)
    assert "5 groups" in note and "4" in note


def test_without_a_wizard_configuration_the_page_controls_run():
    import numpy as np

    from pypath_shiny.pages.ecospace import run_ecospace_params

    params, note = run_ecospace_params(_grid(), 4, _page_arrays(4))
    np.testing.assert_array_equal(params.gravity_strength, [0.3] * 4)
    assert note is None


def test_the_habitat_display_shows_the_wizard_mean_on_its_grid():
    import numpy as np

    from pypath_shiny.pages.ecospace import habitat_for_display

    grid = _grid()
    wizard = _wizard_params(grid, 4)
    page = np.array([0.2, 0.6, 1.0])
    np.testing.assert_allclose(
        habitat_for_display(wizard, grid, page),
        wizard.habitat_preference.mean(axis=0),
    )
    np.testing.assert_array_equal(habitat_for_display(wizard, _grid(), page), page)


def test_a_wizard_configuration_runs_each_group_on_its_own_preferences():
    import numpy as np
    from pypath.spatial import rsim_run_spatial

    from pypath_shiny.pages.ecospace import run_ecospace_params
    from pypath_shiny.pages.ecospace_wizard import wizard_environmental_drivers

    scenario = _fished_scenario()  # Fish, Plankton, Detritus, Fleet
    grid = _grid()
    n = scenario.params.NUM_GROUPS
    drivers = wizard_environmental_drivers(np.array([10.0, 20.0, 30.0]))
    wizard = _wizard_params(grid, n, drivers=drivers)
    params, _ = run_ecospace_params(grid, n, _page_arrays(n), wizard_params=wizard)
    result = rsim_run_spatial(scenario, ecospace=params)

    assert np.all(np.isfinite(result.out_Biomass_spatial))
    start = result.out_Biomass_spatial[0]
    for group in (1, 2):  # Fish and Plankton follow their own wizard rows
        row = wizard.habitat_capacity[group - 1]
        np.testing.assert_allclose(start[group] / start[group].sum(), row / row.sum())


def test_the_display_ignores_a_wizard_configuration_runs_would_not_use():
    # With the scenario's group count known, a configuration for another
    # count is not shown, as a run would not use it.
    import numpy as np

    from pypath_shiny.pages.ecospace import habitat_for_display, wizard_params_on_grid

    grid = _grid()
    wizard = _wizard_params(grid, 5)
    page = np.array([0.2, 0.6, 1.0])
    assert wizard_params_on_grid(wizard, grid, n_groups=4) is None
    assert wizard_params_on_grid(wizard, grid, n_groups=5) is wizard
    np.testing.assert_array_equal(
        habitat_for_display(wizard, grid, page, n_groups=4), page
    )


def _handler_reads_wizard_params(source, handler):
    import ast

    tree = ast.parse(source)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == handler
    )
    return "_ecospace_params" in ast.unparse(fn)


def test_the_run_and_the_habitat_views_read_the_wizard_configuration():
    # The wizard's configuration was stored in _ecospace_params, and nothing
    # read it: runs, the habitat map and the effort preview used only the
    # page's controls.
    from pypath_shiny.pages import ecospace

    source = Path(ecospace.__file__).read_text(encoding="utf-8")
    for handler in (
        "_run_spatial_simulation",
        "_push_habitat_map",
        "wizard_config_note",
    ):
        assert _handler_reads_wizard_params(source, handler), handler
