"""Tests for services/background_runs.py: the Ecospace and IBM worker
functions and the checks that decide whether a finished run may still land."""

import ast
from pathlib import Path

import numpy as np
import pytest
from pypath.core.ecosim import RunCancelled
from pypath.spatial import EcospaceParams, create_regular_grid, rsim_run_spatial

from pypath_shiny.pages import ecospace as ecospace_page
from pypath_shiny.pages import ibm as ibm_page
from pypath_shiny.pages.ecospace import page_ecospace_arrays
from pypath_shiny.services import background_runs as runs
from pypath_shiny.services.ecosim_solver import RunControl
from pypath_shiny.services.tasks import TaskStatus, finish_task, new_running_task
from tests.test_ecospace_page import _fished_scenario


def _spatial_inputs():
    scenario = _fished_scenario()
    grid = create_regular_grid(bounds=(0, 0, 3, 1), nx=3, ny=1)
    eco = EcospaceParams(
        grid=grid,
        **page_ecospace_arrays(
            np.array([0.2, 0.6, 1.0]),
            scenario.params.NUM_GROUPS,
            dispersal_rate=5.0,
            advection=False,
            gravity=0.0,
        ),
    )
    return scenario, grid, eco


# --- Ecospace worker ---------------------------------------------------


def test_run_spatial_gives_the_direct_run():
    scenario, _, eco = _spatial_inputs()
    direct = rsim_run_spatial(_fished_scenario(), ecospace=eco)
    worker = runs.run_spatial(scenario, eco, None)
    np.testing.assert_array_equal(
        worker.out_Biomass_spatial, direct.out_Biomass_spatial
    )


def test_run_spatial_reports_each_month_to_its_control():
    scenario, _, eco = _spatial_inputs()
    control = RunControl()
    out = runs.run_spatial(scenario, eco, None, control=control)
    n_months = out.out_Biomass_spatial.shape[0] - 1
    assert control.progress() == (n_months, n_months)


def test_run_spatial_stops_when_its_control_is_asked_to():
    scenario, _, eco = _spatial_inputs()
    control = RunControl()
    control.request_stop()
    with pytest.raises(RunCancelled, match="month 1 of"):
        runs.run_spatial(scenario, eco, None, control=control)
    assert control.progress()[0] == 1


# --- Ecospace staleness ------------------------------------------------


def _spatial_outcome(run_id="spatial-run-1"):
    scenario, grid, _ = _spatial_inputs()
    return runs.SpatialOutcome(
        run_id=run_id, source=scenario, grid=grid, output=object(), switching=None
    )


def test_spatial_result_lands_while_nothing_changed():
    outcome = _spatial_outcome()
    state = new_running_task(outcome.run_id, "Running")
    assert runs.spatial_result_is_current(state, outcome, outcome.source, outcome.grid)


def test_spatial_result_is_dropped_after_a_grid_change():
    # The page is no longer frozen during a run, so the grid can be replaced
    # under it: a result with the old patch count must not reach the maps.
    outcome = _spatial_outcome()
    state = new_running_task(outcome.run_id, "Running")
    other_grid = create_regular_grid(bounds=(0, 0, 2, 1), nx=2, ny=1)
    assert not runs.spatial_result_is_current(
        state, outcome, outcome.source, other_grid
    )
    assert not runs.spatial_result_is_current(state, outcome, outcome.source, None)


def test_spatial_result_is_dropped_after_a_scenario_change():
    outcome = _spatial_outcome()
    state = new_running_task(outcome.run_id, "Running")
    assert not runs.spatial_result_is_current(
        state, outcome, _fished_scenario(), outcome.grid
    )
    assert not runs.spatial_result_is_current(state, outcome, None, outcome.grid)


def test_spatial_result_is_dropped_when_superseded_or_finished():
    outcome = _spatial_outcome()
    newer = new_running_task("spatial-run-2", "Running")
    assert not runs.spatial_result_is_current(
        newer, outcome, outcome.source, outcome.grid
    )
    done = finish_task(
        new_running_task(outcome.run_id, "Running"), success=False, message="x"
    )
    assert done.status is not TaskStatus.RUNNING
    assert not runs.spatial_result_is_current(
        done, outcome, outcome.source, outcome.grid
    )


# --- IBM worker --------------------------------------------------------


class _FakeIbm:
    """Stands in for SmeltIBM: records how it was built and initialised."""

    def __init__(self, group_index, n_groups, params):
        self.built = (group_index, n_groups, params)
        self.initialised = None

    def initialize_from_ecosim(self, biomass, params, n_super_individuals):
        self.initialised = (biomass, params, n_super_individuals)


@pytest.fixture
def ibm_calls(monkeypatch):
    calls = []

    def fake_rsim_run(scenario, **kwargs):
        calls.append((scenario, kwargs))
        return f"output-{len(calls)}"

    monkeypatch.setattr(runs, "rsim_run", fake_rsim_run)
    monkeypatch.setattr(runs, "SmeltIBM", _FakeIbm)
    return calls


def test_ibm_comparison_runs_standard_then_ibm_on_separate_copies(ibm_calls):
    scenario = _fished_scenario()
    spec = runs.IbmRunSpec(group_index=1, n_groups=4, params="P", n_super=7)

    standard, enhanced = runs.run_ibm_comparison(scenario, spec)

    assert (standard, enhanced) == ("output-1", "output-2")
    std_scen, ibm_scen = ibm_calls[0][0], ibm_calls[1][0]
    # Neither run touches the caller's scenario, nor each other's.
    assert std_scen is not scenario and ibm_scen is not scenario
    assert std_scen is not ibm_scen
    assert not getattr(std_scen.params, "ibm_groups", None)
    assert not getattr(scenario.params, "ibm_groups", None)
    fresh = ibm_scen.params.ibm_groups[1]
    assert fresh.built == (1, 4, "P")
    biomass = float(scenario.params.B_BaseRef[1])
    assert fresh.initialised == (biomass, {}, 7)


def test_ibm_comparison_hands_the_stop_hook_to_both_runs(ibm_calls):
    control = RunControl()
    spec = runs.IbmRunSpec(group_index=1, n_groups=4, params="P", n_super=7)
    runs.run_ibm_comparison(_fished_scenario(), spec, control=control)
    assert [c[1].get("on_month") for c in ibm_calls] == [control.on_month] * 2


def test_ibm_comparison_without_control_passes_no_hook(ibm_calls):
    spec = runs.IbmRunSpec(group_index=1, n_groups=4, params="P", n_super=7)
    runs.run_ibm_comparison(_fished_scenario(), spec)
    assert all("on_month" not in c[1] for c in ibm_calls)


def test_ibm_result_is_current_needs_the_same_scenario_and_ibm():
    scenario, ibm = _fished_scenario(), object()
    outcome = runs.IbmOutcome(
        run_id=3, source=scenario, ibm=ibm, standard="s", enhanced="e", switching=None
    )
    assert runs.ibm_result_is_current(3, outcome, scenario, ibm)
    assert not runs.ibm_result_is_current(4, outcome, scenario, ibm)
    assert not runs.ibm_result_is_current(3, outcome, _fished_scenario(), ibm)
    assert not runs.ibm_result_is_current(3, outcome, scenario, object())
    assert not runs.ibm_result_is_current(3, outcome, None, ibm)


# --- The pages run off the event loop ----------------------------------


def _functions(page):
    tree = ast.parse(Path(page.__file__).read_text(encoding="utf-8"))
    return {
        f.name: f
        for f in ast.walk(tree)
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _decorators(fn):
    return [ast.unparse(d) for d in fn.decorator_list]


@pytest.mark.parametrize(
    "page,launcher,task,heavy",
    [
        (ecospace_page, "_run_spatial_simulation", "_spatial_task", "rsim_run_spatial"),
        (ibm_page, "_run_comparison", "_ibm_task", "rsim_run"),
    ],
)
def test_the_launch_effect_does_not_integrate_on_the_event_loop(
    page, launcher, task, heavy
):
    fns = _functions(page)
    launch = ast.unparse(fns[launcher])
    # The effect only prepares and invokes; the integration is the task's.
    assert f"{heavy}(" not in launch
    assert f"{task}.invoke(" in launch
    assert "reactive.extended_task" in _decorators(fns[task])
    body = ast.unparse(fns[task])
    assert "RUN_EXECUTOR.submit(" in body and "wrap_future" in body
    # The worker reads no input and sends no notification.
    assert "input." not in body and "_notify" not in body


@pytest.mark.parametrize(
    "page,launcher,task",
    [
        (ecospace_page, "_run_spatial_simulation", "_spatial_task"),
        (ibm_page, "_run_comparison", "_ibm_task"),
    ],
)
def test_a_second_click_is_refused_while_a_run_is_in_flight(page, launcher, task):
    launch = ast.unparse(_functions(page)[launcher])
    guard = launch.split(f"{task}.invoke(")[0]
    assert f"{task}.status() == 'running'" in guard
    assert "run_slot.busy" in guard


@pytest.mark.parametrize("page", [ecospace_page, ibm_page])
def test_a_closed_session_cancels_its_run(page):
    assert "session.on_ended(" in Path(page.__file__).read_text(encoding="utf-8")


# --- The scenario was edited in place after the run started -------------


def test_fingerprint_is_stable_for_an_untouched_scenario():
    scenario = _fished_scenario()
    first = runs.scenario_fingerprint(scenario)
    assert first and first == runs.scenario_fingerprint(scenario)
    # A deep copy has the same content, so the same fingerprint.
    import copy

    assert runs.scenario_fingerprint(copy.deepcopy(scenario)) == first


def test_fingerprint_changes_with_an_in_place_forcing_edit():
    # The Ecosim page's Biomass Forcing card edits ForcedPrey in place when an
    # Ecosim run starts: the scenario object stays the same, its content not.
    from pypath_shiny.pages.ecosim import apply_prey_forcing

    scenario = _fished_scenario()
    before = runs.scenario_fingerprint(scenario)
    assert apply_prey_forcing(scenario, "Plankton", 2.0)
    assert runs.scenario_fingerprint(scenario) != before
    # Back at a multiplier of 1 the content, and so the fingerprint, returns.
    apply_prey_forcing(scenario, "Plankton", 1.0)
    assert runs.scenario_fingerprint(scenario) == before


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s.fishing.ForcedEffort.__setitem__((0, 1), 3.0),
        lambda s: s.fishing.ForcedFRate.__setitem__((0, 1), 0.7),
        lambda s: s.forcing.ForcedMort.__setitem__((5, 1), 1.5),
        lambda s: s.params.VV.__setitem__(1, s.params.VV[1] * 2),
    ],
    ids=["effort", "frate", "mortality", "vulnerability"],
)
def test_fingerprint_changes_with_other_in_place_edits(edit):
    scenario = _fished_scenario()
    before = runs.scenario_fingerprint(scenario)
    edit(scenario)
    assert runs.scenario_fingerprint(scenario) != before


def test_scenario_changed_since_compares_with_the_launch_fingerprint():
    scenario, grid, _ = _spatial_inputs()
    outcome = runs.SpatialOutcome(
        run_id="r",
        source=scenario,
        grid=grid,
        output=object(),
        switching=None,
        fingerprint=runs.scenario_fingerprint(scenario),
    )
    assert not runs.scenario_changed_since(outcome, scenario)
    scenario.fishing.ForcedEffort[0, 1] = 3.0
    assert runs.scenario_changed_since(outcome, scenario)
    # An outcome without a fingerprint makes no claim.
    bare = runs.SpatialOutcome("r", scenario, grid, object(), None)
    assert not runs.scenario_changed_since(bare, scenario)


@pytest.mark.parametrize(
    "page,launcher,lander",
    [
        (ecospace_page, "_run_spatial_simulation", "_land_spatial_result"),
        (ibm_page, "_run_comparison", "_land_ibm_result"),
    ],
)
def test_the_pages_take_the_fingerprint_at_launch_and_note_a_change(
    page, launcher, lander
):
    fns = _functions(page)
    launch = ast.unparse(fns[launcher])
    assert "scenario_fingerprint(scenario)" in launch
    assert "fingerprint=" in launch.split(".invoke(")[1]
    land = ast.unparse(fns[lander])
    assert "scenario_changed_since(" in land
    # The result still lands: the note comes with it, it does not replace it.
    assert ".set(" in land and "changed after this run started" in land
