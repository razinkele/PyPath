from pypath_shiny.services.tasks import (
    TaskState,
    TaskStatus,
    apply_if_latest,
    finish_task,
    new_running_task,
    transition_task_state,
    update_task_progress,
)


def test_task_state_valid_transition_sequence():
    state = TaskState(run_id="r1")
    running = transition_task_state(state, TaskStatus.RUNNING, message="starting")
    assert running.status is TaskStatus.RUNNING
    assert running.message == "starting"

    done = transition_task_state(
        running, TaskStatus.SUCCESS, message="done", progress=100.0
    )
    assert done.status is TaskStatus.SUCCESS
    assert done.progress == 100.0


def test_task_state_invalid_transition_raises():
    state = TaskState(run_id="r1", status=TaskStatus.SUCCESS)
    try:
        transition_task_state(state, TaskStatus.RUNNING)
    except ValueError as exc:
        assert "Invalid task transition" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_apply_if_latest_ignores_stale_run_id():
    state = TaskState(run_id="latest", status=TaskStatus.RUNNING)

    stale = apply_if_latest(
        state,
        run_id="old",
        updater=lambda s: transition_task_state(s, TaskStatus.SUCCESS),
    )
    assert stale is state
    assert stale.status is TaskStatus.RUNNING

    fresh = apply_if_latest(
        state,
        run_id="latest",
        updater=lambda s: transition_task_state(s, TaskStatus.SUCCESS),
    )
    assert fresh.status is TaskStatus.SUCCESS


def test_update_task_progress_clamps_range():
    state = new_running_task("r2", "running")
    next_state = update_task_progress(state, progress=120.0, message="almost")
    assert next_state.progress == 100.0
    assert next_state.message == "almost"


def test_update_task_progress_requires_running_state():
    state = TaskState(run_id="r3", status=TaskStatus.IDLE)
    try:
        update_task_progress(state, progress=10)
    except ValueError as exc:
        assert "running tasks" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_finish_task_success_sets_terminal_state_and_progress():
    state = new_running_task("r4", "working")
    done = finish_task(state, success=True, message="done")
    assert done.status is TaskStatus.SUCCESS
    assert done.progress == 100.0
    assert done.can_cancel is False
