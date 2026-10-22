"""Task state helpers with explicit transitions and stale-run protection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable


class TaskStatus(str, Enum):
    """Allowed task lifecycle statuses."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


_ALLOWED_TRANSITIONS = {
    TaskStatus.IDLE: {TaskStatus.RUNNING},
    TaskStatus.RUNNING: {TaskStatus.SUCCESS, TaskStatus.ERROR, TaskStatus.CANCELLED},
    TaskStatus.SUCCESS: set(),
    TaskStatus.ERROR: set(),
    TaskStatus.CANCELLED: set(),
}


@dataclass(frozen=True)
class TaskState:
    """Immutable state for long-running operations in UI pages."""

    run_id: str
    status: TaskStatus = TaskStatus.IDLE
    message: str = ""
    progress: float = 0.0
    can_cancel: bool = False
    error: str | None = None


def transition_task_state(
    state: TaskState,
    new_status: TaskStatus,
    *,
    message: str | None = None,
    progress: float | None = None,
    error: str | None = None,
    can_cancel: bool | None = None,
) -> TaskState:
    """Transition a task state while enforcing lifecycle rules."""
    if new_status not in _ALLOWED_TRANSITIONS[state.status]:
        raise ValueError(f"Invalid task transition: {state.status} -> {new_status}")

    return replace(
        state,
        status=new_status,
        message=state.message if message is None else message,
        progress=state.progress if progress is None else float(progress),
        error=state.error if error is None else error,
        can_cancel=(new_status is TaskStatus.RUNNING)
        if can_cancel is None
        else can_cancel,
    )


def apply_if_latest(
    state: TaskState, run_id: str, updater: Callable[[TaskState], TaskState]
) -> TaskState:
    """Apply updater only when the incoming run_id matches the latest state run_id."""
    if run_id != state.run_id:
        return state
    return updater(state)


def new_running_task(run_id: str, message: str) -> TaskState:
    """Create a new running task state, resetting stale terminal state."""
    return TaskState(
        run_id=run_id,
        status=TaskStatus.RUNNING,
        message=message,
        progress=0.0,
        can_cancel=True,
        error=None,
    )


def update_task_progress(
    state: TaskState, *, message: str | None = None, progress: float | None = None
) -> TaskState:
    """Update message/progress while a task is running."""
    if state.status is not TaskStatus.RUNNING:
        raise ValueError("Can only update progress for running tasks")

    next_progress = state.progress if progress is None else float(progress)
    next_progress = max(0.0, min(100.0, next_progress))

    return replace(
        state,
        message=state.message if message is None else message,
        progress=next_progress,
        can_cancel=True,
    )


def finish_task(
    state: TaskState,
    *,
    success: bool,
    message: str,
    error: str | None = None,
) -> TaskState:
    """Mark a running task as success or error with a final message."""
    terminal = TaskStatus.SUCCESS if success else TaskStatus.ERROR
    progress = 100.0 if success else state.progress
    return transition_task_state(
        state,
        terminal,
        message=message,
        progress=progress,
        error=error,
        can_cancel=False,
    )
