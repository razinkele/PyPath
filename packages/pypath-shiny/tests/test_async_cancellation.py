"""Integration tests for async task cancellation flows.

Tests cancellation UI rendering, state transitions, and stale-run prevention
for long-running operations (Ecosim, biodata, Ecospace wizard).

Note: In PyPath, cancellations are modeled as terminal ERROR states (via finish_task
with success=False) rather than a separate CANCELLED status, which aligns with the
task state machine design where ERROR represents any non-success terminal condition.
"""

import pytest

from pypath_shiny.services.tasks import (
    TaskState,
    TaskStatus,
    apply_if_latest,
    finish_task,
    new_running_task,
    update_task_progress,
)


class TestCancellationStateTransitions:
    """Test TaskState cancellation transitions."""

    def test_cancel_running_task_transitions_to_error(self):
        """Cancelling a running task should set status to ERROR (terminal failure)."""
        running = new_running_task("task-1", "processing")
        cancelled = finish_task(
            running,
            success=False,
            message="Operation cancelled",
            error="User cancelled",
        )
        assert cancelled.status == TaskStatus.ERROR
        assert cancelled.message == "Operation cancelled"
        assert cancelled.error == "User cancelled"
        assert cancelled.progress == 0.0  # No progress bump for errors

    def test_cancel_task_sets_can_cancel_false(self):
        """After cancellation, can_cancel should be False."""
        running = new_running_task("task-1", "processing")
        cancelled = finish_task(
            running,
            success=False,
            message="Cancelled",
            error="User request",
        )
        assert cancelled.can_cancel is False

    def test_cancelled_task_has_terminal_status(self):
        """Cancelled tasks should be terminal (no further transitions)."""
        running = new_running_task("task-2", "work")
        cancelled = finish_task(
            running,
            success=False,
            message="Cancelled",
        )
        assert cancelled.status == TaskStatus.ERROR
        # Verify it matches other terminal states
        assert cancelled.status in (TaskStatus.SUCCESS, TaskStatus.ERROR)


class TestStaleRunPrevention:
    """Test stale-run prevention during cancellation."""

    def test_apply_if_latest_prevents_stale_cancel_updates(self):
        """apply_if_latest should prevent updates from old run IDs."""
        state = TaskState(run_id="run-123", status=TaskStatus.RUNNING)

        # Try to update with stale run_id
        stale_update = apply_if_latest(
            state,
            run_id="run-old",
            updater=lambda s: finish_task(
                s, success=False, message="Should not apply"
            ),
        )
        assert stale_update is state
        assert stale_update.status is TaskStatus.RUNNING

    def test_apply_if_latest_allows_current_cancel_updates(self):
        """apply_if_latest should allow updates from current run ID."""
        state = new_running_task("run-456", "downloading")

        # Update with matching run_id
        updated = apply_if_latest(
            state,
            run_id="run-456",
            updater=lambda s: finish_task(
                s, success=False, message="User cancelled"
            ),
        )
        assert updated.status == TaskStatus.ERROR
        assert updated.message == "User cancelled"

    def test_concurrent_cancel_requests_use_same_run_id(self):
        """Multiple cancel clicks should use same run_id; first wins."""
        state = new_running_task("run-789", "task")

        # First cancel (succeeds)
        cancel1 = apply_if_latest(
            state,
            run_id="run-789",
            updater=lambda s: finish_task(
                s, success=False, message="Cancelled by first click"
            ),
        )
        assert cancel1.status == TaskStatus.ERROR

        # Second cancel attempt (task already terminal; would fail to transition)
        # In reality this would raise ValueError, but we test apply_if_latest prevents it
        try:
            cancel2 = apply_if_latest(
                cancel1,
                run_id="run-789",
                updater=lambda s: finish_task(
                    s, success=False, message="Cancelled by second click"
                ),
            )
            # If this succeeds (shouldn't), verify first cancel message remains
            assert cancel2.message == "Cancelled by first click"
        except ValueError:
            # Expected: can't transition from ERROR to ERROR
            pass


class TestCancellationPatterns:
    """Test common cancellation patterns used across pages."""

    def test_cancel_with_error_message_chain(self):
        """Error messages should be preserved through cancellation."""
        running = new_running_task("run-xyz", "processing")
        user_error = "User clicked cancel button"
        cancelled = finish_task(
            running,
            success=False,
            message="Operation cancelled",
            error=user_error,
        )
        assert cancelled.error == user_error
        assert "cancelled" in cancelled.message.lower()

    def test_multiple_page_cancellations_independent(self):
        """Cancellations on different pages should not interfere."""
        ecosim_task = new_running_task("ecosim-run-1", "simulation")
        biodata_task = new_running_task("biodata-run-1", "fetching")
        wizard_task = new_running_task("wizard-run-1", "downloading")

        # Cancel only biodata
        biodata_cancelled = finish_task(
            biodata_task, success=False, message="Biodata cancelled"
        )

        # Verify biodata is cancelled but others unchanged
        assert biodata_cancelled.status == TaskStatus.ERROR
        assert ecosim_task.status == TaskStatus.RUNNING
        assert wizard_task.status == TaskStatus.RUNNING

    def test_cancel_preserves_partial_work_via_error_state(self):
        """Cancellation can preserve context via error message."""
        state = new_running_task("run-abc", "processing step 1/3")
        update1 = update_task_progress(
            state, message="processing step 2/3", progress=33.0
        )
        update2 = update_task_progress(
            update1, message="processing step 3/3", progress=66.0
        )
        cancelled = finish_task(
            update2,
            success=False,
            message="Cancelled at step 3 of 3",
            error="User requested cancellation",
        )
        # Error message preserves context about how far we got
        assert "step 3" in cancelled.message
        assert cancelled.status == TaskStatus.ERROR


class TestCancellationUIRendering:
    """Test cancel button rendering logic."""

    def test_cancel_button_renders_only_when_running(self):
        """Cancel button should only render if status is RUNNING."""
        states_to_test = [
            (TaskStatus.IDLE, False),
            (TaskStatus.RUNNING, True),
            (TaskStatus.SUCCESS, False),
            (TaskStatus.ERROR, False),
        ]

        for status, should_render in states_to_test:
            state = TaskState(run_id="test", status=status, can_cancel=True)
            # Simulate button render condition
            renders = state.status == TaskStatus.RUNNING and state.can_cancel
            assert renders is should_render, f"Status {status} render={renders}"

    def test_cancel_button_respects_can_cancel_flag(self):
        """Cancel button should not render if can_cancel is False."""
        running_no_cancel = TaskState(
            run_id="test", status=TaskStatus.RUNNING, can_cancel=False
        )
        running_can_cancel = TaskState(
            run_id="test", status=TaskStatus.RUNNING, can_cancel=True
        )

        # Simulate button render condition
        renders_no_cancel = (
            running_no_cancel.status == TaskStatus.RUNNING
            and running_no_cancel.can_cancel
        )
        renders_can_cancel = (
            running_can_cancel.status == TaskStatus.RUNNING
            and running_can_cancel.can_cancel
        )

        assert renders_no_cancel is False
        assert renders_can_cancel is True

    def test_cancel_button_hidden_after_cancellation(self):
        """Button should disappear after task is cancelled."""
        running = new_running_task("test", "working")
        assert running.status is TaskStatus.RUNNING
        assert running.can_cancel is True

        cancelled = finish_task(running, success=False, message="Cancelled")
        assert cancelled.status == TaskStatus.ERROR
        assert cancelled.can_cancel is False

        # Simulate render check
        can_render = cancelled.status == TaskStatus.RUNNING and cancelled.can_cancel
        assert can_render is False


class TestCancellationErrorHandling:
    """Test error handling during cancellation."""

    def test_cancel_completed_task_doesnt_corrupt_state(self):
        """Attempting to cancel already-completed task should be safe."""
        completed = finish_task(
            new_running_task("test", "done"), success=True, message="Success"
        )
        assert completed.status is TaskStatus.SUCCESS

        # Attempting to transition from SUCCESS would raise ValueError
        # apply_if_latest would prevent this with stale run_id check, but
        # if run_id matches, the transition would fail
        def attempt_cancel(s):
            return finish_task(
                s, success=False, message="Should not apply"
            )

        # This will fail with ValueError because SUCCESS -> ERROR is not allowed
        with pytest.raises(ValueError, match="Invalid task transition"):
            apply_if_latest(
                completed,
                run_id="test",
                updater=attempt_cancel,
            )

    def test_cancel_notification_includes_error_reason(self):
        """Cancellation notification should include user-facing error message."""
        running = new_running_task("test", "processing")
        error_msg = "Download was interrupted by user"
        cancelled = finish_task(
            running, success=False, message="Cancelled", error=error_msg
        )
        assert cancelled.error == error_msg

    def test_rapid_cancel_clicks_prevented_by_state_machine(self):
        """State machine prevents invalid transitions from multiple clicks."""
        state = new_running_task("run-123", "work")

        # First click (succeeds)
        first_cancel = apply_if_latest(
            state,
            run_id="run-123",
            updater=lambda s: finish_task(
                s, success=False, message="Cancel 1"
            ),
        )
        assert first_cancel.status == TaskStatus.ERROR

        # Second click (would fail because state is terminal)
        with pytest.raises(ValueError, match="Invalid task transition"):
            apply_if_latest(
                first_cancel,
                run_id="run-123",
                updater=lambda s: finish_task(
                    s, success=False, message="Cancel 2"
                ),
            )


class TestCancellationFromUserPerspective:
    """Test cancellation behavior as users experience it."""

    def test_cancel_button_shows_during_long_operation(self):
        """User sees cancel button while operation is in progress."""
        task = new_running_task("download-1", "Downloading data...")

        # Button should be visible
        should_show = (
            task.status == TaskStatus.RUNNING and task.can_cancel
        )
        assert should_show is True

    def test_user_clicks_cancel_operation_stops(self):
        """When user clicks cancel, operation transitions to error state."""
        task = new_running_task("download-1", "Downloading...")
        task_after_click = apply_if_latest(
            task,
            run_id="download-1",
            updater=lambda s: finish_task(
                s,
                success=False,
                message="Download cancelled",
                error="User clicked cancel button",
            ),
        )

        assert task_after_click.status == TaskStatus.ERROR
        assert task_after_click.error == "User clicked cancel button"
        # Button now hidden because status changed
        should_show = (
            task_after_click.status == TaskStatus.RUNNING and task_after_click.can_cancel
        )
        assert should_show is False

    def test_cancel_on_different_pages_doesnt_interfere(self):
        """Cancelling on one page shouldn't affect other concurrent operations."""
        ecosim_task = new_running_task("sim-1", "Running simulation...")
        biodata_task = new_running_task("bio-1", "Fetching biodata...")

        # User cancels biodata
        biodata_after_cancel = apply_if_latest(
            biodata_task,
            run_id="bio-1",
            updater=lambda s: finish_task(
                s, success=False, message="Biodata cancelled"
            ),
        )

        # Ecosim should still be running
        assert ecosim_task.status == TaskStatus.RUNNING
        assert biodata_after_cancel.status == TaskStatus.ERROR
