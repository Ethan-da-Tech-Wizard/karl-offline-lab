# ==============================================================================
# Program: test_task_supervisor_improvements.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising task supervisor improvements functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import os
from app.engine.task_supervisor import TaskSupervisor, TaskStatus


def test_task_supervisor_terminal_state_protection():
    """Terminal states are sticky: once a task has failed (ERROR with message), a later finish() must not change it to FINISHED or clear the error. This stops a watchdog failure being overwritten by a late success."""
    # Use reset_instance to get a clean singleton for testing
    TaskSupervisor.reset_instance()
    ts = TaskSupervisor.instance()

    # 1. Test registration
    task_id = ts.register("Test Task")
    assert ts.status(task_id) == TaskStatus.RUNNING

    # 2. Transition to fail (terminal state)
    ts.fail(task_id, "Some watchdog timeout error")
    assert ts.status(task_id) == TaskStatus.ERROR
    assert ts.error(task_id) == "Some watchdog timeout error"

    # 3. Try to transition to finish (success) after failure
    ts.finish(task_id)
    # The status must remain ERROR, not transition to FINISHED
    assert ts.status(task_id) == TaskStatus.ERROR
    assert ts.error(task_id) == "Some watchdog timeout error"


def test_task_supervisor_finish_once_terminal():
    """The same in reverse: after finish(), a late fail() is a no-op and no error is recorded."""
    # Step 1: Validate input parameters, operational preconditions, and state
    TaskSupervisor.reset_instance()
    ts = TaskSupervisor.instance()

    task_id = ts.register("Another Test Task")
    assert ts.status(task_id) == TaskStatus.RUNNING

    ts.finish(task_id)
    assert ts.status(task_id) == TaskStatus.FINISHED

    # Attempting to fail after finish should be a no-op
    ts.fail(task_id, "Late failure attempt")
    assert ts.status(task_id) == TaskStatus.FINISHED
    assert ts.error(task_id) == ""


def test_cpu_affinity_helper_availability():
    """On platforms with CPU affinity support, sched_getaffinity returns a non-empty set (used for pinning worker threads); skipped or safe elsewhere."""
    # Verify os.sched_getaffinity / os.sched_setaffinity exist or behave safely on this OS (Linux)
    if hasattr(os, "sched_getaffinity"):
        affinity = os.sched_getaffinity(0)
        assert isinstance(affinity, set)
        assert len(affinity) > 0
