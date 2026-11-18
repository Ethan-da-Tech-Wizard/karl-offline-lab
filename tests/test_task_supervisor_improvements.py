# Program: karl-offline-lab - test_task_supervisor_improvements.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Pytest tests that a TaskSupervisor task cannot leave a terminal state, plus a check that CPU affinity helpers behave on this OS.
"""Tests that TaskSupervisor tasks stay in their terminal state."""
import os
from app.engine.task_supervisor import TaskSupervisor, TaskStatus


def test_task_supervisor_terminal_state_protection():
    """Scenario: a task is failed and then finished.
    Expected: the status stays ERROR and the original error message is kept.
    """
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
    """Scenario: a task is finished and then failed.
    Expected: the status stays FINISHED and no error message is recorded.
    """
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
    """Scenario: the OS may or may not provide CPU affinity calls.
    Expected: when it does, the affinity of this process is a non-empty set.
    """
    # Verify os.sched_getaffinity / os.sched_setaffinity exist or behave safely on this OS (Linux)
    if hasattr(os, "sched_getaffinity"):
        affinity = os.sched_getaffinity(0)
        assert isinstance(affinity, set)
        assert len(affinity) > 0
