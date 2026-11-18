# Program: karl-offline-lab - test_task_supervisor.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Unit tests for TaskSupervisor covering task registration, lifecycle transitions, cancellation, cleanup hooks, singleton behaviour and the model loader generation lock.
"""Tests for TaskSupervisor — task registry, lifecycle, and cancellation."""

from __future__ import annotations

import time
import threading
import unittest

from app.engine.task_supervisor import TaskSupervisor, TaskStatus


def _fresh() -> TaskSupervisor:
    """Return a brand-new TaskSupervisor instance, isolated from the singleton."""
    TaskSupervisor.reset_instance()
    sup = TaskSupervisor.instance()
    return sup


class TestTaskRegistration(unittest.TestCase):
    """Tests for registering tasks and reading their status, progress and error."""

    def setUp(self):
        """Create a fresh supervisor before each test."""
        self.sup = _fresh()

    def tearDown(self):
        """Reset the singleton so later tests start clean."""
        TaskSupervisor.reset_instance()

    def test_register_returns_unique_ids(self):
        """Scenario: two tasks are registered.
        Expected: their ids differ.
        """
        id1 = self.sup.register("task A")
        id2 = self.sup.register("task B")
        self.assertNotEqual(id1, id2)

    def test_register_explicit_task_id(self):
        """Scenario: a task is registered with a chosen id.
        Expected: that exact id is returned.
        """
        tid = self.sup.register("explicit", task_id="fixed-id-123")
        self.assertEqual(tid, "fixed-id-123")

    def test_registered_task_is_running(self):
        """Scenario: a new task is registered.
        Expected: its status is RUNNING.
        """
        tid = self.sup.register("job")
        self.assertEqual(self.sup.status(tid), TaskStatus.RUNNING)

    def test_unknown_task_returns_none_status(self):
        """Scenario: status is requested for an id that was never registered.
        Expected: None.
        """
        self.assertIsNone(self.sup.status("no-such-id"))

    def test_progress_zero_at_registration(self):
        """Scenario: a new task is registered.
        Expected: progress is 0.0.
        """
        tid = self.sup.register("job")
        self.assertAlmostEqual(self.sup.progress(tid), 0.0)

    def test_update_progress_clamped(self):
        """Scenario: progress is set to 0.5, then above 1.0, then below 0.0.
        Expected: values are clamped into the range 0.0 to 1.0.
        """
        tid = self.sup.register("job")
        self.sup.update_progress(tid, 0.5)
        self.assertAlmostEqual(self.sup.progress(tid), 0.5)
        self.sup.update_progress(tid, 2.0)   # above 1.0
        self.assertAlmostEqual(self.sup.progress(tid), 1.0)
        self.sup.update_progress(tid, -1.0)  # below 0.0
        self.assertAlmostEqual(self.sup.progress(tid), 0.0)

    def test_error_empty_at_registration(self):
        """Scenario: a new task is registered.
        Expected: the error message is an empty string.
        """
        tid = self.sup.register("job")
        self.assertEqual(self.sup.error(tid), "")


class TestTaskLifecycle(unittest.TestCase):
    """Tests for finishing and failing tasks and for listing active and all tasks."""

    def setUp(self):
        """Create a fresh supervisor before each test."""
        self.sup = _fresh()

    def tearDown(self):
        """Reset the singleton so later tests start clean."""
        TaskSupervisor.reset_instance()

    def test_finish_transitions_to_finished(self):
        """Scenario: a running task is finished.
        Expected: status is FINISHED.
        """
        tid = self.sup.register("job")
        self.sup.finish(tid)
        self.assertEqual(self.sup.status(tid), TaskStatus.FINISHED)

    def test_finish_sets_progress_to_one(self):
        """Scenario: a running task is finished.
        Expected: progress is 1.0.
        """
        tid = self.sup.register("job")
        self.sup.finish(tid)
        self.assertAlmostEqual(self.sup.progress(tid), 1.0)

    def test_fail_transitions_to_error(self):
        """Scenario: a running task is failed.
        Expected: status is ERROR.
        """
        tid = self.sup.register("job")
        self.sup.fail(tid, "something broke")
        self.assertEqual(self.sup.status(tid), TaskStatus.ERROR)

    def test_fail_stores_error_message(self):
        """Scenario: a task is failed with a message.
        Expected: the message can be read back.
        """
        tid = self.sup.register("job")
        self.sup.fail(tid, "disk full")
        self.assertEqual(self.sup.error(tid), "disk full")

    def test_finish_on_unknown_id_is_noop(self):
        """Scenario: finish is called for an unknown id.
        Expected: no exception.
        """
        self.sup.finish("ghost-id")  # must not raise

    def test_fail_on_unknown_id_is_noop(self):
        """Scenario: fail is called for an unknown id.
        Expected: no exception.
        """
        self.sup.fail("ghost-id", "whatever")  # must not raise

    def test_active_tasks_includes_running(self):
        """Scenario: one task is running.
        Expected: it appears in the active task list.
        """
        tid = self.sup.register("active")
        self.assertTrue(any(r.task_id == tid for r in self.sup.active_tasks()))

    def test_active_tasks_excludes_finished(self):
        """Scenario: a task is finished.
        Expected: it no longer appears in the active task list.
        """
        tid = self.sup.register("done")
        self.sup.finish(tid)
        self.assertFalse(any(r.task_id == tid for r in self.sup.active_tasks()))

    def test_all_tasks_returns_all(self):
        """Scenario: one task is finished and one is running.
        Expected: both appear in the full task list.
        """
        t1 = self.sup.register("t1")
        t2 = self.sup.register("t2")
        self.sup.finish(t1)
        ids = {r.task_id for r in self.sup.all_tasks()}
        self.assertIn(t1, ids)
        self.assertIn(t2, ids)


class TestCancellation(unittest.TestCase):
    """Tests for cancelling one task or all tasks and for the stop callback."""

    def setUp(self):
        """Create a fresh supervisor before each test."""
        self.sup = _fresh()

    def tearDown(self):
        """Reset the singleton so later tests start clean."""
        TaskSupervisor.reset_instance()

    def test_cancel_transitions_to_cancelling(self):
        """Scenario: a task with a stoppable object is cancelled.
        Expected: cancel returns True and the status is CANCELLING.
        """
        class _Stoppable:
            """Minimal cancellable object used by the tests."""
            def request_stop(self):
                """Stop request hook called by the supervisor on cancel."""
                pass

        tid = self.sup.register("job", cancellable=_Stoppable())
        result = self.sup.cancel(tid)
        self.assertTrue(result)
        self.assertEqual(self.sup.status(tid), TaskStatus.CANCELLING)

    def test_cancel_calls_request_stop(self):
        """Scenario: a task with a stoppable object is cancelled.
        Expected: request_stop is called exactly once.
        """
        stop_called = []

        class _Stoppable:
            """Minimal cancellable object used by the tests."""
            def request_stop(self):
                """Stop request hook called by the supervisor on cancel."""
                stop_called.append(True)

        tid = self.sup.register("job", cancellable=_Stoppable())
        self.sup.cancel(tid)
        self.assertEqual(stop_called, [True])

    def test_cancel_without_cancellable_returns_false(self):
        """Scenario: a task with no stoppable object is cancelled.
        Expected: cancel returns False but the status still becomes CANCELLING.
        """
        tid = self.sup.register("no-cancel")
        result = self.sup.cancel(tid)
        # status transitions to CANCELLING even without cancellable
        self.assertEqual(self.sup.status(tid), TaskStatus.CANCELLING)
        self.assertFalse(result)

    def test_cancel_finished_task_is_noop(self):
        """Scenario: a finished task is cancelled.
        Expected: cancel returns False and the status stays FINISHED.
        """
        tid = self.sup.register("job")
        self.sup.finish(tid)
        result = self.sup.cancel(tid)
        self.assertFalse(result)
        self.assertEqual(self.sup.status(tid), TaskStatus.FINISHED)

    def test_cancel_unknown_id_returns_false(self):
        """Scenario: cancel is called for an unknown id.
        Expected: False.
        """
        self.assertFalse(self.sup.cancel("ghost"))

    def test_cancel_all_cancels_all_active(self):
        """Scenario: two running tasks and cancel_all.
        Expected: the count is 2 and both stop callbacks run.
        """
        stop_counts = []

        class _Stoppable:
            """Minimal cancellable object used by the tests."""
            def request_stop(self):
                """Stop request hook called by the supervisor on cancel."""
                stop_counts.append(1)

        self.sup.register("a", cancellable=_Stoppable())
        self.sup.register("b", cancellable=_Stoppable())
        count = self.sup.cancel_all()
        self.assertEqual(count, 2)
        self.assertEqual(sum(stop_counts), 2)

    def test_cancel_all_skips_finished(self):
        """Scenario: the only task is already finished.
        Expected: cancel_all reports 0 cancelled.
        """
        tid = self.sup.register("done")
        self.sup.finish(tid)
        count = self.sup.cancel_all()
        self.assertEqual(count, 0)

    def test_request_stop_exception_does_not_propagate(self):
        """A crashing request_stop() must not take down cancel()."""
        class _Buggy:
            """Cancellable object whose stop request always raises an error."""
            def request_stop(self):
                """Stop request hook called by the supervisor on cancel."""
                raise RuntimeError("intentional crash in request_stop")

        tid = self.sup.register("buggy", cancellable=_Buggy())
        # Must not raise
        try:
            self.sup.cancel(tid)
        except RuntimeError:
            self.fail("cancel() let request_stop() exception escape")


class TestCleanupHooks(unittest.TestCase):
    """Tests for cleanup callbacks that run when a task ends."""

    def setUp(self):
        """Create a fresh supervisor before each test."""
        self.sup = _fresh()

    def tearDown(self):
        """Reset the singleton so later tests start clean."""
        TaskSupervisor.reset_instance()

    def test_cleanup_hook_called_on_finish(self):
        """Scenario: a task registered with a cleanup hook is finished.
        Expected: the hook runs once.
        """
        called = []
        tid = self.sup.register("job", cleanup_hook=lambda: called.append("finish"))
        self.sup.finish(tid)
        self.assertEqual(called, ["finish"])

    def test_cleanup_hook_called_on_fail(self):
        """Scenario: a task registered with a cleanup hook is failed.
        Expected: the hook runs once.
        """
        called = []
        tid = self.sup.register("job", cleanup_hook=lambda: called.append("fail"))
        self.sup.fail(tid, "oops")
        self.assertEqual(called, ["fail"])

    def test_multiple_cleanup_hooks_all_called(self):
        """Scenario: two hooks are added to a task which is then finished.
        Expected: both hooks run.
        """
        called = []
        tid = self.sup.register("job")
        self.sup.add_cleanup_hook(tid, lambda: called.append(1))
        self.sup.add_cleanup_hook(tid, lambda: called.append(2))
        self.sup.finish(tid)
        self.assertEqual(sorted(called), [1, 2])

    def test_failing_cleanup_hook_does_not_stop_others(self):
        """Scenario: the first hook raises an error.
        Expected: finish does not raise and the second hook still runs.
        """
        called = []

        def _bad():
            """Hook that always raises an error."""
            raise RuntimeError("hook crash")

        def _good():
            """Hook that records that it ran."""
            called.append("good")

        tid = self.sup.register("job")
        self.sup.add_cleanup_hook(tid, _bad)
        self.sup.add_cleanup_hook(tid, _good)
        self.sup.finish(tid)  # must not raise
        self.assertIn("good", called)

    def test_add_cleanup_hook_unknown_id_is_noop(self):
        """Scenario: a hook is added for an unknown id.
        Expected: no exception.
        """
        self.sup.add_cleanup_hook("ghost", lambda: None)  # must not raise


class TestSingleton(unittest.TestCase):
    """Tests that the supervisor is a thread-safe singleton that can be reset."""

    def tearDown(self):
        """Reset the singleton so later tests start clean."""
        TaskSupervisor.reset_instance()

    def test_instance_is_singleton(self):
        """Scenario: instance() is called twice.
        Expected: the same object is returned.
        """
        a = TaskSupervisor.instance()
        b = TaskSupervisor.instance()
        self.assertIs(a, b)

    def test_reset_produces_fresh_instance(self):
        """Scenario: a task is registered, the singleton is reset, and a new instance is fetched.
        Expected: the new instance has no tasks.
        """
        a = TaskSupervisor.instance()
        a.register("job")
        TaskSupervisor.reset_instance()
        b = TaskSupervisor.instance()
        self.assertEqual(b.all_tasks(), [])

    def test_singleton_thread_safe(self):
        """Concurrent instance() calls must all return the same object."""
        TaskSupervisor.reset_instance()
        results = []

        def _get():
            """Fetch the singleton and store it in the shared results list."""
            results.append(TaskSupervisor.instance())

        threads = [threading.Thread(target=_get) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = results[0]
        self.assertTrue(all(r is first for r in results))


class TestCancellationWithRealThread(unittest.TestCase):
    """Integration: cancel() via TaskSupervisor reaches a real worker thread."""

    def setUp(self):
        """Create a fresh supervisor before each test."""
        self.sup = _fresh()

    def tearDown(self):
        """Reset the singleton so later tests start clean."""
        TaskSupervisor.reset_instance()

    def test_real_thread_stops_when_cancelled(self):
        """Scenario: a long-running worker thread is cancelled through the supervisor.
        Expected: the thread stops early and never runs to completion.
        """
        stop_event = threading.Event()
        completed_normally = []

        class _Worker(threading.Thread):
            """Worker thread that can be asked to stop."""
            def request_stop(self):
                """Stop request hook called by the supervisor on cancel."""
                stop_event.set()

            def run(self):
                """Loop for up to about two seconds, returning early if a stop was requested."""
                for _ in range(200):
                    if stop_event.is_set():
                        return
                    time.sleep(0.01)
                completed_normally.append(True)

        worker = _Worker()
        tid = self.sup.register("worker", cancellable=worker)
        worker.start()

        # Give the thread a moment to start, then cancel
        time.sleep(0.05)
        self.sup.cancel(tid)
        worker.join(timeout=3.0)

        self.assertFalse(worker.is_alive(), "worker thread should have stopped")
        self.assertEqual(completed_normally, [], "worker must not have run to completion")
        self.sup.finish(tid)

    def test_finish_after_thread_completes(self):
        """Scenario: a short worker thread finishes by itself and the task is then finished.
        Expected: the thread ran and the status is FINISHED.
        """
        ran = []

        class _Worker(threading.Thread):
            """Worker thread that can be asked to stop."""
            def request_stop(self):
                """Stop request hook called by the supervisor on cancel."""
                pass

            def run(self):
                """Sleep briefly and record that the thread ran."""
                time.sleep(0.02)
                ran.append(True)

        worker = _Worker()
        tid = self.sup.register("worker", cancellable=worker)
        worker.start()
        worker.join()
        self.sup.finish(tid)

        self.assertEqual(ran, [True])
        self.assertEqual(self.sup.status(tid), TaskStatus.FINISHED)


class TestModelLoaderGuard(unittest.TestCase):
    """ModelLoader.is_instance_locked() returns True while generation is active."""

    def test_locked_during_active_count(self):
        """Scenario: a generation is marked as active on the model loader.
        Expected: is_instance_locked() returns True; the original class state is restored afterwards.
        """
        from app.engine.model_loader import ModelLoader
        # Save original state
        orig_count = ModelLoader._active_generation_count
        orig_locked = ModelLoader._instance_locked
        try:
            ModelLoader._active_generation_count = 1
            ModelLoader._instance_locked = True
            self.assertTrue(ModelLoader.is_instance_locked())
        finally:
            ModelLoader._active_generation_count = orig_count
            ModelLoader._instance_locked = orig_locked

    def test_unlocked_when_no_active(self):
        """Scenario: no generation is active.
        Expected: is_instance_locked() returns False; the original class state is restored afterwards.
        """
        from app.engine.model_loader import ModelLoader
        orig_count = ModelLoader._active_generation_count
        orig_locked = ModelLoader._instance_locked
        try:
            ModelLoader._active_generation_count = 0
            ModelLoader._instance_locked = False
            self.assertFalse(ModelLoader.is_instance_locked())
        finally:
            ModelLoader._active_generation_count = orig_count
            ModelLoader._instance_locked = orig_locked


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
