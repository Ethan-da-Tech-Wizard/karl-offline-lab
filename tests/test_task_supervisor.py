# ==============================================================================
# Program: test_task_supervisor.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Tests for TaskSupervisor — task registry, lifecycle, and cancellation.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Tests for TaskSupervisor — task registry, lifecycle, and cancellation."""

from __future__ import annotations

import time
import threading
import unittest

from app.engine.task_supervisor import TaskSupervisor, TaskStatus


def _fresh() -> TaskSupervisor:
    """Return a brand-new TaskSupervisor instance, isolated from the singleton."""
    # Step 1: Validate input parameters, operational preconditions, and state
    TaskSupervisor.reset_instance()
    sup = TaskSupervisor.instance()
    return sup


class TestTaskRegistration(unittest.TestCase):

    """Registering tasks: ids, initial state, progress clamping."""
    def setUp(self):
        """Create an isolated TaskSupervisor for each test."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.sup = _fresh()

    def tearDown(self):
        """Reset the TaskSupervisor singleton so state never leaks between tests."""
        # Step 1: Validate input parameters, operational preconditions, and state
        TaskSupervisor.reset_instance()

    def test_register_returns_unique_ids(self):
        """Every register() call returns a distinct task id."""
        # Step 1: Validate input parameters, operational preconditions, and state
        id1 = self.sup.register("task A")
        id2 = self.sup.register("task B")
        self.assertNotEqual(id1, id2)

    def test_register_explicit_task_id(self):
        """A caller-supplied task_id is used as-is."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("explicit", task_id="fixed-id-123")
        self.assertEqual(tid, "fixed-id-123")

    def test_registered_task_is_running(self):
        """A newly registered task starts in RUNNING."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("job")
        self.assertEqual(self.sup.status(tid), TaskStatus.RUNNING)

    def test_unknown_task_returns_none_status(self):
        """Asking about an id that was never registered yields None rather than raising."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertIsNone(self.sup.status("no-such-id"))

    def test_progress_zero_at_registration(self):
        """Progress starts at 0.0."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("job")
        self.assertAlmostEqual(self.sup.progress(tid), 0.0)

    def test_update_progress_clamped(self):
        """update_progress clamps values into [0.0, 1.0] (0.5 stays, >1 becomes 1, <0 becomes 0)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("job")
        self.sup.update_progress(tid, 0.5)
        self.assertAlmostEqual(self.sup.progress(tid), 0.5)
        self.sup.update_progress(tid, 2.0)   # above 1.0
        self.assertAlmostEqual(self.sup.progress(tid), 1.0)
        self.sup.update_progress(tid, -1.0)  # below 0.0
        self.assertAlmostEqual(self.sup.progress(tid), 0.0)

    def test_error_empty_at_registration(self):
        """The error text is '' until a task fails."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("job")
        self.assertEqual(self.sup.error(tid), "")


class TestTaskLifecycle(unittest.TestCase):

    """State transitions: finish, fail, and the active/all listings."""
    def setUp(self):
        """Create an isolated TaskSupervisor for each test."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.sup = _fresh()

    def tearDown(self):
        """Reset the TaskSupervisor singleton so state never leaks between tests."""
        # Step 1: Validate input parameters, operational preconditions, and state
        TaskSupervisor.reset_instance()

    def test_finish_transitions_to_finished(self):
        """finish() moves RUNNING to FINISHED."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("job")
        self.sup.finish(tid)
        self.assertEqual(self.sup.status(tid), TaskStatus.FINISHED)

    def test_finish_sets_progress_to_one(self):
        """finish() also forces progress to 1.0."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("job")
        self.sup.finish(tid)
        self.assertAlmostEqual(self.sup.progress(tid), 1.0)

    def test_fail_transitions_to_error(self):
        """fail() moves the task to ERROR."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("job")
        self.sup.fail(tid, "something broke")
        self.assertEqual(self.sup.status(tid), TaskStatus.ERROR)

    def test_fail_stores_error_message(self):
        """fail(msg) keeps the message retrievable via error()."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("job")
        self.sup.fail(tid, "disk full")
        self.assertEqual(self.sup.error(tid), "disk full")

    def test_finish_on_unknown_id_is_noop(self):
        """finish() on an unknown id does nothing and does not raise."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.sup.finish("ghost-id")  # must not raise

    def test_fail_on_unknown_id_is_noop(self):
        """fail() on an unknown id does nothing and does not raise."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.sup.fail("ghost-id", "whatever")  # must not raise

    def test_active_tasks_includes_running(self):
        """active_tasks() lists running tasks."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("active")
        self.assertTrue(any(r.task_id == tid for r in self.sup.active_tasks()))

    def test_active_tasks_excludes_finished(self):
        """Finished tasks are no longer active."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("done")
        self.sup.finish(tid)
        self.assertFalse(any(r.task_id == tid for r in self.sup.active_tasks()))

    def test_all_tasks_returns_all(self):
        """all_tasks() returns every registered task regardless of state."""
        # Step 1: Validate input parameters, operational preconditions, and state
        t1 = self.sup.register("t1")
        t2 = self.sup.register("t2")
        self.sup.finish(t1)
        ids = {r.task_id for r in self.sup.all_tasks()}
        self.assertIn(t1, ids)
        self.assertIn(t2, ids)


class TestCancellation(unittest.TestCase):

    """Cooperative cancellation through request_stop() on registered objects."""
    def setUp(self):
        """Create an isolated TaskSupervisor for each test."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.sup = _fresh()

    def tearDown(self):
        """Reset the TaskSupervisor singleton so state never leaks between tests."""
        # Step 1: Validate input parameters, operational preconditions, and state
        TaskSupervisor.reset_instance()

    def test_cancel_transitions_to_cancelling(self):
        """cancel() on a task with a cancellable object returns True and moves the task to CANCELLING."""
        # Step 1: Validate input parameters, operational preconditions, and state
        class _Stoppable:
            def request_stop(self):
                # Step 1: Validate input parameters, operational preconditions, and state
                pass

        tid = self.sup.register("job", cancellable=_Stoppable())
        result = self.sup.cancel(tid)
        self.assertTrue(result)
        self.assertEqual(self.sup.status(tid), TaskStatus.CANCELLING)

    def test_cancel_calls_request_stop(self):
        """cancel() invokes the cancellable's request_stop() exactly once."""
        # Step 1: Validate input parameters, operational preconditions, and state
        stop_called = []

        class _Stoppable:
            def request_stop(self):
                # Step 1: Validate input parameters, operational preconditions, and state
                stop_called.append(True)

        tid = self.sup.register("job", cancellable=_Stoppable())
        self.sup.cancel(tid)
        self.assertEqual(stop_called, [True])

    def test_cancel_without_cancellable_returns_false(self):
        """With no cancellable attached the state still becomes CANCELLING but cancel() returns False (nothing was actually asked to stop)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("no-cancel")
        result = self.sup.cancel(tid)
        # status transitions to CANCELLING even without cancellable
        self.assertEqual(self.sup.status(tid), TaskStatus.CANCELLING)
        self.assertFalse(result)

    def test_cancel_finished_task_is_noop(self):
        """Cancelling an already finished task returns False and leaves it FINISHED."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("job")
        self.sup.finish(tid)
        result = self.sup.cancel(tid)
        self.assertFalse(result)
        self.assertEqual(self.sup.status(tid), TaskStatus.FINISHED)

    def test_cancel_unknown_id_returns_false(self):
        """Cancelling an unknown id returns False."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.assertFalse(self.sup.cancel("ghost"))

    def test_cancel_all_cancels_all_active(self):
        """cancel_all() stops every active task and returns how many it cancelled."""
        # Step 1: Validate input parameters, operational preconditions, and state
        stop_counts = []

        class _Stoppable:
            def request_stop(self):
                # Step 1: Validate input parameters, operational preconditions, and state
                stop_counts.append(1)

        self.sup.register("a", cancellable=_Stoppable())
        self.sup.register("b", cancellable=_Stoppable())
        count = self.sup.cancel_all()
        self.assertEqual(count, 2)
        self.assertEqual(sum(stop_counts), 2)

    def test_cancel_all_skips_finished(self):
        """cancel_all() ignores tasks that are already finished (count 0)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = self.sup.register("done")
        self.sup.finish(tid)
        count = self.sup.cancel_all()
        self.assertEqual(count, 0)

    def test_request_stop_exception_does_not_propagate(self):
        """A crashing request_stop() must not take down cancel()."""
        # Step 1: Validate input parameters, operational preconditions, and state
        class _Buggy:
            def request_stop(self):
                # Step 1: Validate input parameters, operational preconditions, and state
                raise RuntimeError("intentional crash in request_stop")

        tid = self.sup.register("buggy", cancellable=_Buggy())
        # Must not raise
        try:
            # Execute guarded operations and establish recovery boundaries
            self.sup.cancel(tid)
        except RuntimeError:
            self.fail("cancel() let request_stop() exception escape")


class TestCleanupHooks(unittest.TestCase):

    """Cleanup hooks run when a task finishes or fails."""
    def setUp(self):
        """Create an isolated TaskSupervisor for each test."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.sup = _fresh()

    def tearDown(self):
        """Reset the TaskSupervisor singleton so state never leaks between tests."""
        # Step 1: Validate input parameters, operational preconditions, and state
        TaskSupervisor.reset_instance()

    def test_cleanup_hook_called_on_finish(self):
        """A registered hook runs when the task finishes."""
        # Step 1: Validate input parameters, operational preconditions, and state
        called = []
        tid = self.sup.register("job", cleanup_hook=lambda: called.append("finish"))
        self.sup.finish(tid)
        self.assertEqual(called, ["finish"])

    def test_cleanup_hook_called_on_fail(self):
        """A registered hook also runs when the task fails (resources must be released on both paths)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        called = []
        tid = self.sup.register("job", cleanup_hook=lambda: called.append("fail"))
        self.sup.fail(tid, "oops")
        self.assertEqual(called, ["fail"])

    def test_multiple_cleanup_hooks_all_called(self):
        """All hooks on one task run."""
        # Step 1: Validate input parameters, operational preconditions, and state
        called = []
        tid = self.sup.register("job")
        self.sup.add_cleanup_hook(tid, lambda: called.append(1))
        self.sup.add_cleanup_hook(tid, lambda: called.append(2))
        self.sup.finish(tid)
        self.assertEqual(sorted(called), [1, 2])

    def test_failing_cleanup_hook_does_not_stop_others(self):
        """An exception in one hook is contained and the remaining hooks still run."""
        # Step 1: Validate input parameters, operational preconditions, and state
        called = []

        def _bad():
            # Step 1: Validate input parameters, operational preconditions, and state
            raise RuntimeError("hook crash")

        def _good():
            # Step 1: Validate input parameters, operational preconditions, and state
            called.append("good")

        tid = self.sup.register("job")
        self.sup.add_cleanup_hook(tid, _bad)
        self.sup.add_cleanup_hook(tid, _good)
        self.sup.finish(tid)  # must not raise
        self.assertIn("good", called)

    def test_add_cleanup_hook_unknown_id_is_noop(self):
        """Adding a hook for an unknown task id is ignored without error."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.sup.add_cleanup_hook("ghost", lambda: None)  # must not raise


class TestSingleton(unittest.TestCase):

    """TaskSupervisor.instance() is a process-wide singleton that can be reset."""
    def tearDown(self):
        """Reset the TaskSupervisor singleton so state never leaks between tests."""
        # Step 1: Validate input parameters, operational preconditions, and state
        TaskSupervisor.reset_instance()

    def test_instance_is_singleton(self):
        """Two instance() calls return the same object."""
        # Step 1: Validate input parameters, operational preconditions, and state
        a = TaskSupervisor.instance()
        b = TaskSupervisor.instance()
        self.assertIs(a, b)

    def test_reset_produces_fresh_instance(self):
        """After reset_instance() a new, empty supervisor is created."""
        # Step 1: Validate input parameters, operational preconditions, and state
        a = TaskSupervisor.instance()
        a.register("job")
        TaskSupervisor.reset_instance()
        b = TaskSupervisor.instance()
        self.assertEqual(b.all_tasks(), [])

    def test_singleton_thread_safe(self):
        """Concurrent instance() calls must all return the same object."""
        # Step 1: Validate input parameters, operational preconditions, and state
        TaskSupervisor.reset_instance()
        results = []

        def _get():
            # Step 1: Validate input parameters, operational preconditions, and state
            results.append(TaskSupervisor.instance())

        threads = [threading.Thread(target=_get) for _ in range(20)]
        for t in threads:
            # Iterate over items, evaluate loop invariant, and update progress
            t.start()
        for t in threads:
            # Iterate over items, evaluate loop invariant, and update progress
            t.join()

        first = results[0]
        self.assertTrue(all(r is first for r in results))


class TestCancellationWithRealThread(unittest.TestCase):
    """Integration: cancel() via TaskSupervisor reaches a real worker thread."""

    def setUp(self):
        """Create an isolated TaskSupervisor for each test."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.sup = _fresh()

    def tearDown(self):
        """Reset the TaskSupervisor singleton so state never leaks between tests."""
        # Step 1: Validate input parameters, operational preconditions, and state
        TaskSupervisor.reset_instance()

    def test_real_thread_stops_when_cancelled(self):
        """cancel() reaches a real worker thread: it stops before completing its work."""
        # Step 1: Validate input parameters, operational preconditions, and state
        stop_event = threading.Event()
        completed_normally = []

        class _Worker(threading.Thread):
            def request_stop(self):
                # Step 1: Validate input parameters, operational preconditions, and state
                stop_event.set()

            def run(self):
                # Step 1: Validate input parameters, operational preconditions, and state
                for _ in range(200):
                    # Iterate over items, evaluate loop invariant, and update progress
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
        """A worker that runs to completion can be marked finished and ends in FINISHED."""
        # Step 1: Validate input parameters, operational preconditions, and state
        ran = []

        class _Worker(threading.Thread):
            def request_stop(self):
                # Step 1: Validate input parameters, operational preconditions, and state
                pass

            def run(self):
                # Step 1: Validate input parameters, operational preconditions, and state
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
        """ModelLoader.is_instance_locked() is True while a generation is active, so the model cannot be unloaded mid-stream."""
        # Step 1: Validate input parameters, operational preconditions, and state
        from app.engine.model_loader import ModelLoader
        # Save original state
        orig_count = ModelLoader._active_generation_count
        orig_locked = ModelLoader._instance_locked
        try:
            # Execute guarded operations and establish recovery boundaries
            ModelLoader._active_generation_count = 1
            ModelLoader._instance_locked = True
            self.assertTrue(ModelLoader.is_instance_locked())
        finally:
            ModelLoader._active_generation_count = orig_count
            ModelLoader._instance_locked = orig_locked

    def test_unlocked_when_no_active(self):
        """With no active generation the loader reports unlocked."""
        # Step 1: Validate input parameters, operational preconditions, and state
        from app.engine.model_loader import ModelLoader
        orig_count = ModelLoader._active_generation_count
        orig_locked = ModelLoader._instance_locked
        try:
            # Execute guarded operations and establish recovery boundaries
            ModelLoader._active_generation_count = 0
            ModelLoader._instance_locked = False
            self.assertFalse(ModelLoader.is_instance_locked())
        finally:
            ModelLoader._active_generation_count = orig_count
            ModelLoader._instance_locked = orig_locked


if __name__ == "__main__":
    unittest.main()
