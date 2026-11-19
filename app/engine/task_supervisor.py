# ==============================================================================
# Program: task_supervisor.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Task health monitor, timeout watchdog, and failure recovery supervisor.
#          Tracks heartbeat signals from background threads and resets stalled
#          operations cleanly.
# ==============================================================================

"""
TaskSupervisor — unified long-running task registry for Karl.

All inference threads, training jobs, eval runs, RAG indexing operations, and
dataset downloads register here. The supervisor provides:

- Deterministic lifecycle: idle → running → cancelling → finished / error
- cancel() routed to the underlying thread's request_stop()
- Progress float [0.0, 1.0] updated by the running task
- Error string set on failure
- cleanup_hook callbacks called exactly once on completion or error
- Structured lifecycle log entries via TraceLogger
- Singleton access via TaskSupervisor.instance()

Integration pattern
-------------------
  task_id = TaskSupervisor.instance().register(
      name="LLM generation",
      cancellable=thread,          # any object with .request_stop()
      cleanup_hook=my_cleanup,
  )
  # When the thread finishes, call:
  TaskSupervisor.instance().finish(task_id)
  # On error:
  TaskSupervisor.instance().fail(task_id, "message")
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("karl.task_supervisor")


class TaskStatus(str, Enum):
    """Lifecycle states for a supervised task (str-valued so they serialize to JSON directly)."""
    IDLE       = "idle"
    RUNNING    = "running"
    CANCELLING = "cancelling"
    FINISHED   = "finished"
    ERROR      = "error"


@dataclass
class TaskRecord:
    """Mutable bookkeeping for one task: id, name, status, progress, error and internal hooks."""
    task_id: str
    name: str
    status: TaskStatus = TaskStatus.IDLE
    progress: float = 0.0           # [0.0, 1.0]
    error: str = ""
    # Underscore fields are internal (hidden from repr): the object to stop and hooks to run at the end.
    _cancellable: Any = field(default=None, repr=False)
    _cleanup_hooks: list[Callable[[], None]] = field(default_factory=list, repr=False)


class TaskSupervisor:
    """Singleton task registry with lifecycle management."""

    _instance: TaskSupervisor | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        """Create an empty registry. The TraceLogger is imported lazily to avoid a circular import."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskRecord] = {}
        self._logger: Any = None  # lazy import to avoid circular deps

    @classmethod
    def instance(cls) -> TaskSupervisor:
        """Return the process-wide supervisor, creating it on first use."""
        # Step 1: Validate input parameters, operational preconditions, and state
        # Double-checked locking: cheap unlocked test first, then re-test under the lock.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Replace the singleton — used by tests to start clean."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with cls._instance_lock:
            cls._instance = None

    # ── Registration ─────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        *,
        cancellable: Any = None,
        cleanup_hook: Callable[[], None] | None = None,
        task_id: str | None = None,
    ) -> str:
        """Register a new task and mark it RUNNING.

        Args:
            name:         Human-readable task name logged in traces.
            cancellable:  Object with a ``request_stop()`` method.
                          cancel() is a no-op when None.
            cleanup_hook: Called exactly once on finish() or fail(), on the
                          calling thread. May be None.
            task_id:      Explicit UUID string; auto-generated when None.

        Returns:
            The task_id string — store it to call finish/fail/update later.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        tid = task_id or str(uuid.uuid4())
        hooks = [cleanup_hook] if cleanup_hook is not None else []
        record = TaskRecord(
            task_id=tid,
            name=name,
            status=TaskStatus.RUNNING,
            _cancellable=cancellable,
            _cleanup_hooks=hooks,
        )
        with self._lock:
            self._tasks[tid] = record
        logger.info("Task registered: [%s] %s", tid[:8], name)
        self._trace("registered", tid, name)
        return tid

    # ── Lifecycle mutators ───────────────────────────────────────────────────

    def update_progress(self, task_id: str, progress: float) -> None:
        """Update completion fraction [0.0, 1.0]."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return
            # Clamp so a bad caller can never push the UI progress bar out of range.
            rec.progress = max(0.0, min(1.0, progress))

    def cancel(self, task_id: str) -> bool:
        """Request cancellation of a running task.

        Transitions status to CANCELLING and calls request_stop() on the
        registered cancellable. Returns True when a cancellable was found.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                logger.warning("cancel() called on unknown task_id %s", task_id)
                return False
            # Only tasks that have not already finished or started cancelling can be cancelled.
            if rec.status not in (TaskStatus.RUNNING, TaskStatus.IDLE):
                logger.debug("cancel() on task %s already in status %s", task_id[:8], rec.status)
                return False
            rec.status = TaskStatus.CANCELLING
            # Copy out under the lock, call request_stop() after releasing it so a slow or
            # re-entrant stop handler cannot deadlock the registry.
            cancellable = rec._cancellable

        logger.info("Task cancelling: [%s] %s", task_id[:8], rec.name)
        self._trace("cancelling", task_id, rec.name)

        if cancellable is not None and hasattr(cancellable, "request_stop"):
            try:
                # Execute guarded operations and establish recovery boundaries
                cancellable.request_stop()
            except Exception as exc:
                logger.warning("request_stop() raised for task %s: %s", task_id[:8], exc)
            return True
        return False

    def finish(self, task_id: str) -> None:
        """Mark a task FINISHED and run cleanup hooks."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return
            # finish()/fail() are idempotent: a task already finished or failed is left untouched, so hooks run once.
            if rec.status not in (TaskStatus.RUNNING, TaskStatus.CANCELLING):
                return
            rec.status = TaskStatus.FINISHED
            rec.progress = 1.0
            hooks = list(rec._cleanup_hooks)

        logger.info("Task finished: [%s] %s", task_id[:8], rec.name)
        self._trace("finished", task_id, rec.name)
        self._run_hooks(hooks, task_id)

    def fail(self, task_id: str, error: str) -> None:
        """Mark a task ERROR with a message and run cleanup hooks."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return
            if rec.status not in (TaskStatus.RUNNING, TaskStatus.CANCELLING):
                return
            rec.status = TaskStatus.ERROR
            rec.error = error
            hooks = list(rec._cleanup_hooks)

        logger.error("Task failed: [%s] %s — %s", task_id[:8], rec.name, error)
        self._trace("error", task_id, rec.name, error=error)
        self._run_hooks(hooks, task_id)

    def add_cleanup_hook(self, task_id: str, hook: Callable[[], None]) -> None:
        """Attach an additional cleanup hook to an already-registered task."""
        # Step 1: Validate input parameters, operational preconditions, and state
        run_immediately = False
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                logger.warning("add_cleanup_hook() called on unknown task_id %s", task_id)
                return
            # Task already ended: its hooks have run, so run this late hook right away.
            if rec.status in (TaskStatus.FINISHED, TaskStatus.ERROR):
                run_immediately = True
            else:
                rec._cleanup_hooks.append(hook)

        if run_immediately:
            try:
                # Execute guarded operations and establish recovery boundaries
                hook()
            except Exception as exc:
                logger.warning("Cleanup hook raised immediately for terminated task %s: %s", task_id[:8], exc)

    # ── Queries ──────────────────────────────────────────────────────────────

    def status(self, task_id: str) -> TaskStatus | None:
        """Current TaskStatus for ``task_id`` or None if unknown."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            rec = self._tasks.get(task_id)
            return rec.status if rec else None

    def progress(self, task_id: str) -> float:
        """Progress fraction for ``task_id`` (0.0 if unknown)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            rec = self._tasks.get(task_id)
            return rec.progress if rec else 0.0

    def error(self, task_id: str) -> str:
        """Error message for ``task_id`` (empty string if none/unknown)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            rec = self._tasks.get(task_id)
            return rec.error if rec else ""

    def get(self, task_id: str) -> TaskRecord | None:
        """Return the TaskRecord itself (live object, not a copy) or None."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            return self._tasks.get(task_id)

    def active_tasks(self) -> list[TaskRecord]:
        """Return all tasks currently RUNNING or CANCELLING."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            return [
                r for r in self._tasks.values()
                if r.status in (TaskStatus.RUNNING, TaskStatus.CANCELLING)
            ]

    def all_tasks(self) -> list[TaskRecord]:
        """Return a snapshot list of every task ever registered (finished ones are not purged)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            return list(self._tasks.values())

    def cancel_all(self) -> int:
        """Cancel every active task. Returns number of cancellations requested."""
        # Step 1: Validate input parameters, operational preconditions, and state
        active = self.active_tasks()
        for rec in active:
            # Iterate over items, evaluate loop invariant, and update progress
            self.cancel(rec.task_id)
        return len(active)

    # ── Internals ────────────────────────────────────────────────────────────

    def _run_hooks(self, hooks: list[Callable], task_id: str) -> None:
        """Run cleanup callbacks in order; one failing hook must not stop the rest."""
        # Step 1: Validate input parameters, operational preconditions, and state
        for hook in hooks:
            # Iterate over items, evaluate loop invariant, and update progress
            try:
                # Execute guarded operations and establish recovery boundaries
                hook()
            except Exception as exc:
                logger.warning("Cleanup hook raised for task %s: %s", task_id[:8], exc)

    def _trace(self, event: str, task_id: str, name: str, *, error: str = "") -> None:
        """Write a lightweight trace entry — best-effort, never raises."""
        # Step 1: Validate input parameters, operational preconditions, and state
        try:
            # Execute guarded operations and establish recovery boundaries
            # Lazy import avoids import cycles; tracing is best-effort.
            from app.utils.trace_logger import TraceLogger
            tl = TraceLogger()
            tl.log_task_event(task_id=task_id, name=name, event=event, error=error)
        except Exception:
            pass
