# ==============================================================================
# Program: test_correlation_logging.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Correlation ID logging tests.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""
Correlation ID logging tests.

Covers:
  - Thread-local isolation: two concurrent threads each see only their own CID.
  - CorrelationFilter injects the CID into LogRecord.correlation_id.
  - Default CID is "system" when none has been set.
  - Token-based reset restores the prior value.
  - asyncio task isolation: two tasks don't share CIDs.
  - new_correlation_id() returns a non-empty string of reasonable length.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import List

from app.utils.correlation_logger import (
    CorrelationFilter,
    get_correlation_id,
    new_correlation_id,
    reset_correlation_id,
    set_correlation_id,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _CaptureHandler(logging.Handler):
    """Thread-safe log-record collector."""

    def __init__(self):
        """Set up the lock and the list of captured records."""
        # Step 1: Validate input parameters, operational preconditions, and state
        super().__init__()
        self._lock = threading.Lock()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store each log record under the lock."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._lock:
            self.records.append(record)


def _make_logger(name: str) -> tuple[logging.Logger, _CaptureHandler]:
    """Return a (logger, capture_handler) pair wired with CorrelationFilter."""
    # Step 1: Validate input parameters, operational preconditions, and state
    lgr = logging.getLogger(name)
    lgr.setLevel(logging.DEBUG)
    lgr.propagate = False
    # Remove any handlers left over from previous test runs
    lgr.handlers.clear()
    handler = _CaptureHandler()
    handler.addFilter(CorrelationFilter())
    lgr.addHandler(handler)
    return lgr, handler


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_default_correlation_id_is_system():
    """In a fresh thread with no CID set, get_correlation_id() returns 'system'."""

    # Step 1: Validate input parameters, operational preconditions, and state
    result: list[str] = []

    def _check():
        """Read the correlation id inside the fresh thread."""
        # Step 1: Validate input parameters, operational preconditions, and state
        result.append(get_correlation_id())

    t = threading.Thread(target=_check)
    t.start()
    t.join()

    assert result == ["system"], f"Expected ['system'], got {result}"


def test_concurrent_threads_have_isolated_correlation_ids():
    """
    Thread A and Thread B each set their own CID.
    After a synchronisation barrier they must still see only their own IDs,
    proving ContextVar thread-isolation.  Captured LogRecords must carry the
    correct correlation_id attribute.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    lgr, handler = _make_logger("test.corr.threads")
    barrier = threading.Barrier(2)

    def thread_a():
        """Set id 'task-A', log before and after the barrier."""
        # Step 1: Validate input parameters, operational preconditions, and state
        set_correlation_id("task-A")
        lgr.info("A: before barrier")
        barrier.wait()
        lgr.info("A: after barrier")

    def thread_b():
        """Set id 'task-B', log before and after the barrier."""
        # Step 1: Validate input parameters, operational preconditions, and state
        set_correlation_id("task-B")
        lgr.info("B: before barrier")
        barrier.wait()
        lgr.info("B: after barrier")

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    a_records = [r for r in handler.records if r.correlation_id == "task-A"]
    b_records = [r for r in handler.records if r.correlation_id == "task-B"]

    assert len(a_records) == 2, (
        f"Expected 2 log records from task-A, got {len(a_records)}. "
        f"All CIDs: {[r.correlation_id for r in handler.records]}"
    )
    assert len(b_records) == 2, (
        f"Expected 2 log records from task-B, got {len(b_records)}. "
        f"All CIDs: {[r.correlation_id for r in handler.records]}"
    )
    # Verify exact isolation: no A record carries B's ID and vice-versa
    assert all(r.correlation_id == "task-A" for r in a_records)
    assert all(r.correlation_id == "task-B" for r in b_records)


def test_correlation_filter_injects_attribute():
    """CorrelationFilter must add .correlation_id to every LogRecord."""
    # Step 1: Validate input parameters, operational preconditions, and state
    lgr, handler = _make_logger("test.corr.filter")

    token = set_correlation_id("injected-id")
    try:
        # Execute guarded operations and establish recovery boundaries
        lgr.warning("test record")
    finally:
        reset_correlation_id(token)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert hasattr(record, "correlation_id"), "LogRecord missing correlation_id attribute"
    assert record.correlation_id == "injected-id"


def test_reset_restores_previous_value():
    """reset_correlation_id(token) must restore the value that was active before set()."""
    # Step 1: Validate input parameters, operational preconditions, and state
    outer_token = set_correlation_id("outer")
    try:
        # Execute guarded operations and establish recovery boundaries
        assert get_correlation_id() == "outer"
        inner_token = set_correlation_id("inner")
        try:
            # Execute guarded operations and establish recovery boundaries
            assert get_correlation_id() == "inner"
        finally:
            reset_correlation_id(inner_token)
        assert get_correlation_id() == "outer", (
            "reset_correlation_id did not restore the outer value"
        )
    finally:
        reset_correlation_id(outer_token)


def test_new_correlation_id_returns_nonempty_string():
    """new_correlation_id() must return a non-empty string of plausible length."""
    # Step 1: Validate input parameters, operational preconditions, and state
    cid = new_correlation_id()
    assert isinstance(cid, str)
    assert len(cid) >= 8, f"CID too short: {cid!r}"


def test_log_record_format_includes_correlation_id():
    """The CorrelationFilter enables %(correlation_id)s in a Formatter."""
    # Step 1: Validate input parameters, operational preconditions, and state
    lgr, handler = _make_logger("test.corr.format")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] [Corr: %(correlation_id)s] %(message)s"
    ))

    token = set_correlation_id("format-test")
    try:
        # Execute guarded operations and establish recovery boundaries
        lgr.info("formatted message")
    finally:
        reset_correlation_id(token)

    record = handler.records[0]
    formatted = handler.format(record)
    assert "[Corr: format-test]" in formatted, (
        f"Formatted log line missing correlation prefix: {formatted!r}"
    )


def test_asyncio_tasks_have_isolated_correlation_ids():
    """
    asyncio tasks created via asyncio.create_task() each inherit the context at
    the point of creation — subsequent changes in one task do not affect another.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    results: dict[str, str] = {}

    async def _run():
        """Create two tasks from the same context and wait for both."""
        # Step 1: Validate input parameters, operational preconditions, and state
        async def task_x():
            """Set id 'async-X', yield to the other task, then record the id it sees."""
            # Step 1: Validate input parameters, operational preconditions, and state
            set_correlation_id("async-X")
            await asyncio.sleep(0)   # yield to let task_y run
            results["x"] = get_correlation_id()

        async def task_y():
            """Set id 'async-Y', yield to the other task, then record the id it sees."""
            # Step 1: Validate input parameters, operational preconditions, and state
            set_correlation_id("async-Y")
            await asyncio.sleep(0)
            results["y"] = get_correlation_id()

        # Create both tasks before either runs so they inherit the same base context.
        tx = asyncio.create_task(task_x())
        ty = asyncio.create_task(task_y())
        await asyncio.gather(tx, ty)

    asyncio.run(_run())

    assert results.get("x") == "async-X", (
        f"Task X saw wrong CID: {results.get('x')!r}"
    )
    assert results.get("y") == "async-Y", (
        f"Task Y saw wrong CID: {results.get('y')!r}"
    )


def test_thread_default_is_system_even_after_main_thread_sets_id():
    """
    Setting a CID in the main thread must not bleed into a newly spawned thread,
    which always starts with the default 'system' value.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    outer_token = set_correlation_id("main-thread-id")
    child_result: list[str] = []

    def _child():
        """Record the correlation id seen by a new thread."""
        # Step 1: Validate input parameters, operational preconditions, and state
        child_result.append(get_correlation_id())

    t = threading.Thread(target=_child)
    t.start()
    t.join()
    reset_correlation_id(outer_token)

    assert child_result == ["system"], (
        f"New thread should default to 'system', got {child_result}"
    )
