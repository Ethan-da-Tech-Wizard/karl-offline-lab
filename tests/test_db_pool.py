# ==============================================================================
# Program: test_db_pool.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Tests for SQLiteConnectionPool — thread-safety, WAL configuration, and ACID safety under concurrent writes.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""
Tests for SQLiteConnectionPool — thread-safety, WAL configuration, and ACID safety
under concurrent writes.
"""

# What is under test: app/utils/db_pool.py, the SQLite connection pool shared by the trace logger, session repository and RAG
# metadata store. Properties pinned down: every connection is configured for WAL + synchronous=NORMAL, connections are
# handed back after use, the pool size caps concurrency, and concurrent writers neither lose rows nor see "database is locked".
# Each test creates its own database file under pytest's tmp_path so tests are independent and leave nothing behind.
# Run: python -m pytest tests/test_db_pool.py -q
import sqlite3
import threading
import time

import pytest

from app.utils.db_pool import SQLiteConnectionPool


# ── Helpers ───────────────────────────────────────────────────────────────────

# Uses a plain sqlite3 connection (not the pool) so table creation does not depend on the code under test.
def _create_table(db_path: str) -> None:
    """Create the small test table in the SQLite file."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS test_rows (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            value     TEXT,
            thread_id INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def _row_count(db_path: str) -> int:
    """Count rows in the test table using a fresh connection."""
    # timeout=5.0 lets the verification read wait for any in-flight writer instead of failing with "database is locked".
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        return conn.execute("SELECT COUNT(*) FROM test_rows").fetchone()[0]
    finally:
        conn.close()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_connections_use_wal_journal_mode(tmp_path):
    """Every connection dispensed by the pool must use WAL journal mode."""
    db_path = str(tmp_path / "wal.db")
    pool = SQLiteConnectionPool(db_path, pool_size=3)
    try:
        for _ in range(3):
            with pool.get_connection() as conn:
                # WAL lets readers and one writer proceed concurrently; PRAGMA returns the lower-case mode name.
                mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
                assert mode == "wal", f"Expected WAL, got {mode!r}"
    finally:
        pool.close_all()


def test_connections_use_normal_synchronous(tmp_path):
    """All pool connections should be configured with synchronous=NORMAL (1)."""
    db_path = str(tmp_path / "sync.db")
    pool = SQLiteConnectionPool(db_path, pool_size=2)
    try:
        with pool.get_connection() as conn:
            # NORMAL (value 1) is safe with WAL and much faster than FULL; the numeric value is what SQLite reports.
            sync = conn.execute("PRAGMA synchronous;").fetchone()[0]
            assert sync == 1, f"Expected synchronous=NORMAL (1), got {sync}"
    finally:
        pool.close_all()


def test_connection_is_returned_to_pool_after_context_exit(tmp_path):
    """Acquiring and releasing pool_size connections sequentially must not block."""
    db_path = str(tmp_path / "return.db")
    pool_size = 3
    pool = SQLiteConnectionPool(db_path, pool_size=pool_size)
    try:
        # If connections are not returned, this would deadlock after pool_size uses.
        # Looping twice the pool size proves release works: if a connection leaked, the (pool_size+1)-th acquire would block forever.
        for i in range(pool_size * 2):
            with pool.get_connection() as conn:
                assert conn is not None
    finally:
        pool.close_all()


def test_concurrent_inserts_complete_without_lock_errors(tmp_path):
    """
    10 threads each insert rows simultaneously.
    All must finish within 5 seconds and none may raise 'database is locked'.
    """
    db_path = str(tmp_path / "concurrent.db")
    _create_table(db_path)

    # pool_size equals thread_count so the test measures lock contention at the database, not queueing in the pool.
    pool = SQLiteConnectionPool(db_path, pool_size=10)
    thread_count = 10
    rows_per_thread = 5

    # Errors and completions are collected in shared lists guarded by `lock`, because worker threads cannot raise into the test.
    errors: list[str] = []
    completed: list[int] = []
    lock = threading.Lock()

    def worker(thread_id: int) -> None:
        """Insert rows from one thread through the pool, recording any 'database is locked' errors."""
        try:
            with pool.get_connection() as conn:
                # IMMEDIATE takes the write lock up front, so a conflict shows up as a wait (busy timeout) here rather than as a deadlock at COMMIT.
                conn.execute("BEGIN IMMEDIATE")
                for i in range(rows_per_thread):
                    conn.execute(
                        "INSERT INTO test_rows(value, thread_id) VALUES (?, ?)",
                        (f"v_{thread_id}_{i}", thread_id),
                    )
                conn.commit()
            with lock:
                completed.append(thread_id)
        except Exception as exc:
            with lock:
                errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    start = time.perf_counter()
    for t in threads:
        t.start()

    # The join timeout keeps a hung pool from freezing the test run; the elapsed-time assertion below then fails clearly.
    for t in threads:
        t.join(timeout=5.0)
    elapsed = time.perf_counter() - start

    pool.close_all()

    # Performance guard: 50 tiny inserts should take milliseconds; seconds would mean serialisation or busy-wait loops.
    assert elapsed < 5.0, f"Threads took {elapsed:.2f}s — exceeded 5-second budget"
    lock_errors = [e for e in errors if "database is locked" in e.lower()]
    assert not lock_errors, f"Lock errors: {lock_errors}"
    assert not errors, f"Thread errors: {errors}"
    assert len(completed) == thread_count, (
        f"Only {len(completed)}/{thread_count} threads completed"
    )


def test_final_row_count_matches_expected_acid_safety(tmp_path):
    """
    10 threads × 10 rows = 100 rows total.
    Verifies complete ACID safety: no lost writes, no partial transactions.
    """
    db_path = str(tmp_path / "acid.db")
    _create_table(db_path)

    # pool_size (5) is smaller than the thread count (10) so some threads must wait for a connection, exercising the semaphore path.
    pool = SQLiteConnectionPool(db_path, pool_size=5)
    thread_count = 10
    rows_per_thread = 10
    errors: list[str] = []

    def worker(thread_id: int) -> None:
        """Insert rows from one thread; used to verify the final total and per-thread data."""
        try:
            with pool.get_connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany(
                    "INSERT INTO test_rows(value, thread_id) VALUES (?, ?)",
                    [(f"row_{i}", thread_id) for i in range(rows_per_thread)],
                )
                conn.commit()
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    pool.close_all()

    assert not errors, f"Worker errors: {errors}"

    # Counted through a fresh connection, independent of the pool, so cached state cannot hide lost writes.
    total = _row_count(db_path)
    assert total == thread_count * rows_per_thread, (
        f"Expected {thread_count * rows_per_thread} rows, found {total}"
    )

    conn = sqlite3.connect(db_path)
    try:
        thread_ids = {
            row[0]
            for row in conn.execute("SELECT DISTINCT thread_id FROM test_rows")
        }
    finally:
        conn.close()

    # Every thread's rows are present, proving no worker's transaction was silently dropped.
    assert thread_ids == set(range(thread_count)), (
        f"Missing thread_ids: {set(range(thread_count)) - thread_ids}"
    )


def test_rollback_on_exception_leaves_no_partial_rows(tmp_path):
    """
    A transaction that fails mid-way must roll back entirely — no partial rows.
    """
    db_path = str(tmp_path / "rollback.db")
    _create_table(db_path)

    pool = SQLiteConnectionPool(db_path, pool_size=2)
    try:
        with pytest.raises(ValueError, match="injected"):
            with pool.get_connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO test_rows(value, thread_id) VALUES (?, ?)", ("partial", 99)
                )

                # Rollback is called explicitly before raising; the test then checks the pool's context manager does not commit on error.
                conn.rollback()
                raise ValueError("injected failure")
    finally:
        pool.close_all()

    # A leaked partial row would show up here as a count of 1.
    assert _row_count(db_path) == 0, "Rolled-back row must not appear in the table"


def test_semaphore_gates_concurrent_access_to_pool_size(tmp_path):
    """
    With pool_size=3, at most 3 threads should hold connections simultaneously.
    A 4th thread must wait until one of the 3 releases its connection.
    """
    db_path = str(tmp_path / "sem.db")
    pool_size = 3
    pool = SQLiteConnectionPool(db_path, pool_size=pool_size)

    # The barrier makes all holders overlap in time, which maximises the chance of exceeding pool_size if the semaphore were broken.
    barrier = threading.Barrier(pool_size)

    # Single-element lists are used as mutable counters that the nested function can update without `nonlocal`.
    high_watermark = [0]
    hwm_lock = threading.Lock()
    current_holders = [0]

    def holder() -> None:
        """Hold a pooled connection briefly while tracking how many are in use at once."""
        with pool.get_connection() as _conn:
            with hwm_lock:
                current_holders[0] += 1
                if current_holders[0] > high_watermark[0]:
                    high_watermark[0] = current_holders[0]
            barrier.wait(timeout=5.0)
            with hwm_lock:
                current_holders[0] -= 1

    threads = [threading.Thread(target=holder) for _ in range(pool_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    pool.close_all()

    assert high_watermark[0] <= pool_size, (
        f"High watermark {high_watermark[0]} exceeded pool_size {pool_size}"
    )
