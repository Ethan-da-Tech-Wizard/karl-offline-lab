# Program: karl-offline-lab - db_pool.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Provides a thread-safe pool of reusable SQLite connections, with a context manager that borrows and safely returns them.

"""
Thread-safe SQLite connection pool for Karl's metadata store.
"""

# Standard-library imports: queue/semaphore for the pool, sqlite3 for connections.
import queue
import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator


class SQLiteConnectionPool:
    """
    Pre-configured pool of SQLite connections pointing to a single database file.

    Each connection is set up with WAL journal mode, NORMAL synchronous writes,
    and a 30-second busy timeout so concurrent threads queue up at the SQLite
    level instead of raising SQLITE_BUSY immediately.
    """

    def __init__(self, db_path: str, pool_size: int = 5) -> None:
        # Remember settings; the semaphore caps how many threads can borrow at once.
        """Create the pool and open pool_size connections to the database file at db_path."""
        self._db_path = db_path
        self._pool_size = pool_size
        self._semaphore = threading.Semaphore(pool_size)
        self._queue: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=pool_size)
        # Fill the queue with ready-to-use connections up front.
        for _ in range(pool_size):
            self._queue.put(self._make_connection())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_connection(self) -> sqlite3.Connection:
        """Open one SQLite connection configured for shared use (WAL mode, row access by name, foreign keys on)."""
        conn = sqlite3.connect(
            self._db_path,
            # 30s wait on locks; autocommit mode; usable across threads because the pool hands them around.
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        # Rows can be read by column name.
        conn.row_factory = sqlite3.Row
        # WAL lets readers and a writer work at once; NORMAL sync trades a little durability for speed.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    # ── Public API ────────────────────────────────────────────────────────────

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Yield a connection from the pool and return it on exit.

        The semaphore limits concurrent holders to pool_size; get_nowait() is
        safe here because the semaphore guarantees a slot is available.

        If the caller's own error handling leaves the connection mid-transaction
        (e.g. its `rollback()` itself raises — a full disk, a vanished WAL file),
        returning it to the pool as-is would poison every future borrower with
        "cannot start a transaction within a transaction". Roll back defensively
        on any exception (a no-op if the caller already succeeded — SQLite only
        issues ROLLBACK when a transaction is actually open) and replace the
        connection outright if that also fails.
        """
        # Wait for a free slot, then take a connection from the queue.
        self._semaphore.acquire()
        conn = self._queue.get_nowait()
        # "broken" marks a connection that could not be rolled back and must be replaced.
        broken = False
        try:
            yield conn
        except Exception:
            try:
                # Caller hit an error: undo any open transaction before re-raising.
                conn.rollback()
            except sqlite3.Error:
                broken = True
            raise
        finally:
            # Always runs: swap out a broken connection, then give the slot back.
            if broken:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                conn = self._make_connection()
            self._queue.put(conn)
            self._semaphore.release()

    def close_all(self) -> None:
        """Close every connection currently in the pool."""
        # Drain the queue, closing each connection, until it is empty.
        while True:
            try:
                conn = self._queue.get_nowait()
                conn.close()
            except queue.Empty:
                break
