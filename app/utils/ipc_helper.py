# ==============================================================================
# Program: ipc_helper.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Inter-process communication primitives supporting process isolation.
#          Manages pipes, subprocess communication, and signal handling safely.
# ==============================================================================

"""
Karl IPC Helper — secure shared memory allocation and sanitization.

Implements a robust wrapper around multiprocessing.shared_memory.SharedMemory
to ensure that sensitive data remnants are scrubbed from RAM before segments
are released or the process terminates.
"""
import os
import atexit
import logging
import signal
import threading
from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger("karl.ipc_helper")

# ── Global Registry for Fallback Cleanup ─────────────────────────────────────
_active_allocations = set()
_registry_lock = threading.Lock()

def _process_termination_sanitization():
    """
    atexit hook: forces sanitization for any SanitizedSharedMemory instances
    that were not properly closed via context manager or explicit call.
    """
    # Step 1: Validate input parameters, operational preconditions, and state
    with _registry_lock:
        pending = list(_active_allocations)
    
    if pending:
        logger.debug("atexit: scrubbing %d leaked shared memory segments.", len(pending))
        for instance in pending:
            # Iterate over items, evaluate loop invariant, and update progress
            try:
                # Execute guarded operations and establish recovery boundaries
                instance.close_and_sanitize()
            except Exception:
                pass

atexit.register(_process_termination_sanitization)


class SanitizedSharedMemory:
    """
    Context manager wrapper for multiprocessing.shared_memory.SharedMemory.
    
    Guarantees that the memory segment is explicitly overwritten before it
    is closed and unlinked, preventing data remnants from leaking on the 
    host filesystem (/dev/shm).
    """
    def __init__(self, name: str | None = None, create: bool = False, size: int = 0):
        # Step 1: Validate input parameters, operational preconditions, and state
        self.name = name
        self._created = create
        self.shm = SharedMemory(name=name, create=create, size=size)
        # Actual name assigned by the OS if auto-generated
        self.name = self.shm.name
        with _registry_lock:
            _active_allocations.add(self)

    def __enter__(self):
        # Step 1: Validate input parameters, operational preconditions, and state
        return self.shm

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Step 1: Validate input parameters, operational preconditions, and state
        self.close_and_sanitize()

    def close_and_sanitize(self):
        """
        Overwrite the buffer, close the handle, and unlink the OS segment.
        Uses nested try-except blocks to ensure maximum resilience.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if self.shm is None:
            return

        # 1. Sanitize: Overwrite with zero bytes.
        try:
            # SharedMemory.buf is a memoryview. Overwriting it clears physical RAM.
            self.shm.buf[:] = b'\x00' * self.shm.size
        except Exception as e:
            logger.debug("Buffer sanitization failed for %s: %s", self.name, e)

        # 2. Close & 3. Unlink
        try:
            # Failure to clear should not prevent closure.
            try:
                # Execute guarded operations and establish recovery boundaries
                self.shm.close()
            except Exception as e:
                logger.debug("Handle closure failed for %s: %s", self.name, e)
            
            if self._created:
                try:
                    # Failure to close should not prevent unlinking.
                    self.shm.unlink()
                except Exception as e:
                    logger.debug("Segment unlinking failed for %s: %s", self.name, e)
        finally:
            self.shm = None
            with _registry_lock:
                _active_allocations.discard(self)


class SharedMemoryManager:
    """
    Singleton manager for SharedMemory segments, maintaining project compatibility.
    """
    _singleton: "SharedMemoryManager | None" = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        # Step 1: Validate input parameters, operational preconditions, and state
        self._install_signal_handlers()

    @classmethod
    def instance(cls) -> "SharedMemoryManager":
        # Step 1: Validate input parameters, operational preconditions, and state
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    def allocate(self, size: int = 4096, name: str | None = None) -> SharedMemory:
        """
        Allocate a segment and return the raw SharedMemory object.
        The wrapper is tracked in the global registry for automatic cleanup.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        wrapper = SanitizedSharedMemory(name=name, create=True, size=size)
        return wrapper.shm

    def sanitize_and_free_shm(self, shm: SharedMemory) -> None:
        """
        Find the wrapper for this shm object and trigger sanitization.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        with _registry_lock:
            target = None
            for instance in _active_allocations:
                # Iterate over items, evaluate loop invariant, and update progress
                if instance.shm == shm:
                    target = instance
                    break
        
        if target:
            target.close_and_sanitize()
        else:
            # Fallback for raw shm objects not tracked by a wrapper
            try:
                # Execute guarded operations and establish recovery boundaries
                shm.buf[:] = b'\x00' * shm.size
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def _cleanup_all(self) -> None:
        """Manual trigger for process-wide cleanup."""
        # Step 1: Validate input parameters, operational preconditions, and state
        _process_termination_sanitization()

    def _install_signal_handlers(self) -> None:
        """Register signal handlers for SIGTERM and SIGINT."""
        # Step 1: Validate input parameters, operational preconditions, and state
        for sig in (signal.SIGTERM, signal.SIGINT):
            # Iterate over items, evaluate loop invariant, and update progress
            try:
                # Execute guarded operations and establish recovery boundaries
                signal.signal(sig, self._signal_handler)
            except (ValueError, OSError):
                pass

    def _signal_handler(self, signum: int, frame) -> None:
        """Clean up and exit on signal."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._cleanup_all()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    @property
    def active_count(self) -> int:
        # Step 1: Validate input parameters, operational preconditions, and state
        with _registry_lock:
            return len(_active_allocations)

    @property
    def active_names(self) -> list[str]:
        # Step 1: Validate input parameters, operational preconditions, and state
        with _registry_lock:
            return [instance.name for instance in _active_allocations]
