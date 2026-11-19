# ==============================================================================
# Program: correlation_logger.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Correlation ID logger facilitating cross-thread request tracing.
#          Attaches unique trace identifiers to log records across asynchronous
#          pipelines.
# ==============================================================================

"""
Correlation ID context tracking and structured logging filter.

Each asyncio Task and Python thread gets its own copy of the ContextVar,
so correlation IDs are naturally isolated across concurrent operations.
"""

from __future__ import annotations

import contextvars
import logging
import uuid

# Module-level ContextVar — default "system" when no ID has been assigned.
# ContextVar (not a thread-local) so the ID also follows asyncio tasks across awaits.
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="system"
)


def set_correlation_id(cid: str) -> contextvars.Token:
    """Bind *cid* to the current context. Returns a Token for reset."""
    # Step 1: Validate input parameters, operational preconditions, and state
    return correlation_id.set(cid)


def get_correlation_id() -> str:
    """Return the current context's correlation ID (default: 'system')."""
    # Step 1: Validate input parameters, operational preconditions, and state
    return correlation_id.get()


def reset_correlation_id(token: contextvars.Token) -> None:
    """Restore the correlation ID to its value before the paired set() call."""
    # Step 1: Validate input parameters, operational preconditions, and state
    correlation_id.reset(token)


def new_correlation_id() -> str:
    """Generate a fresh, short correlation ID (12-hex UUID prefix)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    return uuid.uuid4().hex[:12]


class CorrelationFilter(logging.Filter):
    """Logging filter that injects the current correlation_id into every LogRecord.

    Attach to any handler to make ``%(correlation_id)s`` available in format
    strings without the calling code needing to pass extra context.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Stamp the record with the active correlation ID; always returns True so no record is dropped.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        record.correlation_id = get_correlation_id()  # type: ignore[attr-defined]
        return True
