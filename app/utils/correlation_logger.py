# Program: karl-offline-lab - correlation_logger.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tracks a per-task correlation ID in a context variable and exposes a logging filter that stamps it onto every log record.


"""
Correlation ID context tracking and structured logging filter.

Each asyncio Task and Python thread gets its own copy of the ContextVar,
so correlation IDs are naturally isolated across concurrent operations.
"""

# Postpone evaluation of type annotations (must stay the first statement after the docstring).
from __future__ import annotations

import contextvars
import logging
import uuid

# Module-level ContextVar — default "system" when no ID has been assigned.

correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="system"
)


def set_correlation_id(cid: str) -> contextvars.Token:
    """Bind *cid* to the current context. Returns a Token for reset."""
    return correlation_id.set(cid)


def get_correlation_id() -> str:
    """Return the current context's correlation ID (default: 'system')."""
    return correlation_id.get()


def reset_correlation_id(token: contextvars.Token) -> None:
    """Restore the correlation ID to its value before the paired set() call."""
    correlation_id.reset(token)


def new_correlation_id() -> str:
    """Generate a fresh, short correlation ID (12-hex UUID prefix)."""
    return uuid.uuid4().hex[:12]


class CorrelationFilter(logging.Filter):
    """Logging filter that injects the current correlation_id into every LogRecord.

    Attach to any handler to make ``%(correlation_id)s`` available in format
    strings without the calling code needing to pass extra context.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach the correlation id to the record and let it through."""
        # Stamp the record so log format strings can use the correlation_id field.
        record.correlation_id = get_correlation_id()  # type: ignore[attr-defined]
        return True
