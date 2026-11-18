# Program: karl-offline-lab - conftest.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Shared pytest helpers that skip tests needing optional dependencies or the embedding model.
"""Shared test helpers: skip markers for optional runtime dependencies.

Some tests need resources that are absent in minimal/CI environments:
- GPUtil (optional GPU probe; pure-python but not always installable)
- the sentence-transformers embedding model (downloaded from Hugging Face)

Those tests are skipped — not failed — when the resource is unavailable, so
the suite stays green on machines that lack them while still running fully
on developer machines.
"""

import functools
import importlib.util

import pytest


def module_available(name: str) -> bool:
    """Return True if the named module can be found without importing it."""
    return importlib.util.find_spec(name) is not None


# Skip marker for tests that need the optional GPU probe library
requires_gputil = pytest.mark.skipif(
    not module_available("GPUtil"),
    reason="GPUtil is an optional dependency and is not installed",
)


@functools.lru_cache(maxsize=1)
def embedding_model_available() -> bool:
    """True when the RAG embedding model can actually be loaded (cached or
    downloadable). Cached for the test session — the load attempt is slow."""
    try:
        from sentence_transformers import SentenceTransformer

        # Loading the model is the only reliable availability check (cached or downloadable)
        SentenceTransformer("all-MiniLM-L6-v2")
        return True
    except Exception:
        # Any failure (offline, missing package) means the model is unavailable
        return False


def requires_embedding_model():
    """Build a skip marker that skips a test when the embedding model cannot be loaded."""
    return pytest.mark.skipif(
        not embedding_model_available(),
        reason="sentence-transformers embedding model is unavailable "
        "(offline and not cached)",
    )
