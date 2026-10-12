"""Regression test for the post-generation glibc malloc_trim() call.

Background: a live stress test on a memory-constrained shared machine showed
this process's RSS climb from ~1.5GB to ~7.5GB over 19 sequential real
generations on the same model (no adapter changes), which went on to get the
process killed outright by the Linux OOM killer once a large-max_tokens
generation pushed system-wide memory over the edge. Investigation found no
leak in Karl's own code -- llama.cpp's buffers are fixed-size, allocated once
at model load -- the growth is glibc allocator arena fragmentation from
per-call token/numpy churn. app/engine/llm_thread.py and
app/engine/agentic_thread.py now call a shared-shape _trim_memory() helper
in their run() method's `finally` block (every exit path: success, error,
circuit-breaker) to ask glibc to release freed arenas back to the OS.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.llm_thread import _trim_memory as llm_trim_memory
from app.engine.agentic_thread import _trim_memory as agentic_trim_memory


def test_trim_memory_is_callable_and_silent_on_both_threads():
    # Must never raise, regardless of platform (best-effort on non-glibc
    # libc such as musl/macOS -- caught internally).
    assert llm_trim_memory() is None
    assert agentic_trim_memory() is None


def test_trim_memory_survives_missing_libc(monkeypatch):
    import ctypes

    def _boom(*_a, **_k):
        raise OSError("no libc.so.6 on this platform")

    monkeypatch.setattr(ctypes, "CDLL", _boom)
    # Should swallow the failure, not propagate it into the generation loop.
    assert llm_trim_memory() is None
    assert agentic_trim_memory() is None


def test_llm_thread_run_finally_calls_trim_memory():
    """The call site matters as much as the function -- assert run()'s
    finally block (every exit path: success, error, circuit-breaker)
    actually invokes it, not just that the helper exists somewhere."""
    import inspect
    import app.engine.llm_thread as llm_thread_module

    source = inspect.getsource(llm_thread_module.LLMThread.run)
    assert "_trim_memory()" in source
    finally_block = source.rsplit("finally:", 1)[-1]
    assert "_trim_memory()" in finally_block


def test_agentic_thread_run_finally_calls_trim_memory():
    import inspect
    import app.engine.agentic_thread as agentic_thread_module

    source = inspect.getsource(agentic_thread_module.AgenticThread.run)
    assert "_trim_memory()" in source
    finally_block = source.rsplit("finally:", 1)[-1]
    assert "_trim_memory()" in finally_block


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
