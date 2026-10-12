"""Regression test for the continuation-loop context-overflow bug.

Background: app/engine/llm_thread.py's and app/engine/agentic_thread.py's
generation loops append raw_output to the prompt on each auto-continuation
pass (when finish_reason == "length") and re-request the model, but kept
requesting the same fixed, originally-configured max_tokens every pass
regardless of how much of the context window the growing prompt had already
consumed. Reproduced live on a 1.5B model / 4096-token context: a long-essay
prompt with a generous max_tokens ran through a few continuation passes and
llama.cpp hard-failed with "Requested tokens (4096) exceed context window of
4096" instead of finishing gracefully. Both threads now clamp the requested
max_tokens to what's actually left via _clamp_max_tokens_to_context(), and
stop the loop cleanly instead of continuing once there's no usable room left.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect

from app.engine.llm_thread import _clamp_max_tokens_to_context as llm_clamp
from app.engine.agentic_thread import _clamp_max_tokens_to_context as agentic_clamp


def test_clamp_returns_configured_value_when_plenty_of_room():
    # 4096-token context, prompt only used 500 -- configured 1024 fits fine.
    assert llm_clamp(1024, 500, 4096) == 1024
    assert agentic_clamp(1024, 500, 4096) == 1024


def test_clamp_shrinks_to_remaining_budget_near_the_edge():
    # This is the exact shape of the reproduced bug: a continuation pass
    # where the prompt has already grown to consume most of the context.
    # 4096 budget, 3200 already used, 8-token safety margin -> 888 left,
    # well under the configured 1024.
    assert llm_clamp(1024, 3200, 4096) == 888
    assert agentic_clamp(1024, 3200, 4096) == 888


def test_clamp_never_goes_negative_when_prompt_already_exceeds_budget():
    # Defensive: even if current_prompt_tokens somehow exceeds the budget
    # (tokenizer counting quirks, off-by-a-few), never return a negative
    # max_tokens -- llama.cpp would reject that outright.
    assert llm_clamp(1024, 4200, 4096) == 0
    assert agentic_clamp(1024, 4200, 4096) == 0


def test_clamp_respects_configured_value_as_a_ceiling():
    # Plenty of room, but the user only asked for 64 tokens -- never hand
    # back more than what was actually configured.
    assert llm_clamp(64, 100, 4096) == 64
    assert agentic_clamp(64, 100, 4096) == 64


def test_clamp_does_not_shrink_a_deliberately_small_configured_value():
    # Regression: a caller that deliberately configures a tiny max_tokens
    # (e.g. tests using max_tokens=8 to run fast, or a real "give me one
    # word" request) must not be treated as "context ran out" just because
    # the value is below the < 16 stop-the-loop threshold -- that threshold
    # only means something when the CONTEXT forced a reduction, not when
    # the request was always going to be small. Plenty of room available
    # here (4096 budget, 4 tokens used) -- the clamp must return exactly
    # what was configured, unreduced.
    assert llm_clamp(8, 4, 4096) == 8
    assert agentic_clamp(8, 4, 4096) == 8


def test_llm_thread_continuation_loop_uses_the_clamp():
    """The call site matters as much as the helper -- assert the
    continuation loop actually uses the clamped value, and stops instead
    of looping once there's no usable room left. Also assert the early-stop
    check is gated on the clamp having actually reduced the value (not just
    on the resulting number being small) -- see
    test_clamp_does_not_shrink_a_deliberately_small_configured_value."""
    import app.engine.llm_thread as llm_thread_module

    source = inspect.getsource(llm_thread_module.LLMThread.run)
    assert "_clamp_max_tokens_to_context(" in source
    assert "max_tokens=effective_max_tokens" in source
    assert "if effective_max_tokens < configured_max_tokens and effective_max_tokens < 16:" in source


def test_agentic_thread_continuation_loop_uses_the_clamp():
    # The continuation loop lives in _run_single_generation(), called from
    # run() once per agentic iteration -- not in run() itself.
    import app.engine.agentic_thread as agentic_thread_module

    source = inspect.getsource(agentic_thread_module.AgenticThread._run_single_generation)
    assert "_clamp_max_tokens_to_context(" in source
    assert "max_tokens=effective_max_tokens" in source
    assert "if effective_max_tokens < configured_max_tokens and effective_max_tokens < 16:" in source


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
