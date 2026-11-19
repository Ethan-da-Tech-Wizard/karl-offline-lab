# ==============================================================================
# Program: test_machine_speak_fallback.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Regression tests for the Machine Speak default and its decompaction fallback path.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""Regression tests for the Machine Speak default and its decompaction
fallback path.

Background: core/interaction_loop.py can rewrite chat history into a dense
"karl.trace.compact.v1" JSON blob before it reaches the model (token-saving
"Machine Speak" mode). A small/weak model asked to reply in that format can
produce text that looks JSON-complete (balanced braces) but fails strict
json.loads() -- e.g. unescaped quotes inside a prose value. Before this fix,
app/engine/llm_thread.py and app/engine/agentic_thread.py silently fed that
raw, still-JSON-wrapped text straight through as the visible chat response.
These tests cover: the setting now defaults off, and a failed decompaction
salvages readable text instead of leaking raw JSON -- while genuinely
unrelated '{'-prefixed model output (e.g. real requested JSON) is untouched.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine import config_store
from app.engine.llm_thread import _emit_compacted_generation as llm_emit
from app.engine.llm_thread import _salvage_compact_trace_fields as llm_salvage
from app.engine.agentic_thread import _emit_compacted_generation as agentic_emit
from app.engine.agentic_thread import _salvage_compact_trace_fields as agentic_salvage
from app.engine.streaming_parser import StreamingThoughtParser


def _make_parser():
    """Build a StreamingThoughtParser that starts inside a thought and collects thought and chat callbacks into lists (events disabled); returns (parser, thoughts, chats)."""
    thoughts, chats = [], []
    parser = StreamingThoughtParser(
        in_thought=True,
        thought_cb=thoughts.append,
        chat_cb=chats.append,
        publish_events=False,
    )
    return parser, thoughts, chats


def test_default_is_off_in_config_store():
    """Machine Speak (compact trace output) defaults to off in the UI config, since small models often produce malformed compact JSON."""
    assert config_store.UI_CONFIG_DEFAULTS["enable_machine_speak"] is False


def test_well_formed_compact_trace_still_decompacts_normally():
    """A well-formed compact trace is still decompacted into the thought and the chat answer (happy path unaffected by the fallback changes)."""
    # Sanity check the happy path is unaffected by the fallback changes.
    buffer = (
        '{"$":"karl.trace.compact.v1","d":{"tk":"reasoning here",'
        '"r":"the actual answer"}}'
    )
    parser, thoughts, chats = _make_parser()
    assert llm_emit(buffer, parser) is True
    assert chats == ["the actual answer"]
    assert thoughts == ["reasoning here"]


def test_malformed_compact_trace_salvages_response_instead_of_leaking_json():
    """A compact trace that is invalid JSON (e.g. an unescaped quote) but brace-balanced has its response salvaged, so the user sees the answer rather than raw 'karl.trace.compact.v1' JSON."""
    # Unescaped double quote inside the "r" value breaks json.loads() even
    # though the buffer is brace-balanced -- this is the exact shape of
    # output that reproduced the bug against a live 1.5B model.
    buffer = (
        '{"$":"karl.trace.compact.v1","d":{"p":"hi there, Karl. '
        'I am a "helpful" assistant","r":"Paris is the capital of France."}}'
    )
    parser, thoughts, chats = _make_parser()
    result = llm_emit(buffer, parser)
    assert result is True
    assert chats == ["Paris is the capital of France."]
    # The raw JSON scaffolding must never show up as chat content.
    assert not any("karl.trace.compact.v1" in c for c in chats)


def test_malformed_compact_trace_with_no_salvageable_fields_falls_back():
    """When nothing can be salvaged the emitter returns False without emitting; the caller then passes the raw text through."""
    buffer = '{"$":"karl.trace.compact.v1","d":{"p":"broken \\ unterminated'
    parser, thoughts, chats = _make_parser()
    assert llm_emit(buffer, parser) is False
    # Caller is responsible for the raw passthrough on a False return;
    # this function itself must not have emitted anything.
    assert chats == []
    assert thoughts == []


def test_unrelated_json_output_is_not_treated_as_a_compact_trace():
    """Ordinary JSON that the model was asked to produce is not mistaken for a failed compact trace; it is left for the normal raw-text path."""
    # A model asked to produce real JSON that happens to start with '{' and
    # isn't valid JSON for some other reason must not be mistaken for a
    # failed compact trace -- no marker, no salvage attempt, clean passthrough
    # signal (False) so the caller's normal raw-text path handles it.
    buffer = '{"name": "incomplete real json"'
    parser, thoughts, chats = _make_parser()
    assert llm_emit(buffer, parser) is False
    assert chats == []
    assert thoughts == []


def test_salvage_helper_handles_escaped_quotes_and_newlines():
    """The salvage helper decodes escaped quotes and newlines in the recovered thought and response."""
    thought, response = llm_salvage(
        '{"$":"karl.trace.compact.v1","d":{"tk":"line one\\nline two",'
        '"r":"she said \\"hi\\" back"}}'
    )
    assert thought == "line one\nline two"
    assert response == 'she said "hi" back'


def test_agentic_thread_mirrors_llm_thread_behavior():
    """AgenticThread has its own copy of the parser fallback and must behave identically to LLMThread (a fix in one must be mirrored in the other)."""
    # AGENTS.md gotcha: the streaming parser (and this fallback) is
    # duplicated across both threads; any fix must land identically in both.
    buffer = (
        '{"$":"karl.trace.compact.v1","d":{"p":"hi there, Karl. '
        'I am a "helpful" assistant","r":"Paris is the capital of France."}}'
    )
    parser, thoughts, chats = _make_parser()
    assert agentic_emit(buffer, parser) is True
    assert chats == ["Paris is the capital of France."]

    assert agentic_salvage(buffer) == llm_salvage(buffer)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
