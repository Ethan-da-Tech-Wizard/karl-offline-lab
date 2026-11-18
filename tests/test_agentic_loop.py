# Program: karl-offline-lab - test_agentic_loop.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests for the stop condition of the agentic loop.
"""Tests for core.agentic_loop.should_continue."""

import core.agentic_loop as agentic_loop


def test_should_continue_stops_on_explicit_final_answer():
    """Scenario: the reply starts with FINAL ANSWER. Expected: the loop stops (should_continue is False)."""
    assert not agentic_loop.should_continue(
        1,
        "FINAL ANSWER: The clean final response.",
    )


def test_should_continue_does_not_stop_on_ordinary_summary_phrase():
    """Scenario: an intermediate reply merely says "In summary". Expected: the loop keeps going."""
    assert agentic_loop.should_continue(
        1,
        "In summary, this intermediate pass has identified the main tradeoffs.",
    )
