# Program: karl-offline-lab - test_cognitive_parser.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests for splitting model output into thought and response text.
"""Tests for core.cognitive_parser.parse_thought_stream."""

import os
import sys

# Make the repository root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_parser import parse_thought_stream

def test_standard_closed():
    """Scenario: one closed think block followed by a reply. Expected: thought and response are split correctly."""
    thought, response = parse_thought_stream("<think>thought content</think>response content")
    assert thought == "thought content"
    assert response == "response content"

def test_unclosed():
    """Scenario: a think block that never closes. Expected: all text is thought and the response is empty."""
    thought, response = parse_thought_stream("<think>unclosed thought content")
    assert thought == "unclosed thought content"
    assert response == ""

def test_mixed_capitalization():
    """Scenario: think tags in mixed case. Expected: parsed the same as lowercase tags."""
    thought, response = parse_thought_stream("<ThInK>thought</tHiNk>response")
    assert thought == "thought"
    assert response == "response"

def test_multiple_blocks():
    """Scenario: two think blocks with replies between. Expected: thoughts and responses are each concatenated in order."""
    thought, response = parse_thought_stream("<think>thought 1</think>response 1<think>thought 2</think>response 2")
    assert thought == "thought 1thought 2"
    assert response == "response 1response 2"

def test_no_blocks():
    """Scenario: plain text with no think tags. Expected: empty thought, whole text is the response."""
    thought, response = parse_thought_stream("just response")
    assert thought == ""
    assert response == "just response"

def test_preseeded_cutoff():
    """Scenario: output starts inside a thought (pre-seeded) and is cut off. Expected: everything is treated as thought."""
    thought, response = parse_thought_stream("unclosed preseeded thought", start_in_thought=True)
    assert thought == "unclosed preseeded thought"
    assert response == ""

if __name__ == "__main__":
    test_standard_closed()
    test_unclosed()
    test_mixed_capitalization()
    test_multiple_blocks()
    test_no_blocks()
    test_preseeded_cutoff()
    print("All cognitive parser unit tests PASSED!")
