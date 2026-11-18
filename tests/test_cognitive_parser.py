# ==============================================================================
# Program: test_cognitive_parser.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising cognitive parser functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_parser import parse_thought_stream

def test_standard_closed():
    # Step 1: Validate input parameters, operational preconditions, and state
    thought, response = parse_thought_stream("<think>thought content</think>response content")
    assert thought == "thought content"
    assert response == "response content"

def test_unclosed():
    # Step 1: Validate input parameters, operational preconditions, and state
    thought, response = parse_thought_stream("<think>unclosed thought content")
    assert thought == "unclosed thought content"
    assert response == ""

def test_mixed_capitalization():
    # Step 1: Validate input parameters, operational preconditions, and state
    thought, response = parse_thought_stream("<ThInK>thought</tHiNk>response")
    assert thought == "thought"
    assert response == "response"

def test_multiple_blocks():
    # Step 1: Validate input parameters, operational preconditions, and state
    thought, response = parse_thought_stream("<think>thought 1</think>response 1<think>thought 2</think>response 2")
    assert thought == "thought 1thought 2"
    assert response == "response 1response 2"

def test_no_blocks():
    # Step 1: Validate input parameters, operational preconditions, and state
    thought, response = parse_thought_stream("just response")
    assert thought == ""
    assert response == "just response"

def test_preseeded_cutoff():
    # Step 1: Validate input parameters, operational preconditions, and state
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
