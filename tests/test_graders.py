# ==============================================================================
# Program: test_graders.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising graders functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import os
import sys
import pytest

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.graders import run_grader, exact_match, json_valid, keyword_hit, groundedness, not_in_context


# ── 1. exact_match ────────────────────────────────────────────────────────────

def test_exact_match_grader():
    """exact_match passes on equal text (ignoring case and surrounding whitespace) with score 1.0 and fails with 0.0 otherwise."""
    # Basic pass case
    res = exact_match("Yes", "yes")
    assert res["passed"] is True
    assert res["score"] == 1.0

    # Spacing and casing pass case
    res = exact_match("  INV-001  \n", "inv-001")
    assert res["passed"] is True
    assert res["score"] == 1.0

    # Fail case
    res = exact_match("No", "yes")
    assert res["passed"] is False
    assert res["score"] == 0.0


# ── 2. json_valid ─────────────────────────────────────────────────────────────

def test_json_valid_grader():
    """json_valid: passes for parseable JSON (with or without markdown fences), dict/list schemas with required keys and empty arrays; fails for malformed JSON, non-object types and missing keys (partial score 0.5 when one of two keys is missing)."""
    # Pass case: basic parseable JSON
    res = json_valid('{"valid": true}')
    assert res["passed"] is True
    assert res["score"] == 1.0

    # Pass case: markdown code blocks
    markdown_json = "```json\n{\n  \"number\": 123\n}\n```"
    res = json_valid(markdown_json)
    assert res["passed"] is True

    # Pass case: valid dict with schema keys
    res = json_valid('{"name": "Karl", "version": "1.4"}', schema_keys=["name", "version"])
    assert res["passed"] is True

    # Pass case: valid list of dicts (checks first item)
    res = json_valid('[{"name": "Karl"}, {"name": "Bob"}]', schema_keys=["name"])
    assert res["passed"] is True

    # Pass case: empty list (treated as valid/no findings)
    res = json_valid("[]", schema_keys=["name"])
    assert res["passed"] is True

    # Fail case: malformed JSON
    res = json_valid('{"invalid": }')
    assert res["passed"] is False
    assert res["score"] == 0.0

    # Fail case: unexpected JSON type (e.g. integer)
    res = json_valid("12345", schema_keys=["name"])
    assert res["passed"] is False

    # Fail case: missing schema keys
    res = json_valid('{"name": "Karl"}', schema_keys=["name", "version"])
    assert res["passed"] is False
    assert res["score"] == 0.5  # 1 of 2 keys present


# ── 3. keyword_hit ────────────────────────────────────────────────────────────

def test_keyword_hit_grader():
    """keyword_hit: require_all needs every keyword, otherwise any one suffices; the score is the fraction of keywords found."""
    # Pass case: require_all=True (all keywords present)
    res = keyword_hit("Karl is an analytical assistant.", keywords=["Karl", "analytical"], require_all=True)
    assert res["passed"] is True
    assert res["score"] == 1.0

    # Pass case: require_all=False (at least one keyword present)
    res = keyword_hit("Karl is here.", keywords=["Karl", "missing"], require_all=False)
    assert res["passed"] is True
    assert res["score"] == 0.5

    # Fail case: require_all=True (missing one keyword)
    res = keyword_hit("Karl is here.", keywords=["Karl", "missing"], require_all=True)
    assert res["passed"] is False
    assert res["score"] == 0.5

    # Fail case: require_all=False (all keywords missing)
    res = keyword_hit("Nothing matches.", keywords=["Karl", "missing"], require_all=False)
    assert res["passed"] is False
    assert res["score"] == 0.0


# ── 4. groundedness ───────────────────────────────────────────────────────────

def test_groundedness_grader():
    """groundedness: a NOT IN CONTEXT refusal passes, answers overlapping the context pass (>=60% of sentences grounded), and answers with no context, no overlap or empty output fail."""
    context = [
        "The standard return policy for electronic items is 15 days.",
        "Meridian Technologies CEO is Sandra Holt."
    ]

    # Pass case: NOT IN CONTEXT refusal
    res = groundedness("This is NOT IN CONTEXT: no policy found", context)
    assert res["passed"] is True
    assert res["score"] == 1.0

    # Pass case: high word overlap with context (>= 60% sentences grounded)
    output_text = "The return policy is 15 days for electronics. Also, Sandra Holt is the CEO of Meridian."
    res = groundedness(output_text, context)
    assert res["passed"] is True
    assert res["score"] == 1.0

    # Fail case: no context chunks
    res = groundedness(output_text, [])
    assert res["passed"] is False
    assert res["score"] == 0.0

    # Fail case: low/no word overlap with context
    res = groundedness("The solar system has eight planets.", context)
    assert res["passed"] is False
    assert res["score"] == 0.0

    # Fail case: empty output
    res = groundedness("", context)
    assert res["passed"] is False


# ── 5. not_in_context ─────────────────────────────────────────────────────────

def test_not_in_context_grader():
    """not_in_context passes only when the output contains the refusal marker."""
    # Pass case: contains refusal marker
    res = not_in_context("I cannot answer because it is not in context.")
    assert res["passed"] is True
    assert res["score"] == 1.0

    # Fail case: model hallucinated/answered
    res = not_in_context("The return policy is 30 days.")
    assert res["passed"] is False
    assert res["score"] == 0.0


# ── 6. Registry Dispatcher ───────────────────────────────────────────────────

def test_grader_dispatch_and_routing():
    """run_grader routes to the named grader and raises KeyError for unknown names."""
    # Test valid dispatch
    res = run_grader("exact_match", "yes", expected="yes")
    assert res["passed"] is True

    # Test unknown grader
    with pytest.raises(KeyError, match="Unknown grader 'invalid_grader'"):
        run_grader("invalid_grader", "output")
