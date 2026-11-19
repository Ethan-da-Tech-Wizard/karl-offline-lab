# ==============================================================================
# Program: test_agentic_loop.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising agentic loop functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import core.agentic_loop as agentic_loop


def test_should_continue_stops_on_explicit_final_answer():
    """The agentic loop stops when the model writes an explicit completion signal such as 'FINAL ANSWER:'."""
    assert not agentic_loop.should_continue(
        1,
        "FINAL ANSWER: The clean final response.",
    )


def test_should_continue_does_not_stop_on_ordinary_summary_phrase():
    """Ordinary phrases like 'in summary' must not stop the loop; only the explicit signals do."""
    assert agentic_loop.should_continue(
        1,
        "In summary, this intermediate pass has identified the main tradeoffs.",
    )
