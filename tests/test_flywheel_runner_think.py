"""Regression: pre-seeded <think> output that never closes stays thought, not response."""
# Scope: the streaming parser core/cognitive_parser.py splits raw model output into (thought, response). flywheel_runner.py
# pre-seeds the prompt with "<think>\n", so the model's text starts INSIDE a thought. Two regressions are pinned down here.
# Run: python -m pytest tests/test_flywheel_runner_think.py -q
import pathlib

from core.cognitive_parser import parse_thought_stream


# If the model never emits </think>, everything it wrote is still reasoning: the response part must stay empty, otherwise
# unfinished reasoning would be exported as a final answer into the training data.
def test_unclosed_preseeded_think_is_thought():
    assert parse_thought_stream("still reasoning", start_in_thought=True) == ("still reasoning", "")


# Source-text check: flywheel_runner.py must call the parser with start_in_thought=True, otherwise the fix above would not apply.
def test_runner_passes_flag():
    assert "start_in_thought=True" in pathlib.Path("flywheel_runner.py").read_text(encoding="utf-8")
