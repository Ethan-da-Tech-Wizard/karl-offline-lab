"""Regression: pre-seeded <think> output that never closes stays thought, not response."""
import pathlib

from core.cognitive_parser import parse_thought_stream


def test_unclosed_preseeded_think_is_thought():
    assert parse_thought_stream("still reasoning", start_in_thought=True) == ("still reasoning", "")


def test_runner_passes_flag():
    assert "start_in_thought=True" in pathlib.Path("flywheel_runner.py").read_text(encoding="utf-8")
