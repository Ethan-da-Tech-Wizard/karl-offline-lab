"""Regression: the model-loading smoke script must not be collected by pytest."""
# Scope: engine_test.py in the repository root is a manual smoke script that loads a real model. pytest's default collection would
# pick up its test_* function and try to run it, so the script marks that function with __test__ = False. This test checks the marker
# is present and that the file still parses. Run: python -m pytest tests/test_engine_test_not_collected.py -q
import ast
import pathlib


# A source-text check (instead of importing engine_test) keeps this test free of any model or Qt dependency.
def test_marked_not_a_test():
    src = pathlib.Path("engine_test.py").read_text(encoding="utf-8")
    assert "test_introspection_engine.__test__ = False" in src
    ast.parse(src)
