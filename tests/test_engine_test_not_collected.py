"""Regression: the model-loading smoke script must not be collected by pytest."""
import ast
import pathlib


def test_marked_not_a_test():
    src = pathlib.Path("engine_test.py").read_text(encoding="utf-8")
    assert "test_introspection_engine.__test__ = False" in src
    ast.parse(src)
