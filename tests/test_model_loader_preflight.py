"""Regression: the CPU pre-flight probe script must be valid Python."""
import ast
import pathlib


def _probe():
    src = pathlib.Path("app/engine/model_loader.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_CPU_PREFLIGHT_SCRIPT":
            return ast.literal_eval(node.value)
    raise AssertionError("constant missing")


def test_preflight_script_compiles():
    compile(_probe(), "<probe>", "exec")
