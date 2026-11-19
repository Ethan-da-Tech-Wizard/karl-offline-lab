"""Regression: the CPU pre-flight probe script must be valid Python."""
import ast
import pathlib


def _probe():
    # The probe is stored as a string constant inside model_loader.py and run in a subprocess; extracting it from the AST avoids importing model_loader (which needs llama-cpp).
    src = pathlib.Path("app/engine/model_loader.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_CPU_PREFLIGHT_SCRIPT":
            # literal_eval returns the string value without executing anything.
            return ast.literal_eval(node.value)
    raise AssertionError("constant missing")


def test_preflight_script_compiles():
    # A syntax error in the embedded script would only surface at model-load time inside the child process; compiling it here catches that in CI.
    compile(_probe(), "<probe>", "exec")
