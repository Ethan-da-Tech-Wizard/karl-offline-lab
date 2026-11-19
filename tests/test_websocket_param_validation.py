"""Regression: custom-agent path params and auto-train numbers are validated."""
# Scope: the two pure validation helpers in app/engine/websocket_server.py that protect the bridge. _is_plain_filename guards
# file names that become paths (custom agent model/adapter); _coerce_auto_train_numbers clamps numbers that become argv entries
# of the auto_train.py subprocess. Importing websocket_server needs websockets and PyQt, so the functions are extracted from the
# source with ast and exec'd in isolation. Run: python -m pytest tests/test_websocket_param_validation.py -q
import ast
import os
import pathlib
import types

import pytest


@pytest.fixture(scope="module")
def ws():
    """Exec just the pure helpers so the test needs neither websockets nor PyQt.

    Module-scoped: parsed once and shared by all tests in this file. `ns` provides `os`, the only global the helpers use.
    """
    src = pathlib.Path("app/engine/websocket_server.py").read_text(encoding="utf-8")

    # Only these two top-level functions are compiled; the rest of the module is never executed.
    wanted = {"_is_plain_filename", "_coerce_auto_train_numbers"}
    ns = {"os": os}

    # Compiling each FunctionDef in its own Module keeps the extraction independent of imports at the top of the real file.
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), "ws", "exec"), ns)
    return types.SimpleNamespace(**{k: ns[k] for k in wanted})


# Each rejected value is a distinct attack or mistake: parent traversal, separators (both styles), dot names, empty string, a
# non-string, and an absolute path.
@pytest.mark.parametrize("bad", ["../x", "a/b", "..", ".", "", "a\\b", 5, "/etc/passwd"])
# Pins down: anything that could escape the data/ folder is not a "plain filename".
def test_plain_filename_rejects(ws, bad):
    assert not ws._is_plain_filename(bad)


# A normal model file name must keep working (guards against over-strict rewrites).
def test_plain_filename_accepts(ws):
    assert ws._is_plain_filename("model.gguf")


# Four behaviours in order: in-range values pass through unchanged; huge values are clamped to the maximums (1000 cases, 100 epochs,
# lr 1.0); non-numeric input returns None; NaN and booleans are rejected (True would otherwise pass as 1).
def test_coerce_numbers(ws):
    assert ws._coerce_auto_train_numbers(15, 3, 2e-4) == (15, 3, 2e-4)
    assert ws._coerce_auto_train_numbers(10**9, 10**9, 50) == (1000, 100, 1.0)
    assert ws._coerce_auto_train_numbers("x", 3, 1e-4) is None
    assert ws._coerce_auto_train_numbers(1, 1, float("nan")) is None
    assert ws._coerce_auto_train_numbers(True, 1, 1e-4) is None
