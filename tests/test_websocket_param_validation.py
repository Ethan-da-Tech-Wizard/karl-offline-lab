"""Regression: custom-agent path params and auto-train numbers are validated."""
import ast
import os
import pathlib
import types

import pytest


@pytest.fixture(scope="module")
def ws():
    """Exec just the pure helpers so the test needs neither websockets nor PyQt."""
    src = pathlib.Path("app/engine/websocket_server.py").read_text(encoding="utf-8")
    wanted = {"_is_plain_filename", "_coerce_auto_train_numbers"}
    ns = {"os": os}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), "ws", "exec"), ns)
    return types.SimpleNamespace(**{k: ns[k] for k in wanted})


@pytest.mark.parametrize("bad", ["../x", "a/b", "..", ".", "", "a\\b", 5, "/etc/passwd"])
def test_plain_filename_rejects(ws, bad):
    assert not ws._is_plain_filename(bad)


def test_plain_filename_accepts(ws):
    assert ws._is_plain_filename("model.gguf")


def test_coerce_numbers(ws):
    assert ws._coerce_auto_train_numbers(15, 3, 2e-4) == (15, 3, 2e-4)
    assert ws._coerce_auto_train_numbers(10**9, 10**9, 50) == (1000, 100, 1.0)
    assert ws._coerce_auto_train_numbers("x", 3, 1e-4) is None
    assert ws._coerce_auto_train_numbers(1, 1, float("nan")) is None
    assert ws._coerce_auto_train_numbers(True, 1, 1e-4) is None
