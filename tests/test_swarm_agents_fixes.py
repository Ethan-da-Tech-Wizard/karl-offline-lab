"""Regression tests for swarm_agents tool cap and prompt/parser consistency.

The module needs heavy optional deps, so the pure tool function is exec'd from source.
"""
import ast
import pathlib

SRC = pathlib.Path("app/engine/swarm_agents.py").read_text(encoding="utf-8")


def _load(name):
    fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    ns = {}
    exec(compile(ast.Module([fn], []), "sa", "exec"), ns)
    return ns[name]


def test_grep_workspace_total_cap(tmp_path):
    for n in range(5):
        (tmp_path / f"f{n}.py").write_text("hit\n" * 30)
    out = _load("_tool_grep_workspace")(str(tmp_path), {"pattern": "hit"})
    assert len(out.splitlines()) == 40


def test_prompt_and_traceback_markers():
    assert "tool_call_call" not in SRC
    assert "Traceback (most recent call last):" in SRC
