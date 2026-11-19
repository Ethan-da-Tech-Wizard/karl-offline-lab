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


def _tool_call_pattern():
    """Pull the tool-call regex literal out of the source (the module itself needs heavy deps)."""
    for node in ast.walk(ast.parse(SRC)):
        if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str) and node.args[0].value.startswith("<tool_call name=")):
            return node.args[0].value
    raise AssertionError("tool_call findall pattern not found")


def test_tool_call_parser_accepts_single_and_double_quotes():
    """default_prompts.py tells the coder to write name="write_file"; the parser must accept it."""
    import re
    pat = _tool_call_pattern()
    single = "<tool_call name='write_file'>path: a.py</tool_call>"
    double = '<tool_call name="write_file">path: a.py</tool_call>'
    assert re.findall(pat, single, re.DOTALL) == [("write_file", "path: a.py")]
    assert re.findall(pat, double, re.DOTALL) == [("write_file", "path: a.py")]
