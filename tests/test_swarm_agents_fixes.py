"""Regression tests for swarm_agents tool cap and prompt/parser consistency.

The module needs heavy optional deps, so the pure tool function is exec'd from source.
"""
# Scope: guards for app/engine/swarm_agents.py. The module needs heavy optional dependencies (PyQt, llama-cpp), so individual
# functions are extracted from the source with ast and exec'd, and some checks read the source text. Covered: the workspace
# grep tool has a total output cap, the prompt text contains no known typo, and the tool-call parser matches the quote styles
# the prompts tell the model to use. Run: python -m pytest tests/test_swarm_agents_fixes.py -q
import ast
import pathlib

# Read once at import time; relative path, so run pytest from the repository root.
SRC = pathlib.Path("app/engine/swarm_agents.py").read_text(encoding="utf-8")


# Compiles one top-level function in isolation. Decorators are cleared because they may reference names that only exist when
# the whole module is imported.
def _load(name):
    fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    ns = {}
    exec(compile(ast.Module([fn], []), "sa", "exec"), ns)
    return ns[name]


# Scenario: 5 files x 30 matching lines each = 150 matches.
# Expected: the tool returns exactly 40 lines, a total cap across all files (not per file), so a broad pattern cannot flood the prompt.
def test_grep_workspace_total_cap(tmp_path):
    for n in range(5):
        (tmp_path / f"f{n}.py").write_text("hit\n" * 30)
    out = _load("_tool_grep_workspace")(str(tmp_path), {"pattern": "hit"})
    assert len(out.splitlines()) == 40


# Source-text guards: a past typo ("tool_call_call") must not return, and the exact Python traceback marker
# that the failure-trace extraction code looks for must stay in the source.
def test_prompt_and_traceback_markers():
    assert "tool_call_call" not in SRC
    assert "Traceback (most recent call last):" in SRC


def _tool_call_pattern():
    """Pull the tool-call regex literal out of the source (the module itself needs heavy deps)."""
    # Finds the first call whose first argument is a string starting with "<tool_call name=": that is the findall regex in the agent loop.
    for node in ast.walk(ast.parse(SRC)):
        if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str) and node.args[0].value.startswith("<tool_call name=")):
            return node.args[0].value
    raise AssertionError("tool_call findall pattern not found")


# Regression for the mismatch between core/default_prompts.py (double quotes in the coder prompt) and the parser (once single quotes only).
def test_tool_call_parser_accepts_single_and_double_quotes():
    """default_prompts.py tells the coder to write name="write_file"; the parser must accept it."""
    # Local import: only this test needs it.
    import re
    pat = _tool_call_pattern()
    single = "<tool_call name='write_file'>path: a.py</tool_call>"
    double = '<tool_call name="write_file">path: a.py</tool_call>'
    assert re.findall(pat, single, re.DOTALL) == [("write_file", "path: a.py")]

    # DOTALL lets the body span lines, as real write_file payloads do.
    assert re.findall(pat, double, re.DOTALL) == [("write_file", "path: a.py")]
