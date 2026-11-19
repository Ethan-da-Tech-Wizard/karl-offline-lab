"""Regression: parse_tool_calls accepts attribute order/quote variants."""
# Scope: tool_executor.parse_tool_calls turns model output such as <tool_call server='s' name='n'>path: a.txt</tool_call> into
# call dicts for the MCP client. Models write the attributes in either order and with either quote style, so the parser must
# accept all variants. Run: python -m pytest tests/test_tool_executor_parse_attrs.py -q
import pytest

# Skipped when the engine package (and its optional dependencies) cannot be imported.
te = pytest.importorskip("app.engine.tool_executor")


# Three spellings of the same call: server first, name first, and double quotes.
@pytest.mark.parametrize("tag", [
    "<tool_call server='s' name='n'>", "<tool_call name='n' server='s'>",
    '<tool_call name="n" server="s">',
])

# Expected: every variant yields the same single call with the body parsed as "key: value" args.
def test_attr_variants(tag):
    c = te.parse_tool_calls(tag + "path: a.txt</tool_call>")
    assert c == [{"server": "s", "name": "n", "args": {"path": "a.txt"}}]


# A call with only a name (the "done" marker) gets an empty server string; a call with no name at all is dropped, because it
# could not be dispatched anywhere.
def test_name_only_and_nameless():
    assert te.parse_tool_calls("<tool_call name='done'></tool_call>")[0]["server"] == ""
    assert te.parse_tool_calls("<tool_call server='s'>x: 1</tool_call>") == []
