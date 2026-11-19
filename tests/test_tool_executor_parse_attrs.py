"""Regression: parse_tool_calls accepts attribute order/quote variants."""
import pytest

te = pytest.importorskip("app.engine.tool_executor")


@pytest.mark.parametrize("tag", [
    "<tool_call server='s' name='n'>", "<tool_call name='n' server='s'>",
    '<tool_call name="n" server="s">',
])
def test_attr_variants(tag):
    c = te.parse_tool_calls(tag + "path: a.txt</tool_call>")
    assert c == [{"server": "s", "name": "n", "args": {"path": "a.txt"}}]


def test_name_only_and_nameless():
    assert te.parse_tool_calls("<tool_call name='done'></tool_call>")[0]["server"] == ""
    assert te.parse_tool_calls("<tool_call server='s'>x: 1</tool_call>") == []
