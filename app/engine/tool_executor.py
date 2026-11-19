# ==============================================================================
# Program: tool_executor.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Sandboxed tool execution subsystem for local development actions.
#          Executes read, write, grep, git, and shell commands within strict
#          security boundaries.
# ==============================================================================

"""
tool_executor.py — MCP tool call execution for LLMThread.
Provides build_tool_schema_prompt() and execute_tool_call() for the chat loop.
"""
from __future__ import annotations
import logging
import re
from typing import Any

# Logger for tool-call execution problems.
logger = logging.getLogger("karl.tool_executor")


def build_tool_schema_prompt(tools: list[dict]) -> str:
    """Convert MCP tool list to a prompt-injectable XML schema block."""
    if not tools:
        return ""
    # The model is told about tools through a small XML-like block in the prompt (no native function-calling API).
    lines = ["<available_tools>"]
    for t in tools:
        schema = t.get("input_schema", {})
        props = schema.get("properties", {})
        param_strs = [
            f"    {k}: {v.get('description', v.get('type', 'any'))}"
            for k, v in props.items()
        ]
        lines.append(f"  <tool name='{t['name']}' server='{t['server_name']}'>")
        lines.append(f"    {t['description']}")
        if param_strs:
            lines.append("    Parameters:")
            lines.extend(param_strs)
        lines.append("  </tool>")
    lines.append("</available_tools>")
    lines.append(
        # Usage instructions: the model must reply with <tool_call> blocks and finish with the special "done" call.
        "\nTo call a tool: <tool_call server='SERVER' name='TOOL'>\n"
        "  param: value\n"
        "</tool_call>\n"
        "Tool results will be returned to you. Call done() when finished.\n"
        "<tool_call name='done'></tool_call>"
    )
    return "\n".join(lines)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract all <tool_call> blocks from model output."""
    calls = []
    # Match <tool_call server='..' name='..'>body</tool_call>; DOTALL lets bodies span lines.
    # Attributes are parsed separately so server= and name= are accepted in either order and
    # with either quote style; a call with no name cannot be dispatched and is skipped.
    for m in re.finditer(r"<tool_call((?:\s+[a-zA-Z_]+\s*=\s*(?:'[^']*'|\"[^\"]*\"))*)\s*>(.*?)</tool_call>", text, re.DOTALL):
        attrs = {
            a.group(1): a.group(2) if a.group(2) is not None else a.group(3)
            for a in re.finditer(r"([a-zA-Z_]+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")", m.group(1))
        }
        server = attrs.get("server") or ""
        name = attrs.get("name") or ""
        if not name:
            continue
        body = m.group(2).strip()
        # Body lines look like "key: value". Simple keys take one line.
        # Robust YAML-like parser
        args: dict[str, Any] = {}
        lines = body.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s*:(.*)$", line)
            if match:
                key = match.group(1).strip()
                val = match.group(2).strip()
                # Free-form keys (file content, shell commands, code...) swallow the rest of the body
                # so multi-line values and colons inside code are preserved.
                if key in ("content", "command", "code", "text", "script", "query"):
                    # The rest of the tool body belongs to this key
                    remaining_lines = lines[i:]
                    first_line = remaining_lines[0]
                    _, _, first_line_val = first_line.partition(":")
                    remaining_content = [first_line_val] + remaining_lines[1:]
                    raw_val = "\n".join(remaining_content).lstrip("\n")
                    
                    # Models often wrap code in ``` fences; strip them, and tolerate an unclosed final fence (truncated output).
                    # Extract code from markdown block if present
                    raw_val = raw_val.strip()
                    match_block = re.search(r"```(?:[a-zA-Z0-9+#-]+)?\n(.*?)\n```", raw_val, re.DOTALL)
                    if match_block:
                        raw_val = match_block.group(1).strip()
                    else:
                        match_unclosed = re.search(r"```(?:[a-zA-Z0-9+#-]+)?\n(.*)$", raw_val, re.DOTALL)
                        if match_unclosed:
                            raw_val = match_unclosed.group(1).strip()
                            
                    args[key] = raw_val
                    break
                else:
                    args[key] = val
                    i += 1
            else:
                i += 1
        calls.append({"server": server, "name": name, "args": args})
    return calls


def execute_tool_calls(calls: list[dict]) -> list[str]:
    """Execute a list of parsed tool calls via MCPClientManager. Returns result strings."""
    # Lazy import: MCP is optional and heavy.
    from app.engine.mcp_client import MCPClientManager
    results = []
    try:
        manager = MCPClientManager.get_instance()
    except Exception as e:
        return [f"<tool_result>ERROR: MCP manager unavailable: {e}</tool_result>"]

    for call in calls:
        if call["name"] == "done":
            results.append("<tool_result name='done'>OK</tool_result>")
            continue
        try:
            resp = manager.call_tool(call["server"], call["name"], call["args"])
            if resp.get("is_error"):
                text = resp.get("error", "Unknown error")
            else:
                content = resp.get("content", [])
                # Only text content blocks are forwarded; other block types (e.g. images) are dropped.
                text = "\n".join(c.get("text", "") for c in content if "text" in c)
            results.append(f"<tool_result server='{call['server']}' name='{call['name']}'>\n{text}\n</tool_result>")
        except Exception as e:
            results.append(f"<tool_result name='{call['name']}'>ERROR: {e}</tool_result>")
    return results
