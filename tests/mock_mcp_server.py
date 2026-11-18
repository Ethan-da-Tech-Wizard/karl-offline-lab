# Program: karl-offline-lab - mock_mcp_server.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tiny stdio MCP server with two trivial tools, used by integration tests.
"""
Mock MCP Server for Testing — Karl Workbench
===========================================
A standard stdio-based MCP server using FastMCP. Registers basic tools for integration testing.
"""

from mcp.server.fastmcp import FastMCP

# Create the server instance; tools register on it via the decorator
mcp = FastMCP("MockServer")


@mcp.tool()
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"


@mcp.tool()
def get_version() -> str:
    """Get mock server version."""
    return "1.0.0"


# Run over stdio when launched as a subprocess by a test
if __name__ == "__main__":
    mcp.run()
