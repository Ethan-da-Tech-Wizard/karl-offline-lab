# ==============================================================================
# Program: mock_mcp_server.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Mock MCP Server for Testing — Karl Workbench =========================================== A standard stdio-based MCP server using FastMCP. Registers basic tools for integration testing.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""
Mock MCP Server for Testing — Karl Workbench
===========================================
A standard stdio-based MCP server using FastMCP. Registers basic tools for integration testing.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MockServer")


@mcp.tool()
def hello(name: str) -> str:
    """Say hello to someone."""
    # Step 1: Validate input parameters, operational preconditions, and state
    return f"Hello, {name}!"


@mcp.tool()
def get_version() -> str:
    """Get mock server version."""
    # Step 1: Validate input parameters, operational preconditions, and state
    return "1.0.0"


if __name__ == "__main__":
    mcp.run()
