# ==============================================================================
# Program: test_mcp_client.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: MCP Client Manager Tests — Karl Workbench ========================================== Integration tests verifying the lifecycle, tool listing, and execution routines using a local, stdio-based mock Python MCP server.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""
MCP Client Manager Tests — Karl Workbench
==========================================
Integration tests verifying the lifecycle, tool listing, and execution routines
using a local, stdio-based mock Python MCP server.
"""

import tests.qt_test_helper  # noqa: F401

import os
import json
import tempfile
import sys
import unittest
import pytest

from app.engine.mcp_client import MCPClientManager

pytestmark = [pytest.mark.integration]


def _running_under_bwrap() -> bool:
    """True when PID 1 is bubblewrap (sandboxed run where subprocess/IPC may be restricted)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    try:
        # Execute guarded operations and establish recovery boundaries
        with open("/proc/1/comm", "r", encoding="utf-8") as f:
            return f.read().strip() == "bwrap"
    except OSError:
        return False


class TestMCPClient(unittest.TestCase):
    """Integration test for the MCP client manager against a local mock MCP server subprocess (tests/mock_mcp_server.py)."""
    def setUp(self):
        """Write a temporary MCP config file pointing at the mock server run with the current Python interpreter."""
        # Create a temporary config file pointing to mock_mcp_server.py
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "mcp_config.json")
        
        # Determine current python interpreter and absolute mock server path
        python_exe = sys.executable
        mock_server_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "mock_mcp_server.py")
        )
        
        config_data = {
            "mcpServers": {
                "mock_server": {
                    "command": python_exe,
                    "args": [mock_server_path]
                }
            }
        }
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

    def tearDown(self):
        """Delete the temp config and make sure any running MCP client manager is stopped and cleared."""
        # Clean up temporary files
        self.temp_dir.cleanup()
        # Ensure any active client manager instance is stopped and cleared
        MCPClientManager.reset_instance()

    def test_mcp_client_lifecycle_and_tools(self):
        """Full lifecycle: start servers, list the two tools (hello, get_version), call each and check the returned text ('Hello, TestUser!' and '1.0.0'), then shut down leaving no servers."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if _running_under_bwrap():
            pytest.skip("Codex sandbox blocks the mock MCP stdio handshake")

        # Retrieve singleton instance configured with temp config
        manager = MCPClientManager.get_instance(self.config_path)
        
        # 1. Start all configured servers
        manager.start()
        self.assertIn("mock_server", manager.servers)
        
        # 2. List tools across all servers
        tools = manager.list_tools()
        self.assertEqual(len(tools), 2)
        
        tool_names = [t["name"] for t in tools]
        self.assertIn("hello", tool_names)
        self.assertIn("get_version", tool_names)
        
        # 3. Call tool 'hello' with parameters
        res = manager.call_tool("mock_server", "hello", {"name": "TestUser"})
        self.assertFalse(res["is_error"])
        self.assertTrue(len(res["content"]) > 0)
        self.assertEqual(res["content"][0]["text"], "Hello, TestUser!")
        
        # 4. Call tool 'get_version' (no params)
        res = manager.call_tool("mock_server", "get_version", {})
        self.assertFalse(res["is_error"])
        self.assertTrue(len(res["content"]) > 0)
        self.assertEqual(res["content"][0]["text"], "1.0.0")
        
        # 5. Clean shutdown
        manager.stop()
        self.assertEqual(len(manager.servers), 0)


if __name__ == "__main__":
    unittest.main()
