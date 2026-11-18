# Program: karl-offline-lab - test_agent_profile_rpc.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests that the agent profile RPC methods work end to end through the websocket handler.
"""
Dispatch-level tests for the swarm agent-profile RPC endpoints:
get_agent_profiles / save_agent_profile.

Uses the same MockWebSocket/manager pattern as test_websocket_contract.py to
exercise WebSocketServerManager._handler() directly instead of opening a real
socket.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tests.qt_test_helper  # noqa: F401

from app.engine.websocket_server import WebSocketServerManager
from tests.test_swarm_debugger_rpc import MockWebSocket, make_manager


async def _rpc(manager, message: dict) -> dict:
    """Send one JSON-RPC message through the handler using a mock socket and return the parsed reply."""
    # Mock socket preloaded with the single request
    websocket = MockWebSocket([json.dumps(message)])
    # Call the handler directly instead of opening a real socket
    await manager._handler(websocket, path="/?token=test-token")
    assert websocket.sent, "handler produced no response"
    return json.loads(websocket.sent[-1])


class TestAgentProfileRpc(unittest.TestCase):
    """RPC tests for get_agent_profiles and save_agent_profile using a temporary profile file."""
    def setUp(self):
        """Redirect profile storage to a temporary directory so real data is untouched."""
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch(
            "app.utils.swarm_agent_profiles.PROFILE_PATH",
            str(Path(self._tmp.name) / "agent_profiles.json"),
        )
        self._patcher.start()

    def tearDown(self):
        """Stop the patch and delete the temporary directory."""
        self._patcher.stop()
        self._tmp.cleanup()

    def test_get_agent_profiles_returns_builtin_defaults(self):
        """Scenario: request profiles with no saved data. Expected: exactly the architect, coder and tester built-ins are returned."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 1, "method": "get_agent_profiles", "params": {},
        }))
        self.assertIn("result", resp)
        self.assertEqual(set(resp["result"].keys()), {"architect", "coder", "tester"})
        self.assertTrue(resp["result"]["architect"]["builtin"])

    def test_save_agent_profile_persists_custom_role(self):
        """Scenario: save a new custom profile then list profiles. Expected: save reports saved and the new profile appears as a non-built-in."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 2, "method": "save_agent_profile",
            "params": {"profile": {
                "id": "security_reviewer",
                "name": "Security Reviewer",
                "icon": "S",
                "system_prompt": "Audit every diff for injection and auth bypass risks.",
                "temperature": 0.1,
                "context_limit": 2048,
                "tools": {"read_files": True, "write_files": False, "execute_sandbox": False, "query_rag": True},
            }},
        }))
        self.assertIn("result", resp)
        self.assertEqual(resp["result"]["status"], "saved")
        self.assertEqual(resp["result"]["id"], "security_reviewer")

        follow_up = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 3, "method": "get_agent_profiles", "params": {},
        }))
        self.assertIn("security_reviewer", follow_up["result"])
        self.assertEqual(follow_up["result"]["security_reviewer"]["name"], "Security Reviewer")
        self.assertFalse(follow_up["result"]["security_reviewer"]["builtin"])

    def test_save_agent_profile_rejects_missing_profile(self):
        """Scenario: save with no profile parameter. Expected: an invalid-params error (code -32602)."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 4, "method": "save_agent_profile", "params": {},
        }))
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)

    def test_save_agent_profile_rejects_missing_id(self):
        """Scenario: save a profile that has no id. Expected: an error response."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 5, "method": "save_agent_profile",
            "params": {"profile": {"name": "No ID Here"}},
        }))
        self.assertIn("error", resp)

    def test_save_agent_profile_rejects_invalid_id_format(self):
        """Scenario: save a profile whose id has spaces and punctuation. Expected: an error response."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 6, "method": "save_agent_profile",
            "params": {"profile": {"id": "bad id with spaces!", "name": "Bad"}},
        }))
        self.assertIn("error", resp)

    def test_methods_are_registered_and_scoped(self):
        """Scenario: inspect the server method tables. Expected: both profile methods are registered and have permission scopes."""
        self.assertIn("get_agent_profiles", WebSocketServerManager._RPC_METHODS)
        self.assertIn("save_agent_profile", WebSocketServerManager._RPC_METHODS)
        self.assertIn("get_agent_profiles", WebSocketServerManager.METHOD_SCOPES)
        self.assertIn("save_agent_profile", WebSocketServerManager.METHOD_SCOPES)


if __name__ == "__main__":
    unittest.main()
