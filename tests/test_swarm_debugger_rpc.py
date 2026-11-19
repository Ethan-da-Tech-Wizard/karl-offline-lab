# ==============================================================================
# Program: test_swarm_debugger_rpc.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Dispatch-level tests for the swarm interactive-debugger RPC endpoints: swarm_pause / swarm_resume / swarm_step / swarm_override_task / swarm_get_history.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""
Dispatch-level tests for the swarm interactive-debugger RPC endpoints:
swarm_pause / swarm_resume / swarm_step / swarm_override_task / swarm_get_history.

Uses the same MockWebSocket/manager pattern as test_websocket_contract.py to
exercise WebSocketServerManager._handler() directly instead of opening a real
socket — real end-to-end socket connections are unreliable in this sandbox
(the same failure reproduces on the pre-existing, unmodified
test_auto_train.py::test_websocket_start_auto_train_rpc).
"""

import asyncio
import json
import threading
import unittest

import tests.qt_test_helper  # noqa: F401

from app.engine.websocket_server import WebSocketServerManager


class MockWebSocket:
    """Minimal async websocket double recording what the server sends."""
    def __init__(self, messages):
        """Store the incoming messages and an empty sent list."""
        self._messages = list(messages)
        self.sent = []
        self.remote_address = ("127.0.0.1", 50000)
        self.closed = False

    def __aiter__(self):
        """Return self so 'async for' works."""
        return self

    async def __anext__(self):
        """Yield the next queued message or stop."""
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def send(self, payload):
        """Record an outgoing payload."""
        self.sent.append(payload)

    async def close(self, code=None, reason=None):
        """Record the close code and reason."""
        self.closed = True
        self.close_code = code
        self.close_reason = reason


def make_manager():
    """Build a WebSocketServerManager with its heavy collaborators mocked."""
    manager = WebSocketServerManager.__new__(WebSocketServerManager)
    manager.port = 19998
    manager.clients = set()
    manager.client_metadata = {}
    manager.client_histories = {}
    manager.loop = None
    manager.server = None
    manager.orchestrator = None
    manager.chat_thread = None
    manager.mini_train_thread = None
    manager.kb_ingest_thread = None
    manager._threads_lock = threading.Lock()
    manager._clients_lock = threading.Lock()
    manager._validate_token = lambda token: True
    return manager


async def _rpc(manager, message: dict) -> dict:
    """Send one JSON-RPC message through the handler and return the decoded response; fails if no response was produced."""
    websocket = MockWebSocket([json.dumps(message)])
    await manager._handler(websocket, path="/?token=test-token")
    assert websocket.sent, "handler produced no response"
    return json.loads(websocket.sent[-1])


class TestSwarmDebuggerRpc(unittest.TestCase):
    """JSON-RPC contract tests for the swarm debugger controls (pause, resume, step, override_task, get_history) when no swarm run is active or arguments are invalid."""
    def test_swarm_pause_without_active_run_returns_error(self):
        """Pausing a run that does not exist returns an error mentioning 'No swarm task'."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 1, "method": "swarm_pause", "params": {"run_id": "no-such-run"},
        }))
        self.assertIn("error", resp)
        self.assertIn("No swarm task", resp["error"]["message"])

    def test_swarm_resume_without_active_run_returns_error(self):
        """Resuming a non-existent run returns an error."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 2, "method": "swarm_resume", "params": {"run_id": "no-such-run"},
        }))
        self.assertIn("error", resp)

    def test_swarm_step_without_active_run_returns_error(self):
        """Stepping a non-existent run returns an error."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 3, "method": "swarm_step", "params": {"run_id": "no-such-run"},
        }))
        self.assertIn("error", resp)

    def test_swarm_override_task_without_active_run_returns_error(self):
        """Overriding a task of a non-existent run returns an error."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 4, "method": "swarm_override_task",
            "params": {"run_id": "no-such-run", "filepath": "a.py", "instructions": "do X"},
        }))
        self.assertIn("error", resp)

    def test_swarm_pause_rejects_missing_run_id(self):
        """swarm_pause without run_id is -32602 Invalid params."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 5, "method": "swarm_pause", "params": {},
        }))
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)

    def test_swarm_override_task_rejects_missing_instructions(self):
        """swarm_override_task without instructions is -32602."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 6, "method": "swarm_override_task",
            "params": {"run_id": "x", "filepath": "a.py"},
        }))
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)

    def test_swarm_get_history_without_run_id_lists_runs(self):
        """swarm_get_history with no run_id lists known runs."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 7, "method": "swarm_get_history", "params": {},
        }))
        self.assertIn("result", resp)
        self.assertIn("runs", resp["result"])
        self.assertIsInstance(resp["result"]["runs"], list)

    def test_swarm_get_history_with_unknown_run_id_returns_empty_steps(self):
        """An unknown run_id gives an empty steps list, not an error."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 8, "method": "swarm_get_history",
            "params": {"run_id": "definitely-not-a-real-run"},
        }))
        self.assertIn("result", resp)
        self.assertEqual(resp["result"]["steps"], [])

    def test_swarm_get_history_rejects_non_string_run_id(self):
        """A non-string run_id is -32602."""
        resp = asyncio.run(_rpc(make_manager(), {
            "jsonrpc": "2.0", "id": 9, "method": "swarm_get_history", "params": {"run_id": 123},
        }))
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32602)


if __name__ == "__main__":
    unittest.main()
