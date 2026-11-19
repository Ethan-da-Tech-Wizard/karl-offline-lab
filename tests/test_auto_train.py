# ==============================================================================
# Program: test_auto_train.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Unit Tests for Karl One-Click Auto-Train Flywheel ================================================= Verifies task generation logic, sandboxed verification helper mocking, PyQt6 background training thread parameters, and WebSocket RPC routing.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""
Unit Tests for Karl One-Click Auto-Train Flywheel
=================================================
Verifies task generation logic, sandboxed verification helper mocking,
PyQt6 background training thread parameters, and WebSocket RPC routing.
"""

import os
import sys
import json
import unittest
import asyncio
from unittest.mock import patch, MagicMock

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.qt_test_helper  # noqa: F401
from PyQt6.QtCore import QCoreApplication
import websockets
import pytest

from auto_train import generate_fallback_tasks, verify_solution
from app.ui.workspaces.training_studio import AutoTrainThread
from app.engine.websocket_server import WebSocketServerManager


def _running_under_bwrap() -> bool:
    """True when PID 1 is bubblewrap (sandboxed run)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    try:
        # Execute guarded operations and establish recovery boundaries
        with open("/proc/1/comm", "r", encoding="utf-8") as f:
            return f.read().strip() == "bwrap"
    except OSError:
        return False


class TestAutoTrain(unittest.TestCase):
    """Tests for the one-click auto-train pipeline helpers and its WebSocket start RPC."""
    def test_generate_fallback_tasks(self):
        """The template fallback produces the requested number of tasks, each with an id, the topic as category, a numbered add_numbers_i problem, a verification script and type 'unit_test'."""
        # Step 1: Validate input parameters, operational preconditions, and state
        topic = "regex_parsing"
        count = 5
        tasks = generate_fallback_tasks(topic, count)
        
        self.assertEqual(len(tasks), count)
        for i, task in enumerate(tasks):
            # Iterate over items, evaluate loop invariant, and update progress
            self.assertIn("id", task)
            self.assertEqual(task["category"], topic)
            self.assertIn(f"add_numbers_{i}", task["problem_statement"])
            self.assertIn("verification_script", task)
            self.assertEqual(task["verification_type"], "unit_test")

    @patch("auto_train.SafePythonSandbox")
    def test_verify_solution_passing(self, mock_sandbox_class):
        """verify_solution returns (True, trace) when the sandbox reports success."""
        # Step 1: Validate input parameters, operational preconditions, and state
        mock_sandbox = MagicMock()
        mock_sandbox.run_code.return_value = (True, "output match")
        mock_sandbox_class.return_value = mock_sandbox

        task = {
            "verification_script": "def verify(response): return True",
            "ground_truth_answer": "mock answer"
        }
        passed, trace = verify_solution(task, "some solution")
        self.assertTrue(passed)
        self.assertEqual(trace, "output match")

    @patch("auto_train.SafePythonSandbox")
    def test_verify_solution_failing(self, mock_sandbox_class):
        """verify_solution returns (False, traceback text) when the sandbox reports failure."""
        # Step 1: Validate input parameters, operational preconditions, and state
        mock_sandbox = MagicMock()
        mock_sandbox.run_code.return_value = (False, "AssertionError: verification failed")
        mock_sandbox_class.return_value = mock_sandbox

        task = {
            "verification_script": "def verify(response): return False",
            "ground_truth_answer": "mock answer"
        }
        passed, trace = verify_solution(task, "failing solution")
        self.assertFalse(passed)
        self.assertEqual(trace, "AssertionError: verification failed")

    def test_auto_train_thread_init(self):
        """AutoTrainThread stores its topic, adapter name and config."""
        # Step 1: Validate input parameters, operational preconditions, and state
        topic = "binary_search"
        adapter_name = "binary_search_specialist"
        config = {
            "count": 10,
            "epochs": 2,
            "lr": 1e-4,
            "rank": 8,
            "alpha": 16,
            "use_qlora": True
        }
        thread = AutoTrainThread(topic, adapter_name, config)
        self.assertEqual(thread.topic, topic)
        self.assertEqual(thread.adapter_name, adapter_name)
        self.assertEqual(thread.config, config)

    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_start_auto_train_rpc(self):
        """The start_auto_train RPC answers 'started' (id 100) and spawns the training subprocess; Popen is mocked so nothing runs."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if _running_under_bwrap():
            self.skipTest("Codex sandbox blocks reliable localhost WebSocket tests")

        # Ensure a QCoreApplication exists (needed for the event loop this
        # WebSocket test relies on) -- the instance itself isn't used directly.
        QCoreApplication.instance() or QCoreApplication(sys.argv)
        port = 8082
        manager = WebSocketServerManager.get_instance(port=port)
        manager.started_event.wait(timeout=5.0)

        # Get token and SSL configuration
        token = manager.bridge_token
        proto = "ws"
        ssl_context = None
        if os.path.exists(manager._SSL_CERT_PATH):
            proto = "wss"
            import ssl
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        uri = f"{proto}://localhost:{port}/?token={token}"

        async def run_client():
            """Async client: connect with the bridge token, send start_auto_train and check the reply."""
            # Step 1: Validate input parameters, operational preconditions, and state
            async with websockets.connect(uri, ssl=ssl_context, close_timeout=2) as ws:
                # Mock Popen inside websocket_server to run a dummy/no-op python command
                with patch("subprocess.Popen") as mock_popen:
                    mock_proc = MagicMock()
                    mock_proc.stdout = []
                    mock_proc.wait.return_value = 0
                    mock_popen.return_value = mock_proc

                    payload = {
                        "jsonrpc": "2.0",
                        "id": 100,
                        "method": "start_auto_train",
                        "params": {
                            "topic": "test_mcp_train",
                            "adapter_name": "test_mcp_adapter",
                            "count": 5,
                            "epochs": 1,
                            "lr": 1e-4
                        }
                    }
                    await ws.send(json.dumps(payload))

                    raw_resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    resp = json.loads(raw_resp)

                    self.assertEqual(resp.get("id"), 100)
                    self.assertIn("result", resp)
                    self.assertEqual(resp["result"]["status"], "started")

        loop = asyncio.new_event_loop()
        try:
            # Execute guarded operations and establish recovery boundaries
            loop.run_until_complete(run_client())
        finally:
            loop.close()
            WebSocketServerManager.reset_instance()


if __name__ == "__main__":
    unittest.main()
