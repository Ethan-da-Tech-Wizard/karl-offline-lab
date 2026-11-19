# ==============================================================================
# Program: test_auto_train.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Unit Tests for Karl One-Click Auto-Train Flywheel ================================================= Verifies task generation logic, sandboxed verification helper mocking, PyQt6 background training thread parameters, and WebSocket RPC routing.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

# Scope: the "one-click auto-train flywheel" (auto_train.py CLI + AutoTrainThread + the
# start_auto_train RPC of the WebSocket bridge). Heavy parts (sandbox, training, subprocess) are
# mocked, so these tests pin interfaces and routing rather than training quality.
"""
Unit Tests for Karl One-Click Auto-Train Flywheel
=================================================
Verifies task generation logic, sandboxed verification helper mocking,
PyQt6 background training thread parameters, and WebSocket RPC routing.
"""

# Standard imports: unittest for the TestCase style, asyncio/json for the RPC client, Path for the
# AST-based CLI check at the bottom, MagicMock/patch to replace the sandbox and subprocess.
import os
import sys
import json
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in path (auto_train.py sits at the repo root, outside any package).
# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Side-effect import: Qt environment must exist before importing UI/engine modules.
import tests.qt_test_helper  # noqa: F401
from PyQt6.QtCore import QCoreApplication
import websockets
import pytest

# generate_fallback_tasks: template tasks used when the LLM synthesiser is unavailable;
# verify_solution: runs a task's verification script in the sandbox.
from auto_train import generate_fallback_tasks, verify_solution
# Imported from the package __init__ to check the re-export that the Training tab relies on.
from app.ui.workspaces.training_studio import AutoTrainThread
from app.engine.websocket_server import WebSocketServerManager


# Sandboxed environments (bwrap) block loopback sockets, so the websocket test is skipped there.
def _running_under_bwrap() -> bool:
    """True when PID 1 is bubblewrap (sandboxed run)."""
    try:
        with open("/proc/1/comm", "r", encoding="utf-8") as f:
            return f.read().strip() == "bwrap"
    except OSError:
        return False


class TestAutoTrain(unittest.TestCase):
    """Tests for the one-click auto-train pipeline helpers and its WebSocket start RPC."""
    # Group 1: pure helpers (no Qt, no network).
    def test_generate_fallback_tasks(self):
        """The template fallback produces the requested number of tasks, each with an id, the topic as category, a numbered add_numbers_i problem, a verification script and type 'unit_test'."""
        # Fallback tasks keep the pipeline usable without a model for task synthesis: deterministic text,
        # one verification script per task, so tests and demos still train something.
        topic = "regex_parsing"
        count = 5
        # Request 5 tasks for a topic so the count and the per-task index in the problem name can be checked.
        tasks = generate_fallback_tasks(topic, count)
        
        self.assertEqual(len(tasks), count)
        for i, task in enumerate(tasks):
            # id is required downstream (dataset records are keyed by it); category carries the user's topic.
            self.assertIn("id", task)
            self.assertEqual(task["category"], topic)
            self.assertIn(f"add_numbers_{i}", task["problem_statement"])
            self.assertIn("verification_script", task)
            self.assertEqual(task["verification_type"], "unit_test")

    # SafePythonSandbox is patched where auto_train imported it, so no code is executed for real.
    # These two tests pin the contract: the sandbox's (success, output) pair is returned unchanged.
    @patch("auto_train.SafePythonSandbox")
    def test_verify_solution_passing(self, mock_sandbox_class):
        """verify_solution returns (True, trace) when the sandbox reports success."""
        # run_code returns (ok, trace); "output match" stands in for the success trace.
        mock_sandbox = MagicMock()
        mock_sandbox.run_code.return_value = (True, "output match")
        mock_sandbox_class.return_value = mock_sandbox

        # Minimal task dict: only the keys verify_solution reads.
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

    # The thread must store its inputs verbatim; run() would later forward them to the subprocess as CLI flags.
    def test_auto_train_thread_init(self):
        """AutoTrainThread stores its topic, adapter name and config."""
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
        # Constructing the thread starts nothing: no process is spawned until run().
        thread = AutoTrainThread(topic, adapter_name, config)
        self.assertEqual(thread.topic, topic)
        self.assertEqual(thread.adapter_name, adapter_name)
        self.assertEqual(thread.config, config)

    # Integration group: markers let CI skip tests that need a local TCP server.
    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_start_auto_train_rpc(self):
        # Id 100 is an arbitrary request id used to prove the reply is matched to this request.
        """The start_auto_train RPC answers 'started' (id 100) and spawns the training subprocess; Popen is mocked so nothing runs."""
        if _running_under_bwrap():
            self.skipTest("Codex sandbox blocks reliable localhost WebSocket tests")

        # Ensure a QCoreApplication exists (needed for the event loop this WebSocket test relies on) -- the
        # instance itself isn't used directly.
        # Ensure a QCoreApplication exists (needed for the event loop this
        # WebSocket test relies on) -- the instance itself isn't used directly.
        QCoreApplication.instance() or QCoreApplication(sys.argv)
        # Port 8082 is test-only so the default bridge port (8080) can stay in use by a running app.
        port = 8082
        manager = WebSocketServerManager.get_instance(port=port)
        manager.started_event.wait(timeout=5.0)

        # Get token and SSL configuration: the client must authenticate like a real editor extension, and
        # use TLS when the server created a certificate.
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
            async with websockets.connect(uri, ssl=ssl_context, close_timeout=2) as ws:
                # Mock Popen inside websocket_server to run a dummy/no-op python command: the RPC spawns
                # auto_train.py as a subprocess, which would start real training. The fake process prints nothing
                # and exits 0.
                # Mock Popen inside websocket_server to run a dummy/no-op python command
                with patch("subprocess.Popen") as mock_popen:
                    mock_proc = MagicMock()
                    mock_proc.stdout = []
                    mock_proc.wait.return_value = 0
                    mock_popen.return_value = mock_proc

                    # The RPC only validates and acknowledges; log lines and the finished notification arrive later.
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

        # Always close the loop and tear the server down so the next test can bind the port.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_client())
        finally:
            loop.close()
            WebSocketServerManager.reset_instance()


# Regression test for a CLI bug: --qlora used to be action="store_true" with default=True, so it
# could never be turned off. The fix added --no-qlora (BooleanOptionalAction-style) and kept the
# default on for existing callers.
class TestQloraFlag(unittest.TestCase):
    """--qlora must be switchable off from the CLI (it used to be store_true with default True)."""

    def _build_parser(self):
        # Extracting the add_argument call with ast and rebuilding a parser avoids importing auto_train.py,
        # which pulls heavy dependencies and may start work at import time.
        """Rebuild an argparse parser from auto_train.py's real --qlora add_argument call (AST-extracted)."""
        import argparse
        import ast
        src = Path(__file__).resolve().parent.parent.joinpath("auto_train.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--qlora"):
                parser = argparse.ArgumentParser()
                expr = ast.Expression(ast.Call(
                    func=ast.Attribute(ast.Name("parser", ast.Load()), "add_argument", ast.Load()),
                    args=node.args, keywords=node.keywords))
                eval(compile(ast.fix_missing_locations(expr), "qlora", "eval"), {"parser": parser, "argparse": argparse})
                return parser
        raise AssertionError("--qlora argument not found")

    # Backward compatibility: no flag means QLoRA stays enabled; --no-qlora switches it off.
    def test_default_on_and_no_qlora_turns_off(self):
        """Default stays True for existing callers; --no-qlora yields False."""
        parser = self._build_parser()
        self.assertTrue(parser.parse_args([]).qlora)
        self.assertFalse(parser.parse_args(["--no-qlora"]).qlora)


if __name__ == "__main__":
    unittest.main()
