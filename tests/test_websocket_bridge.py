# ==============================================================================
# Program: test_websocket_bridge.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: WebSocket Bridge Tests — Karl Workbench ======================================== Integration tests verifying JSON-RPC task submission, signal-to-socket routing, and connection teardown using a local mock client.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

"""
WebSocket Bridge Tests — Karl Workbench
========================================
Integration tests verifying JSON-RPC task submission, signal-to-socket routing,
and connection teardown using a local mock client.
"""

# Side-effect import: prepares the Qt platform (offscreen) before QCoreApplication is created.
import tests.qt_test_helper  # noqa: F401

# What this file is: end-to-end tests of the editor/remote bridge (app/engine/websocket_server.py),
# the JSON-RPC 2.0 server that the VS Code extension, Neovim client and remote Karl instances use.
# Each test boots the real server on a local port and talks to it with a real websockets client,
# so routing, serialisation, authentication and Qt-signal-to-socket forwarding are all exercised.
import os
import sys
import json
import tempfile
# tempfile: per-test sandbox workspace; asyncio: the client is async; unittest: the TestCase base.
import asyncio
import unittest
# patch/MagicMock replace the LLM (and time.time in the token tests); nothing loads a real model.
from unittest.mock import patch, MagicMock

# websockets is the client library; the module cannot be tested without it.
import websockets
import pytest
# QCoreApplication supplies the event loop needed to deliver cross-thread Qt signals to the server.
from PyQt6.QtCore import QCoreApplication
from app.engine.websocket_server import WebSocketServerManager

# Marks every test in the module so CI can deselect socket-based tests with -m "not websocket".
pytestmark = [pytest.mark.integration, pytest.mark.websocket]


# Inside bubblewrap sandboxes loopback sockets can be blocked, so tests skip instead of timing out.
def _running_under_bwrap() -> bool:
    """True when PID 1 is bubblewrap, i.e. the tests run inside a sandbox where local sockets or system calls may be restricted (used to skip such cases)."""
    try:
        with open("/proc/1/comm", "r", encoding="utf-8") as f:
            return f.read().strip() == "bwrap"
    except OSError:
        return False


# Test double for RAGPipeline. The bridge's knowledge-base RPCs call ingest_file/retrieve*/
# total_chunks/documents; implementing just those with plain lists avoids FAISS and an embedding model.
class FakeBridgeRAG:
    """In-memory stand-in for RAGPipeline: stores ingested chunks in a list and answers retrieval by simple lookup, so knowledge-base RPCs can be tested without FAISS, SQLite or an embedding model."""
    def __init__(self):
        """Start with an empty document list."""
        self.documents = []

    # The real pipeline loads a FAISS index lazily from disk; the fake has nothing to load.
    def _load_index(self):
        """No-op replacement for loading a persisted index."""
        return None

    @property
    def total_chunks(self):
        """Number of stored chunks."""
        return len(self.documents)

    # One chunk per file is enough for the assertions (chunk_size/overlap are accepted but ignored).
    def ingest_file(self, filepath, chunk_size=200, overlap=50):
        """Store a file's content as one chunk and return how many chunks were added."""
        # errors="ignore" so a binary file cannot crash the fake.
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        self.documents.append({
            "text": text,
            "source_file": filename,
            "chunk_id": len(self.documents),
            "ingested_at": "2026-06-11T00:00:00Z",
        })
        return 1

    # threshold is a maximum distance (lower is closer), matching the real pipeline's semantics.
    def retrieve(self, query, top_k=3, source_filter=None, threshold=0.0):
        """Return the texts of matching chunks, applying the distance threshold when one is given."""
        results = self.retrieve_with_metadata(query, top_k=top_k, source_filter=source_filter)
        if threshold > 0:
            results = [r for r in results if r["distance"] <= threshold]
        return [r["text"] for r in results]

    def retrieve_with_metadata(self, query, top_k=3, source_filter=None):
        """Return chunk dicts (optionally filtered by source file) with a fake distance: 0.05 when the query is a substring of the chunk text, otherwise 1.5."""
        results = []
        for idx, doc in enumerate(self.documents):
            if source_filter and doc.get("source_file") != source_filter:
                continue
            # Substring match fakes "semantic" similarity deterministically: 0.05 is a strong hit, 1.5 a miss.
            distance = 0.05 if query.lower() in doc.get("text", "").lower() else 1.5
            results.append({
                **doc,
                "rank": idx,
                "distance": distance,
            })
        return results[:top_k]


# Test class lifecycle: setUp starts the server, tearDown resets the singleton, so every test sees a
# fresh server, a fresh token and no leftover connections.
class TestWebSocketBridge(unittest.TestCase):
    """Integration tests that start the real WebSocketServerManager on a test port and drive it with a websockets client: RPC methods, notifications, token auth and rotation. Qt events are pumped by a spinner task because signals cross threads."""
    def setUp(self):
        """Create a QCoreApplication, start the bridge server on port 8081 (a test-only port) and prepare a sandbox directory with a trivially passing test script."""
        # Returned from setUp, so the test is reported as skipped rather than failed.
        if _running_under_bwrap():
            self.skipTest("Codex sandbox blocks reliable localhost WebSocket tests")

        # Initialize a PyQt application instance for QObject/QThread creation safety.
        # Initialize a PyQt application instance for QObject/QThread creation safety
        self.app = QCoreApplication.instance() or QCoreApplication(sys.argv)

        # Start server on unique port (8081) to avoid local workspace collisions.
        # Start server on unique port (8081) to avoid local workspace collisions
        # Port 8081 is test-only: the real app uses a different default, so a developer's running Karl instance cannot collide with the suite.
        self.port = 8081
        self.manager = WebSocketServerManager.get_instance(port=self.port)
        # started_event is set by the server thread once it is listening.
        # The server runs in its own thread; without waiting for this event the first client connect would race the listener and fail with connection refused.
        self.manager.started_event.wait(timeout=5.0)

        # Set up a sandbox directory: swarm tasks write their files here, never into the repo.
        # Set up a sandbox directory
        self.sandbox_dir = tempfile.TemporaryDirectory()
        self.workspace_path = self.sandbox_dir.name

        # Create a dummy python test runner script that succeeds instantly; the swarm tester runs it as the
        # user's "test command".
        # Create a dummy python test runner script that succeeds instantly
        self.test_script_path = os.path.join(self.workspace_path, "run_tests.py")
        with open(self.test_script_path, "w", encoding="utf-8") as f:
            f.write("import sys\nsys.dont_write_bytecode = True\nsys.exit(0)\n")

    def tearDown(self):
        """Remove the sandbox and reset the server singleton so the next test starts clean."""
        self.sandbox_dir.cleanup()
        # Resetting closes the listening socket and discards the token so ports are free for the next test.
        # Clean shutdown and reset the singleton
        WebSocketServerManager.reset_instance()

    def _get_uri(self):
        """Build the client URL with the live bridge token, using wss when a TLS cert exists and ws otherwise."""
        # The token is generated per server instance and rotated on expiry; clients pass it as ?token=.
        token = self.manager.bridge_token
        proto = "ws"
        # The server uses TLS when data/ssl/localhost.crt exists, otherwise plain ws.
        if os.path.exists(self.manager._SSL_CERT_PATH):
            proto = "wss"
        return f"{proto}://localhost:{self.port}/?token={token}"

    def _get_ssl_context(self):
        """Client SSL context that accepts the self-signed test certificate, or None for plain ws."""
        # The test certificate is self-signed, so hostname and chain verification are disabled. This is
        # acceptable only because the client talks to localhost inside the test.
        import ssl
        if os.path.exists(self.manager._SSL_CERT_PATH):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            return ssl_context
        return None

    # Smoke test for the heartbeat method the VS Code extension polls every few seconds (id 30).
    def test_runtime_status_rpc(self):
        """The runtime_status RPC returns bridge port, idle runtime state and model/adapter/system sections."""
        async def run_client():
            """Connect, call the runtime status RPC and assert the structure of the answer."""
            # The id must be echoed back; the extension relies on id 30 to recognise the heartbeat reply.
            async with websockets.connect(self._get_uri(), ssl=self._get_ssl_context(), close_timeout=2) as ws:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 30,
                    "method": "get_runtime_status"
                }
                await ws.send(json.dumps(payload))

                raw_resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                resp = json.loads(raw_resp)

                self.assertEqual(resp.get("id"), 30)
                self.assertIn("result", resp)
                result = resp["result"]
                # "idle" means no swarm task or chat is running; a fresh server must start in that state.
                self.assertEqual(result["bridge"]["port"], self.port)
                self.assertEqual(result["runtime"]["state"], "idle")
                self.assertIn("model", result)
                self.assertIn("adapter", result)
                self.assertIn("system", result)

        # A fresh event loop per test: avoids sharing state through asyncio's default loop between tests.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_client())
        finally:
            loop.close()

    def test_model_registry_rpc(self):
        """The model registry RPCs list known model files and let a client change the active model; the response reflects the new selection."""
        # Registry RPCs let editors choose which GGUF is active without touching the desktop UI.
        active_path = os.path.join("data", "active_model.json")
        model_dir = os.path.join("data", "models")
        model_name = "test-websocket-model.gguf"
        model_path = os.path.join(model_dir, model_name)
        previous_active = None
        if os.path.exists(active_path):
            # The real active_model.json belongs to the user; keep its content so the test can restore it.
            with open(active_path, "r", encoding="utf-8") as f:
                # Snapshot the user's real active-model pointer so the finally block can restore it; this test writes into the real data/ directory.
                previous_active = f.read()

        # _list_models also reports stray .gguf files in data/models that have no registry entry, so a 4-byte dummy file is enough to show up without a real model.
        # Create a minimal fake .gguf so list_models can see it without a real model.
        os.makedirs(model_dir, exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(b"test")

        async def run_client():
            """Connect, list registry models, set the active model and check the response."""
            async with websockets.connect(self._get_uri(), ssl=self._get_ssl_context(), close_timeout=2) as ws:
                # The numeric ids (31, 32) are arbitrary but distinct so replies can be matched to requests.
                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 31,
                    "method": "list_models"
                }))
                list_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                self.assertEqual(list_resp.get("id"), 31)
                filenames = [m["filename"] for m in list_resp["result"]["models"]]
                self.assertIn(model_name, filenames)

                # Switching models rewrites data/active_model.json and resets the loaded model; the reply contains
                # the new selection.
                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 32,
                    "method": "set_active_model",
                    "params": {"filename": model_name}
                }))
                set_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                self.assertEqual(set_resp.get("id"), 32)
                self.assertEqual(set_resp["result"]["active"]["filename"], model_name)

        # Run on a private loop and always restore the user's files afterwards.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_client())
        finally:
            loop.close()
            # Restore (or delete) the active-model pointer exactly as it was.
            if previous_active is None:
                try:
                    os.remove(active_path)
                except FileNotFoundError:
                    pass
            else:
                with open(active_path, "w", encoding="utf-8") as f:
                    f.write(previous_active)
            # Remove the fake model file.
            try:
                os.remove(model_path)
            except FileNotFoundError:
                pass

    def test_prompt_pair_rpc(self):
        # Prompt pairs are saved A/B prompt experiments stored as JSON under data/prompt_pairs/.
        # This test drives the full CRUD cycle through RPC ids 42 (save), 40 (list), 41 (get), 43 (delete).
        """Prompt pairs (system A / system B) can be saved, listed, fetched and deleted through RPC."""
        pair_dir = os.path.join("data", "prompt_pairs")
        pair_name = "websocket_pair_test"
        pair_path = os.path.join(pair_dir, f"{pair_name}.json")
        # Start from a clean state so an earlier failed run cannot make "save" look like an overwrite.
        os.makedirs(pair_dir, exist_ok=True)
        try:
            os.remove(pair_path)
        except FileNotFoundError:
            pass

        async def run_client():
            """Connect and exercise save, list, get and delete of a prompt pair."""
            async with websockets.connect(self._get_uri(), ssl=self._get_ssl_context(), close_timeout=2) as ws:
                # save_prompt_pair persists both system prompts, the shared user input and the raw outputs.
                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 42,
                    "method": "save_prompt_pair",
                    "params": {
                        "name": pair_name,
                        "system_a": "A system",
                        "user_a": "same input",
                        "system_b": "B system",
                        "user_b": "same input",
                        "output_a_raw": "alpha",
                        "output_b_raw": "beta"
                    }
                }))
                save_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                # JSON-RPC ids are echoed back verbatim; matching them proves each reply is correlated to its request rather than to an earlier notification.
                self.assertEqual(save_resp.get("id"), 42)
                self.assertTrue(save_resp["result"]["saved"])

                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 40,
                    "method": "list_prompt_pairs"
                }))
                list_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                # The saved pair must appear in the listing.
                names = [p["name"] for p in list_resp["result"]["pairs"]]
                self.assertIn(pair_name, names)

                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 41,
                    "method": "get_prompt_pair",
                    "params": {"name": pair_name}
                }))
                get_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                # get returns the same fields that were saved (round trip).
                self.assertEqual(get_resp["result"]["system_a"], "A system")
                self.assertEqual(get_resp["result"]["system_b"], "B system")

                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 43,
                    "method": "delete_prompt_pair",
                    "params": {"name": pair_name}
                }))
                delete_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                # delete reports success and (in the finally block below) the file is removed anyway.
                self.assertTrue(delete_resp["result"]["deleted"])

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_client())
        finally:
            loop.close()
            try:
                os.remove(pair_path)
            except FileNotFoundError:
                pass

    # Covers ids 50-52: stats, asynchronous ingest and search.
    def test_knowledge_base_rpc(self):
        """Knowledge-base RPCs: initial stats are empty, an ingest starts asynchronously and emits progress and finished notifications, and a search returns the ingested chunk with a close distance."""
        # Swap in the in-memory RAG before any RPC runs so ingest/search never touch FAISS or the embedding model.
        # Swap in the in-memory RAG before any RPC runs so ingest/search never touch FAISS or the embedding model.
        self.manager.rag = FakeBridgeRAG()
        doc_path = os.path.join(self.workspace_path, "notes.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Karl retrieves local project knowledge for prompt context.")

        async def run_client():
            # The empty-store check proves the fake starts clean and list_kb_sources reports totals.
            """Connect, check empty stats, ingest a file, wait for the finish notification and search."""
            async with websockets.connect(self._get_uri(), ssl=self._get_ssl_context(), close_timeout=2) as ws:
                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 50,
                    "method": "list_kb_sources"
                }))
                initial_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                self.assertEqual(initial_resp.get("id"), 50)
                self.assertEqual(initial_resp["result"]["total_chunks"], 0)

                # Ingest runs on a background thread in the server; the RPC reply only acknowledges it.
                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 51,
                    "method": "ingest_path",
                    "params": {
                        "path": doc_path,
                        "chunk_size": 100,
                        "overlap": 10,
                    }
                }))
                ingest_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                self.assertEqual(ingest_resp.get("id"), 51)
                # Ingest is asynchronous by design: the RPC must return immediately and report completion via notifications, otherwise large files would block the socket.
                self.assertEqual(ingest_resp["result"]["status"], "started")
                self.assertEqual(ingest_resp["result"]["file_count"], 1)

                # Collect notification method names (messages without an id) in arrival order.
                notifications = []
                finished = None
                # Bounded read loop: progress notifications may arrive in any number before the finish message, but an endless wait would hang the suite if the server never emitted it.
                for _ in range(10):
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                    method = msg.get("method")
                    if method:
                        notifications.append(method)
                    if method == "kb_ingest_finished":
                        finished = msg["params"]
                        break

                self.assertIn("kb_ingest_progress", notifications)
                self.assertIn("kb_ingest_finished", notifications)
                self.assertIsNotNone(finished)
                self.assertEqual(finished["chunks_added"], 1)
                self.assertEqual(finished["snapshot"]["total_chunks"], 1)

                # search_kb with a threshold filters by distance, exactly like the real retrieval path.
                await ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 52,
                    "method": "search_kb",
                    "params": {
                        "query": "project knowledge",
                        "top_k": 5,
                        # FakeBridgeRAG gives a substring match distance 0.05 and a miss 1.5, so 0.2 keeps the hit and would drop anything unrelated.
                        "threshold": 0.2,
                    }
                }))
                search_resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                self.assertEqual(search_resp.get("id"), 52)
                self.assertEqual(len(search_resp["result"]["results"]), 1)
                result = search_resp["result"]["results"][0]
                self.assertEqual(result["source_file"], "notes.md")
                self.assertLessEqual(result["distance"], 0.2)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_client())
        finally:
            loop.close()

    # Largest test: a Swarm run through the bridge, which is what the VS Code "Swarm" tab does.
    # Path: submit_task RPC -> SwarmOrchestratorThread (Architect, Coder, Tester) -> Qt signals ->
    # bridge notifications (task_plan_created, file_edited, finished_swarm) -> client.
    @patch("app.engine.model_loader.ModelLoader.get_instance")
    def test_websocket_bridge_flow(self, mock_get_llm):
        """Full swarm run over the bridge with a mocked LLM: the submit RPC answers 'started', then task_plan_created, file_edited and finished_swarm notifications are broadcast to the client."""
        # The fake distinguishes the architect (planning) call from the file-writing call by whether the prompt mentions 'tasks'; planning gets the JSON plan, everything else gets file content.
        # Mock LLM to return simple JSON plan and script contents
        def mock_llm(prompt, **kwargs):
            """Return a JSON architect plan and Python file content depending on the prompt."""
            if "tasks" in prompt:
                plan = {
                    "explanation": "Create dummy python file",
                    "tasks": [
                        {
                            "filepath": "dummy.py",
                            "instructions": "Write print('hello')"
                        }
                    ]
                }
                return {"choices": [{"text": json.dumps(plan)}]}
            return {"choices": [{"text": "print('hello')\n"}]}

        mock_get_llm.return_value = MagicMock(side_effect=mock_llm)

        # The orchestrator thread emits Qt signals that the server forwards from the main thread; without
        # processEvents() running, no notification would ever be sent. 10 ms keeps latency low without busy-waiting.
        async def qt_event_loop_spinner():
            """Periodically processes Qt events to deliver cross-thread signals."""
            while True:
                QCoreApplication.processEvents()
                await asyncio.sleep(0.01)

        async def run_client():
            """Submit a swarm task over the bridge and collect notifications until finished_swarm."""
            # Spawn the Qt event loop spinner task in the background.
            # Spawn the Qt event loop spinner task in the background
            spinner = asyncio.create_task(qt_event_loop_spinner())

            try:
                # Connect to Karl's WebSocket Server
                async with websockets.connect(self._get_uri(), ssl=self._get_ssl_context(), close_timeout=2) as ws:
                    # 1. Submit the swarm task: id 1 is arbitrary and is echoed in the reply.
                    # 1. Submit the swarm task
                    task_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "submit_task",
                        "params": {
                            "objective": "Write a dummy python file",
                            "workspace_path": self.workspace_path,
                            "test_command": f"{sys.executable} run_tests.py"
                        }
                    }
                    await ws.send(json.dumps(task_payload))

                    # Check response with timeout
                    raw_resp = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    resp = json.loads(raw_resp)

                    self.assertEqual(resp.get("id"), 1)
                    self.assertIn("result", resp)
                    self.assertEqual(resp["result"]["status"], "started")

                    # 2. Consume progress notifications until finished_swarm is received.
                    # 2. Consume progress notifications until finished_swarm is received
                    notifications = []
                    # Cap at 50 messages so a missing finished_swarm fails the test instead of hanging it.
                    # x
                    for _ in range(50):  # Safety iteration ceiling
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=15.0)

                        msg = json.loads(raw_msg)
                        method = msg.get("method")
                        if method:
                            notifications.append(method)
                            if method == "finished_swarm":
                                break

                    # Validate expected notifications were broadcast.
                    # Validate expected notifications were broadcast
                    self.assertIn("task_plan_created", notifications)
                    self.assertIn("file_edited", notifications)
                    self.assertIn("finished_swarm", notifications)
            finally:
                # Clean shutdown of the spinner task: cancel it and swallow the expected CancelledError.
                # Clean shutdown of the spinner task
                spinner.cancel()
                try:
                    await spinner
                except asyncio.CancelledError:
                    pass

        # Run async client loop in main thread.
        # Run async client loop in main thread
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_client())
        finally:
            loop.close()

    # Chat path: submit_chat -> LLMThread -> streaming parser splits <think> from the answer ->
    # notifications chat_thought_token / chat_response_token / chat_finished.
    @patch("app.engine.model_loader.ModelLoader.get_instance")
    def test_websocket_chat_flow(self, mock_get_llm):
        """Chat over the bridge: the RPC answers 'started' and streamed thought tokens, response tokens and a chat_finished notification arrive."""
        def mock_llm_stream(prompt, **kwargs):
            """When streaming, return the chunks <think>, 'Thinking', </think>, 'Hello world' (only 'Hello world' once the prompt already contains more than one </think>); non-streaming calls return 'Response'."""
            # stream=True yields incremental chunks like llama-cpp's streaming API. The branch on
            # prompt.count("</think>") covers the auto-continuation pass, which re-queries with the finished
            # thought already in the prompt.
            if kwargs.get("stream"):
                if prompt.count("</think>") > 1:
                    return [
                        {"choices": [{"text": "Hello world", "finish_reason": "stop"}]}
                    ]
                return [
                    {"choices": [{"text": "<think>", "finish_reason": None}]},
                    {"choices": [{"text": "Thinking", "finish_reason": None}]},
                    {"choices": [{"text": "</think>", "finish_reason": None}]},
                    {"choices": [{"text": "Hello world", "finish_reason": "stop"}]}
                ]
            return {"choices": [{"text": "Response"}]}

        # tokenize is used for context budgeting inside LLMThread; any list of ints works.
        mock_llm_instance = MagicMock()
        mock_llm_instance.tokenize.return_value = [1, 2, 3]
        mock_llm_instance.side_effect = mock_llm_stream
        mock_get_llm.return_value = mock_llm_instance

        async def qt_event_loop_spinner():
            """Periodically process Qt events so cross-thread signals are delivered while the async client waits."""
            while True:
                QCoreApplication.processEvents()
                await asyncio.sleep(0.01)

        async def run_client():
            """Submit a chat request over the bridge and collect token notifications until chat_finished."""
            spinner = asyncio.create_task(qt_event_loop_spinner())
            try:
                async with websockets.connect(self._get_uri(), ssl=self._get_ssl_context(), close_timeout=2) as ws:
                    # agentic_loop_enabled False keeps the test to a single generation; rag_enabled False avoids retrieval.
                    chat_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "submit_chat",
                        "params": {
                            "message": "Hello Karl",
                            "workspace_path": self.workspace_path,
                            "hyperparams": {
                                "temperature": 0.7,
                                "top_p": 0.95,
                                "max_tokens": 100,
                                "rag_enabled": False,
                                "agentic_loop_enabled": False
                            }
                        }
                    }
                    await ws.send(json.dumps(chat_payload))

                    raw_resp = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    resp = json.loads(raw_resp)
                    self.assertEqual(resp.get("id"), 1)
                    self.assertEqual(resp["result"]["status"], "started")

                    # Same pattern as the swarm test: collect notification names until the terminal one arrives.
                    notifications = []
                    for _ in range(50):
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                        msg = json.loads(raw_msg)
                        method = msg.get("method")
                        if method:
                            notifications.append(method)
                            if method == "chat_finished":
                                break

                    self.assertIn("chat_thought_token", notifications)
                    self.assertIn("chat_response_token", notifications)
                    self.assertIn("chat_finished", notifications)
            finally:
                spinner.cancel()
                try:
                    await spinner
                except asyncio.CancelledError:
                    pass

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_client())
        finally:
            loop.close()

    # The desktop status bar polls WebSocketServerManager to show whether the editor bridge is up.
    def test_status_bar_reflection(self):
        """The GUI status bar shows 'VS Code: listening' while the bridge is up and 'VS Code: offline' after the server instance is reset."""
        from app.ui.widgets.status_bar import StatusBar

        # Instantiate StatusBar
        # StatusBar needs no AppState for this check.
        status_bar = StatusBar()

        # The WebSocketServerManager is currently listening on port 8081 (from setUp)
        # _tick is the periodic refresh slot; calling it directly avoids waiting for the real timer.
        status_bar._tick()
        self.assertEqual(status_bar._vscode_lbl.text(), "VS Code: listening")

        # Reset the server instance
        # Tearing the server down must flip the label on the next tick.
        WebSocketServerManager.reset_instance()

        # Tick again - should show offline
        status_bar._tick()
        self.assertEqual(status_bar._vscode_lbl.text(), "VS Code: offline")

    # Security group (token tests): the bridge refuses clients without a valid, unexpired token.
    def test_token_connection_success(self):
        """A client presenting the valid bridge token can connect and make an RPC."""
        async def run_client():
            """Connect with the valid token and make a simple RPC."""
            async with websockets.connect(self._get_uri(), ssl=self._get_ssl_context(), close_timeout=2) as ws:
                payload = {"jsonrpc": "2.0", "id": 1, "method": "get_runtime_status"}
                await ws.send(json.dumps(payload))
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                self.assertIn("result", resp)

        asyncio.run(run_client())

    @patch("time.time")
    def test_token_expiry_fail(self, mock_time):
        """A token older than its lifetime (13 hours in the faked clock) is rejected: the server closes the connection with code 4001."""
        # 1. Set current time to a fixed point: patching time.time makes expiry testable without sleeping.
        # 1. Set current time to a fixed point
        now = 1700000000.0
        mock_time.return_value = now
        
        # 2. Force token initialization at this time.
        # 2. Force token initialization at this time
        self.manager._rotate_token()
        expired_token = self.manager.bridge_token
        
        # 3. Fast-forward time by 13 hours (46800 seconds); the lifetime is shorter than that (rotation test below uses 50000 s).
        # 3. Fast-forward time by 13 hours (46800 seconds)
        mock_time.return_value = now + 46800.0
        
        # Rejection is signalled with WebSocket close code 4001, the app-level "unauthorized" code that the
        # VS Code extension also special-cases (it stops reconnecting on 4001).
        async def run_client():
            """Try to connect with an expired token and report the close code the server sends."""
            uri = f"ws://localhost:{self.port}/?token={expired_token}"
            if os.path.exists(self.manager._SSL_CERT_PATH):
                uri = uri.replace("ws://", "wss://")
            
            import websockets.exceptions
            try:
                # We expect the server to close the connection with 4001
                async with websockets.connect(uri, ssl=self._get_ssl_context(), close_timeout=2) as ws:
                    await asyncio.wait_for(ws.recv(), timeout=2.0)
            # Some versions of websockets raise ConnectionClosedError, others InvalidStatusCode during handshake.
            except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidStatusCode) as e:
                # Some versions of websockets raise ConnectionClosedError, others InvalidStatusCode during handshake
                code = getattr(e, "code", getattr(e, "status_code", None))
                self.assertEqual(code, 4001)
                return
            except Exception as e:
                # If it's a generic ConnectionClosed, check the code
                if hasattr(e, "code") and e.code == 4001:
                    return
                raise e
            
            self.fail("Connection was not rejected with 4001")

        asyncio.run(run_client())

    @patch("time.time")
    # Rotation is lazy: the server notices the expired token on the next connection attempt, mints a
    # new one and writes it to data/bridge_token.json so editors can re-read it.
    def test_token_rotation_on_disk(self, mock_time):
        """After the token expires, a connection attempt triggers rotation: a new token is written to disk with the new creation time."""
        # 1. Start at a fixed time
        now = 1700000000.0
        mock_time.return_value = now
        self.manager._rotate_token()
        old_token = self.manager.bridge_token
        
        # 2. Expire the token
        mock_time.return_value = now + 50000.0
        
        # Attempting a connection with the old token should trigger rotation
        async def run_client():
            """Attempt a connection with the old token, which triggers token rotation."""
            uri = f"ws://localhost:{self.port}/?token={old_token}"
            if os.path.exists(self.manager._SSL_CERT_PATH):
                uri = uri.replace("ws://", "wss://")
            try:
                import websockets.exceptions
                async with websockets.connect(uri, ssl=self._get_ssl_context(), close_timeout=2):
                    pass
            except (websockets.exceptions.InvalidStatusCode, websockets.exceptions.ConnectionClosedError):
                pass
        
        asyncio.run(run_client())
            
        # 3. Verify token has changed on disk; created_at must equal the faked time so the lifetime
        # restarts from the moment of rotation.
        # 3. Verify token has changed on disk
        with open("data/bridge_token.json", "r") as f:
            data = json.load(f)
            self.assertNotEqual(data["token"], old_token)
            self.assertEqual(data["created_at"], now + 50000.0)

# Direct execution runs the whole module with the standard unittest runner.
if __name__ == "__main__":
    unittest.main()
