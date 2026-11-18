# Program: karl-offline-lab - test_websocket_contract.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests of the JSON-RPC error contract and rate limiting of the WebSocket server handler, using a mock socket and a manager built without starting a real server.
"""Contract tests for JSON-RPC error codes and rate limits in the WebSocket handler."""
import asyncio
import json
import threading

import tests.qt_test_helper  # noqa: F401

from app.engine.websocket_server import WebSocketServerManager


class MockWebSocket:
    """Minimal fake WebSocket that yields queued messages, records sent payloads and records how it was closed."""
    def __init__(self, messages):
        """Queue the messages to deliver and prepare the sent list and close state."""
        self._messages = list(messages)
        self.sent = []
        self.remote_address = ("127.0.0.1", 50000)
        self.closed = False

    def __aiter__(self):
        """Make the object its own async iterator."""
        return self

    async def __anext__(self):
        """Return the next queued message or stop when none remain."""
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def send(self, payload):
        """Record a payload that the handler sent."""
        self.sent.append(payload)

    async def close(self, code=None, reason=None):
        """Record that the socket was closed with the given code and reason."""
        self.closed = True
        self.close_code = code
        self.close_reason = reason


def make_manager():
    # Skip __init__ so no server, threads or sockets are started.
    """Build a WebSocketServerManager without running its constructor, setting only the fields the handler needs and accepting every token."""
    manager = WebSocketServerManager.__new__(WebSocketServerManager)
    manager.port = 19999
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
    # Accept any token so the handler gets past authentication.
    manager._validate_token = lambda token: True
    return manager


async def run_handler_once(manager, message):
    """Feed one message to the handler and return the last response it sent, decoded from JSON."""
    websocket = MockWebSocket([message])
    await manager._handler(websocket, path="/?token=test-token")
    assert websocket.sent
    return json.loads(websocket.sent[-1])


def assert_rpc_error(response, code, req_id):
    """Assert that a response is a JSON-RPC 2.0 error with the given code and request id."""
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == req_id
    assert response["error"]["code"] == code
    assert isinstance(response["error"]["message"], str)


def test_parse_error_returns_json_rpc_error():
    """Scenario: the client sends text that is not valid JSON.
    Expected: error code -32700 with a null id.
    """
    response = asyncio.run(run_handler_once(make_manager(), "{bad json"))
    assert_rpc_error(response, -32700, None)


def test_missing_jsonrpc_returns_invalid_request():
    """Scenario: the request lacks the version field.
    Expected: error code -32600 for that request id.
    """
    response = asyncio.run(run_handler_once(
        make_manager(),
        json.dumps({"id": 10, "method": "get_runtime_status"}),
    ))
    assert_rpc_error(response, -32600, 10)


def test_unknown_method_returns_method_not_found():
    """Scenario: the request names a method that does not exist.
    Expected: error code -32601.
    """
    response = asyncio.run(run_handler_once(
        make_manager(),
        json.dumps({"jsonrpc": "2.0", "id": 11, "method": "no_such_method"}),
    ))
    assert_rpc_error(response, -32601, 11)


def test_missing_required_params_returns_invalid_params():
    """Scenario: set_active_model is called without parameters.
    Expected: error code -32602.
    """
    response = asyncio.run(run_handler_once(
        make_manager(),
        json.dumps({"jsonrpc": "2.0", "id": 12, "method": "set_active_model", "params": {}}),
    ))
    assert_rpc_error(response, -32602, 12)


def test_handler_exception_returns_internal_error():
    """Scenario: the method handler raises an exception.
    Expected: error code -32603 with the exception text as the error data.
    """
    manager = make_manager()

    def raise_runtime_error():
        """Always raise an error with the text "boom"."""
        raise ValueError("boom")

    manager._list_models = raise_runtime_error
    response = asyncio.run(run_handler_once(
        manager,
        json.dumps({"jsonrpc": "2.0", "id": 13, "method": "list_models"}),
    ))
    assert_rpc_error(response, -32603, 13)
    assert response["error"]["data"] == "boom"


def test_global_rate_limiter_exceeded():
    """Scenario: 22 requests arrive in a row against a global capacity of 20.
    Expected: the first 20 are not rate limited and the last 2 get error -32005 mentioning the global rate limit.
    """
    manager = make_manager()
    # Cheap fake so the rate limit test does not do real work.
    manager._list_models = lambda: {"models": []}

    # Global rate limit is capacity=20.0
    requests = [
        json.dumps({"jsonrpc": "2.0", "id": i, "method": "list_models"})
        for i in range(22)
    ]
    websocket = MockWebSocket(requests)
    
    asyncio.run(manager._handler(websocket, path="/?token=test-token"))
    
    assert len(websocket.sent) == 22
    for i in range(20):
        resp = json.loads(websocket.sent[i])
        if "error" in resp:
            assert resp["error"]["code"] != -32005
    
    for i in range(20, 22):
        resp = json.loads(websocket.sent[i])
        assert_rpc_error(resp, -32005, i)
        assert "global rate limit exceeded" in resp["error"]["message"]


def test_heavy_rate_limiter_exceeded():
    """Scenario: 5 heavy-method requests arrive against a heavy capacity of 3.
    Expected: the first 3 are not rate limited and the last 2 get error -32005 mentioning the heavy method limit.
    """
    manager = make_manager()
    # Mock set_active_model which is a heavy method
    manager._set_active_model = lambda filename, adapter=None: {"active": {}}

    # Heavy limit capacity=3.0
    requests = [
        json.dumps({
            "jsonrpc": "2.0",
            "id": i,
            "method": "set_active_model",
            "params": {"filename": "model.gguf"}
        })
        for i in range(5)
    ]
    websocket = MockWebSocket(requests)
    
    asyncio.run(manager._handler(websocket, path="/?token=test-token"))
    
    assert len(websocket.sent) == 5
    for i in range(3):
        resp = json.loads(websocket.sent[i])
        if "error" in resp:
            assert resp["error"]["code"] != -32005
    
    for i in range(3, 5):
        resp = json.loads(websocket.sent[i])
        assert_rpc_error(resp, -32005, i)
        assert "rate limit exceeded for heavy method" in resp["error"]["message"]
