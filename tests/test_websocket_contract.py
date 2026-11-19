# ==============================================================================
# Program: test_websocket_contract.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising websocket contract functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import asyncio
import json
import threading

import tests.qt_test_helper  # noqa: F401

from app.engine.websocket_server import WebSocketServerManager


class MockWebSocket:
    """Minimal async websocket double: yields the given incoming messages, records everything sent, and records close()."""
    def __init__(self, messages):
        """Store incoming messages and start with empty sent list."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._messages = list(messages)
        self.sent = []
        self.remote_address = ("127.0.0.1", 50000)
        self.closed = False

    def __aiter__(self):
        """Return self so the handler can use 'async for'."""
        # Step 1: Validate input parameters, operational preconditions, and state
        return self

    async def __anext__(self):
        """Yield the next queued message or end iteration."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def send(self, payload):
        """Record an outgoing payload instead of sending it."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.sent.append(payload)

    async def close(self, code=None, reason=None):
        """Record the close code and reason."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self.closed = True
        self.close_code = code
        self.close_reason = reason


def make_manager():
    """Build a WebSocketServerManager with heavy collaborators mocked so JSON-RPC handling can be tested without a network or Qt."""
    # Step 1: Validate input parameters, operational preconditions, and state
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
    manager._validate_token = lambda token: True
    return manager


async def run_handler_once(manager, message):
    """Feed one raw message to the connection handler and return the parsed JSON response; fails if nothing was sent."""
    # Step 1: Validate input parameters, operational preconditions, and state
    websocket = MockWebSocket([message])
    await manager._handler(websocket, path="/?token=test-token")
    assert websocket.sent
    return json.loads(websocket.sent[-1])


def assert_rpc_error(response, code, req_id):
    """Assert a response is a well-formed JSON-RPC 2.0 error with the expected code and request id."""
    # Step 1: Validate input parameters, operational preconditions, and state
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == req_id
    assert response["error"]["code"] == code
    assert isinstance(response["error"]["message"], str)


def test_parse_error_returns_json_rpc_error():
    """Invalid JSON yields error -32700 with a null id."""
    # Step 1: Validate input parameters, operational preconditions, and state
    response = asyncio.run(run_handler_once(make_manager(), "{bad json"))
    assert_rpc_error(response, -32700, None)


def test_missing_jsonrpc_returns_invalid_request():
    """A request lacking jsonrpc '2.0' yields -32600 (Invalid Request) carrying the request id."""
    # Step 1: Validate input parameters, operational preconditions, and state
    response = asyncio.run(run_handler_once(
        make_manager(),
        json.dumps({"id": 10, "method": "get_runtime_status"}),
    ))
    assert_rpc_error(response, -32600, 10)


def test_unknown_method_returns_method_not_found():
    """An unknown method yields -32601."""
    # Step 1: Validate input parameters, operational preconditions, and state
    response = asyncio.run(run_handler_once(
        make_manager(),
        json.dumps({"jsonrpc": "2.0", "id": 11, "method": "no_such_method"}),
    ))
    assert_rpc_error(response, -32601, 11)


def test_missing_required_params_returns_invalid_params():
    """A known method missing required params yields -32602."""
    # Step 1: Validate input parameters, operational preconditions, and state
    response = asyncio.run(run_handler_once(
        make_manager(),
        json.dumps({"jsonrpc": "2.0", "id": 12, "method": "set_active_model", "params": {}}),
    ))
    assert_rpc_error(response, -32602, 12)


def test_handler_exception_returns_internal_error():
    """If the method handler raises, the client gets -32603 with the exception text in 'data', and the server stays up."""
    # Step 1: Validate input parameters, operational preconditions, and state
    manager = make_manager()

    def raise_runtime_error():
        """Handler stub that raises RuntimeError('boom')."""
        # Step 1: Validate input parameters, operational preconditions, and state
        raise ValueError("boom")

    manager._list_models = raise_runtime_error
    response = asyncio.run(run_handler_once(
        manager,
        json.dumps({"jsonrpc": "2.0", "id": 13, "method": "list_models"}),
    ))
    assert_rpc_error(response, -32603, 13)
    assert response["error"]["data"] == "boom"


def test_global_rate_limiter_exceeded():
    """The global token bucket (capacity 20) allows the first 20 calls and answers later ones with -32005 'global rate limit exceeded', protecting the bridge from floods."""
    # Step 1: Validate input parameters, operational preconditions, and state
    manager = make_manager()
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
        # Iterate over items, evaluate loop invariant, and update progress
        resp = json.loads(websocket.sent[i])
        if "error" in resp:
            assert resp["error"]["code"] != -32005
    
    for i in range(20, 22):
        # Iterate over items, evaluate loop invariant, and update progress
        resp = json.loads(websocket.sent[i])
        assert_rpc_error(resp, -32005, i)
        assert "global rate limit exceeded" in resp["error"]["message"]


def test_heavy_rate_limiter_exceeded():
    """Heavy methods (such as set_active_model) have a tighter bucket (capacity 3); extra calls get -32005 'rate limit exceeded for heavy method'."""
    # Step 1: Validate input parameters, operational preconditions, and state
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
        # Iterate over items, evaluate loop invariant, and update progress
        resp = json.loads(websocket.sent[i])
        if "error" in resp:
            assert resp["error"]["code"] != -32005
    
    for i in range(3, 5):
        # Iterate over items, evaluate loop invariant, and update progress
        resp = json.loads(websocket.sent[i])
        assert_rpc_error(resp, -32005, i)
        assert "rate limit exceeded for heavy method" in resp["error"]["message"]
