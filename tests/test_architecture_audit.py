# Program: karl-offline-lab - test_architecture_audit.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Source-level and behavioural audit tests for model locking, guarded generation threads, queued signals and websocket client handling.
"""Architecture audit tests: guard rails that keep concurrency and cleanup behaviour from regressing."""

import asyncio
import inspect
import threading
from pathlib import Path

import pytest


def test_model_loader_blocks_reset_during_active_inference(monkeypatch):
    """Scenario: acquire the model, then try to reset it. Expected: reset raises while inference is active and the lock clears after unlock."""
    # Skip when the optional llama runtime is not installed
    pytest.importorskip("llama_cpp")
    from app.engine.model_loader import ModelLoader

    sentinel = object()

    def fake_get_instance(cls, *args, **kwargs):
        """Stand-in loader that installs a sentinel object instead of loading a model."""
        cls._instance = sentinel
        return sentinel

    monkeypatch.setattr(ModelLoader, "get_instance", classmethod(fake_get_instance))
    ModelLoader._instance = None
    ModelLoader._instance_locked = False
    ModelLoader._active_generation_count = 0

    llm = ModelLoader.acquire_instance()
    assert llm is sentinel
    assert ModelLoader.is_instance_locked()

    with pytest.raises(RuntimeError, match="inference is active"):
        ModelLoader.reset_instance()

    ModelLoader.unlock_instance()
    assert not ModelLoader.is_instance_locked()
    assert ModelLoader._active_generation_count == 0


def test_generation_threads_use_guarded_model_acquisition():
    """Scenario: read the generation thread sources. Expected: both acquire the model through the guarded call and close their generators."""
    # Read the source text because this test checks code structure
    llm_source = Path("app/engine/llm_thread.py").read_text(encoding="utf-8")
    agentic_source = Path("app/engine/agentic_thread.py").read_text(encoding="utf-8")

    assert "ModelLoader.acquire_instance" in llm_source
    assert "ModelLoader.acquire_instance" in agentic_source
    assert "response_generator" in llm_source and 'getattr(response_generator, "close", None)' in llm_source
    assert "response_gen" in agentic_source and 'getattr(response_gen, "close", None)' in agentic_source


def test_app_state_queues_cross_thread_signal_emissions():
    """Scenario: read the app state source. Expected: it uses queued Qt invocation for cross-thread signals."""
    source = Path("app/state.py").read_text(encoding="utf-8")

    assert "QMetaObject.invokeMethod" in source
    assert "Qt.ConnectionType.QueuedConnection" in source
    assert "@pyqtSlot(str, object)" in source


def test_websocket_broadcast_prunes_failed_clients():
    """Scenario: broadcast to one failing and one healthy client. Expected: the healthy one gets the payload and the failing one is removed everywhere."""
    from app.engine.websocket_server import WebSocketServerManager

    class FailingClient:
        """Fake client whose send always fails."""
        async def send(self, payload):
            """Always raise to simulate a closed connection."""
            raise RuntimeError("closed")

    class GoodClient:
        """Fake client that records payloads it is sent."""
        def __init__(self):
            """Start with an empty list of received payloads."""
            self.payloads = []

        async def send(self, payload):
            """Store the payload."""
            self.payloads.append(payload)

    # Build a bare manager without starting a server
    manager = WebSocketServerManager.__new__(WebSocketServerManager)
    manager.clients = set()
    manager.client_metadata = {}
    manager.client_histories = {}
    manager._clients_lock = threading.Lock()

    failing = FailingClient()
    good = GoodClient()
    manager.clients.update({failing, good})
    manager.client_metadata[failing] = {"id": "bad"}
    manager.client_metadata[good] = {"id": "good"}
    manager.client_histories[failing] = []
    manager.client_histories[good] = []

    asyncio.run(manager._broadcast_notification("status_update", {"message": "ok"}))

    assert good in manager.clients
    assert good.payloads
    assert failing not in manager.clients
    assert failing not in manager.client_metadata
    assert failing not in manager.client_histories


def test_websocket_manager_exposes_thread_safe_client_count():
    """Scenario: inspect the client_count property source. Expected: it takes the clients lock."""
    from app.engine.websocket_server import WebSocketServerManager

    assert "client_count" in dir(WebSocketServerManager)
    source = inspect.getsource(WebSocketServerManager.client_count)
    assert "_clients_lock" in source
