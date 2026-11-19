# ==============================================================================
# Program: test_architecture_audit.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising architecture audit functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import asyncio
import inspect
import threading
from pathlib import Path

import pytest


def test_model_loader_blocks_reset_during_active_inference(monkeypatch):
    """While a generation holds the model (acquire_instance), reset_instance raises 'inference is active'; after release the loader is unlocked and the active count is back to 0."""
    # Skips on machines without llama-cpp-python: importing ModelLoader pulls it in, and this test is about locking semantics only.
    pytest.importorskip("llama_cpp")
    from app.engine.model_loader import ModelLoader

    sentinel = object()

    def fake_get_instance(cls, *args, **kwargs):
        """Return a sentinel instead of loading a real model."""
        cls._instance = sentinel
        return sentinel

    # acquire_instance is the code under test; only the expensive get_instance (which would load a GGUF) is replaced, and monkeypatch restores it afterwards.
    monkeypatch.setattr(ModelLoader, "get_instance", classmethod(fake_get_instance))
    ModelLoader._instance = None
    ModelLoader._instance_locked = False
    # Class-level state is reset by hand because earlier tests in the same process may have left the loader locked or counted.
    ModelLoader._active_generation_count = 0

    llm = ModelLoader.acquire_instance()
    assert llm is sentinel
    assert ModelLoader.is_instance_locked()

    # Resetting (unloading) the model while a generation is streaming from it would free memory the C library is still reading; the loader must refuse instead.
    with pytest.raises(RuntimeError, match="inference is active"):
        ModelLoader.reset_instance()

    ModelLoader.unlock_instance()
    assert not ModelLoader.is_instance_locked()
    assert ModelLoader._active_generation_count == 0


def test_generation_threads_use_guarded_model_acquisition():
    """Source-level guard: LLMThread and AgenticThread must acquire the model through ModelLoader.acquire_instance and close their response generators; reverting that would reintroduce use-after-free crashes."""
    llm_source = Path("app/engine/llm_thread.py").read_text(encoding="utf-8")
    agentic_source = Path("app/engine/agentic_thread.py").read_text(encoding="utf-8")

    # Pinned by source text because exercising both threads needs a real model; this guard prevents a regression back to get_instance, which would not take the generation lock.
    assert "ModelLoader.acquire_instance" in llm_source
    assert "ModelLoader.acquire_instance" in agentic_source
    assert "response_generator" in llm_source and 'getattr(response_generator, "close", None)' in llm_source
    assert "response_gen" in agentic_source and 'getattr(response_gen, "close", None)' in agentic_source


def test_app_state_queues_cross_thread_signal_emissions():
    """Source-level guard: AppState emits cross-thread signals via QMetaObject.invokeMethod with a queued connection and a pyqtSlot, so worker threads never touch Qt objects directly."""
    source = Path("app/state.py").read_text(encoding="utf-8")

    assert "QMetaObject.invokeMethod" in source
    assert "Qt.ConnectionType.QueuedConnection" in source
    assert "@pyqtSlot(str, object)" in source


def test_websocket_broadcast_prunes_failed_clients():
    """A broadcast drops clients whose send() fails (from clients, metadata and histories) while delivering to healthy clients."""
    from app.engine.websocket_server import WebSocketServerManager

    class FailingClient:
        """Client double whose send() raises."""
        async def send(self, payload):
            """Raise to simulate a dead connection."""
            raise RuntimeError("closed")

    class GoodClient:
        """Client double that records what it receives."""
        def __init__(self):
            """Start with no payloads."""
            self.payloads = []

        async def send(self, payload):
            """Record the payload."""
            self.payloads.append(payload)

    # __new__ skips __init__ so no server thread, SSL setup or port binding happens; only the attributes _broadcast_notification touches are filled in by hand.
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

    # Broadcast is a coroutine, so it is driven to completion on a throwaway event loop.
    asyncio.run(manager._broadcast_notification("status_update", {"message": "ok"}))

    assert good in manager.clients
    assert good.payloads
    assert failing not in manager.clients
    assert failing not in manager.client_metadata
    # All three per-client dicts must be pruned together; leaving one behind would leak memory and make later broadcasts retry a dead socket.
    assert failing not in manager.client_histories


def test_websocket_manager_exposes_thread_safe_client_count():
    """Source-level guard: the manager exposes client_count and protects its client set with _clients_lock for cross-thread reads."""
    from app.engine.websocket_server import WebSocketServerManager

    # The client set is mutated by the asyncio server thread, so a count read from any other thread must take _clients_lock to avoid iterating a changing set.
    assert "client_count" in dir(WebSocketServerManager)
    source = inspect.getsource(WebSocketServerManager.client_count)
    assert "_clients_lock" in source
