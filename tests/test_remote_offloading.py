# ==============================================================================
# Program: test_remote_offloading.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising remote offloading functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

# Postponed annotation evaluation keeps the test helpers importable on older interpreters and
# avoids forward-reference problems in the nested fakes.
from __future__ import annotations

# types.ModuleType builds the fake `websockets` package tree injected via monkeypatch.
import types

# pytest provides fixtures (monkeypatch + the ModelLoader state fixture); MagicMock/patch replace
# the heavy llama-cpp and network collaborators so every test runs offline in milliseconds.
import pytest
from unittest.mock import MagicMock, patch


# Contract under test: "Remote offloading" lets Karl send prompts to another Karl machine over a
# websocket. If that machine is unreachable the client must never leave the user hanging -- it
# must tell the UI why (on_fallback) and keep the answer flowing from the local model with the
# same sampling settings. This is the main failure-path test of RemoteRPCModel.
def test_remote_rpc_timeout_disables_remote_and_streams_local_fallback(monkeypatch):
    """If the websocket handshake to the remote engine times out, RemoteRPCModel must (1) report the reason via on_fallback, (2) connect with the token in the URL and a relaxed (LAN self-signed) TLS context, and (3) stream the answer from the local fallback model with the caller's sampling parameters unchanged. The websockets package is faked entirely so no network is used."""
    # Imported inside the test so a missing optional dependency fails this test, not collection.
    from app.engine.remote_rpc_client import RemoteRPCModel

    # captured is a plain dict the fakes write into; the assertions at the bottom read it back.
    captured = {
        "fallback_reason": None,
        "local_prompt": None,
        "local_kwargs": None,
    }

    # The real client does `async with connect(...)`; raising in __aenter__ reproduces exactly what a
    # TLS/handshake timeout looks like to it.
    class FailingWebSocketContext:
        """Async context manager whose entry raises a handshake TimeoutError."""
        async def __aenter__(self):
            """Raise TimeoutError('handshake timeout')."""
            raise TimeoutError("handshake timeout")

        async def __aexit__(self, exc_type, exc, tb):
            """Do not suppress exceptions."""
            return False

    # Stand-in for websockets.asyncio.client.connect. Recording url/ssl/close_timeout lets the test
    # verify how the client dials without opening a socket.
    def fake_connect(url, ssl=None, close_timeout=None):
        """Replacement for websockets connect: record the URL, TLS context and close timeout, then return the failing context."""
        captured["connect_url"] = url
        captured["ssl_context"] = ssl
        captured["close_timeout"] = close_timeout
        return FailingWebSocketContext()

    # Install stub websockets modules in sys.modules so the client's lazy imports resolve to the fakes
    # below; no real network library is touched.
    monkeypatch.setitem(__import__("sys").modules, "websockets", types.ModuleType("websockets"))
    monkeypatch.setitem(__import__("sys").modules, "websockets.asyncio", types.ModuleType("websockets.asyncio"))
    client_module = types.ModuleType("websockets.asyncio.client")
    client_module.connect = fake_connect
    monkeypatch.setitem(__import__("sys").modules, "websockets.asyncio.client", client_module)

    # The fallback model is called like llama-cpp-python (callable, stream=True yields chunk dicts).
    # Yielding one OpenAI-style chunk lets the test assert the stream is passed through untouched.
    class LocalFallbackModel:
        """Local model double that records the prompt and kwargs it was called with."""
        def __call__(self, prompt, **kwargs):
            """Capture the prompt and kwargs and stream a single 'local answer' chunk."""
            captured["local_prompt"] = prompt
            captured["local_kwargs"] = kwargs
            yield {"choices": [{"text": "local answer", "finish_reason": "stop"}]}

    def on_fallback(reason):
        """Record the reason the remote client gave for falling back."""
        captured["fallback_reason"] = reason

    # handshake_timeout is tiny so the failure path is exercised immediately.
    # local_fallback_factory is lazy: the local model is only created if the remote actually fails.
    # handshake_timeout is tiny so the failure path is exercised immediately.
    model = RemoteRPCModel(
        server_url="wss://192.168.1.50:8080",
        auth_token="secret-token",
        on_fallback=on_fallback,
        local_fallback_factory=lambda: LocalFallbackModel(),
        handshake_timeout=0.01,
    )

    # Consuming the generator drives the whole connect -> fail -> fallback sequence. The unusual
    # sampling kwargs (123 tokens, 0.25 temperature...) are chosen so that any silently dropped or
    # defaulted parameter would be caught by the equality checks below.
    chunks = list(model(
        "hello remote",
        stream=True,
        max_tokens=123,
        temperature=0.25,
        top_p=0.8,
        stop=["<stop>"],
        echo=False,
    ))

    # Output came entirely from the local model.
    assert chunks == [{"choices": [{"text": "local answer", "finish_reason": "stop"}]}]
    # The auth token travels as a query parameter on the websocket URL.
    assert captured["connect_url"] == "wss://192.168.1.50:8080?token=secret-token"
    assert captured["close_timeout"] == 2
    # LAN peers use self-signed certificates, so hostname checking is relaxed for the remote engine.
    assert captured["ssl_context"].check_hostname is False
    assert "handshake timeout" in captured["fallback_reason"]
    assert captured["local_prompt"] == "hello remote"
    assert captured["local_kwargs"]["max_tokens"] == 123
    assert captured["local_kwargs"]["temperature"] == 0.25
    assert captured["local_kwargs"]["top_p"] == 0.8
    assert captured["local_kwargs"]["stream"] is True
    assert captured["local_kwargs"]["stop"] == ["<stop>"]


# Persistence matters: without writing remote_engine_enabled=False to disk, every later request
# (or the next app launch) would pay the connect timeout again against a dead machine.
def test_model_loader_remote_fallback_persists_disabled(monkeypatch):
    """ModelLoader._remote_fallback must persist remote_engine_enabled=False (keeping url/token), drop the cached remote instance and remember the reason, so a dead remote is not retried on every request or after restart."""
    from app.engine.model_loader import ModelLoader

    # calls records each (enabled, url, token) written by the config-store stub.
    calls = []

    # Fake config so no real data/*.json file is read or written during the test.
    monkeypatch.setattr(
        "app.engine.config_store.get_engine_config",
        lambda: {
            "remote_engine_enabled": True,
            "remote_engine_url": "wss://localhost:8080",
            "remote_engine_token": "token",
            "engine_mode": "remote",
            "remote_server_url": "wss://localhost:8080",
            "remote_auth_token": "token",
        },
    )
    # The `... or True` makes the lambda return a truthy success value after recording the call, like
    # the real set_remote_engine_config returns True on success.
    monkeypatch.setattr(
        "app.engine.config_store.set_remote_engine_config",
        lambda enabled, url=None, token=None: calls.append((enabled, url, token)) or True,
    )

    # ModelLoader holds class-level state, so the originals are saved and restored in finally to stop
    # this test from leaking a fake remote into other tests.
    original_remote = ModelLoader._remote_instance
    original_reason = ModelLoader._remote_fallback_reason
    try:
        # Pretend a remote client is cached so the fallback has something to drop.
        ModelLoader._remote_instance = object()
        ModelLoader._remote_fallback("connection dropped")

        # URL and token are kept (only the enabled flag flips) so the user can re-enable with one click.
        assert calls == [(False, "wss://localhost:8080", "token")]
        # The cached client is dropped so get_instance() will not hand out the dead remote again.
        assert ModelLoader._remote_instance is None
        assert ModelLoader.last_remote_fallback_reason() == "connection dropped"
    finally:
        ModelLoader._remote_instance = original_remote
        ModelLoader._remote_fallback_reason = original_reason


# ── ModelLoader.get_instance() remote-mode wiring ────────────────────────────
# (Section map: the fixture and patch helper below are shared by the five get_instance tests;
# each test pins one rule of the routing: remote on, cache hit, cache invalidation, remote off,
# and mid-stream fallback.)
# ── ModelLoader.get_instance() remote-mode wiring ────────────────────────────
# These tests exercise the branch that get_instance() takes when remote engine
# mode is enabled: it must construct/reuse a RemoteRPCModel and return it
# *without* touching the local llama-cpp load path, and must fall through to
# the exact unmodified local path when remote mode is disabled (the default).

# Fixture, not setup/teardown, so tests opt in by naming it as an argument.
@pytest.fixture
def isolated_model_loader_remote_state():
    """Fixture: blank ModelLoader class-level state (local and remote) and restore the saved values afterwards, because the loader is a class-level singleton."""
    from app.engine.model_loader import ModelLoader

    # Snapshot every class attribute that get_instance() or the remote branch can mutate.
    saved = {
        "_instance": ModelLoader._instance,
        "_model_path": ModelLoader._model_path,
        "_model_name": ModelLoader._model_name,
        "_active_adapter": ModelLoader._active_adapter,
        "_n_ctx": ModelLoader._n_ctx,
        "_instance_locked": ModelLoader._instance_locked,
        "_active_generation_count": ModelLoader._active_generation_count,
        "_draft_instance": ModelLoader._draft_instance,
        "_draft_model_path": ModelLoader._draft_model_path,
        "_remote_instance": ModelLoader._remote_instance,
        "_remote_fallback_reason": ModelLoader._remote_fallback_reason,
    }
    # Reset to the "nothing loaded" state; _n_ctx 2048 is the loader's default context size.
    ModelLoader._instance = None
    ModelLoader._model_path = None
    ModelLoader._model_name = None
    ModelLoader._active_adapter = None
    ModelLoader._n_ctx = 2048
    ModelLoader._instance_locked = False
    ModelLoader._active_generation_count = 0
    ModelLoader._draft_instance = None
    ModelLoader._draft_model_path = None
    ModelLoader._remote_instance = None
    ModelLoader._remote_fallback_reason = None

    # Everything before yield is setup; everything after is teardown that restores the snapshot.
    yield

    for key, value in saved.items():
        setattr(ModelLoader, key, value)


# Without these patches the local path would try to find a GGUF on disk, probe the GPU, and spawn
# threads. patch.multiple swaps each helper for a MagicMock returning a neutral value.
def _local_load_patches():
    """Patch ModelLoader's disk/GPU/thread helpers so the local llama-cpp load path can run entirely with mocks."""
    from app.engine.model_loader import ModelLoader
    return patch.multiple(
        ModelLoader,
        _resolve_model_path=MagicMock(return_value="data/models/mock.gguf"),
        _read_registry_n_ctx=MagicMock(return_value=2048),
        _registry_entry=MagicMock(return_value={}),
        preflight_model_load=MagicMock(return_value=None),
        _bench_vram_bandwidth=MagicMock(return_value=None),
        _attach_kv_cache=MagicMock(return_value=None),
        _inspect_adapter_vocab=MagicMock(return_value=None),
        _start_idle_watcher=MagicMock(return_value=None),
    )


# Rule 1: remote mode on -> get_instance() returns a RemoteRPCModel and never loads llama-cpp.
# This is the whole point of offloading: the weak machine must not also allocate VRAM.
def test_get_instance_routes_to_remote_when_enabled(isolated_model_loader_remote_state):
    """With remote mode enabled, get_instance() builds a RemoteRPCModel (with url, token and fallback callbacks), caches it and never touches the local Llama constructor."""
    from app.engine.model_loader import ModelLoader

    # MagicMock stands in for the remote client; server_url/auth_token are set because the cache check
    # compares them against the config.
    fake_remote = MagicMock(name="RemoteRPCModel-instance")
    fake_remote.server_url = "wss://example.lan:9090"
    fake_remote.auth_token = "tok-1"

    # Patch targets are names as imported inside model_loader (not the origin modules), which is where
    # the loader looks them up.
    with patch(
        "app.engine.model_loader.config_store.get_engine_config",
        return_value={
            "remote_engine_enabled": True,
            "remote_engine_url": "wss://example.lan:9090",
            "remote_engine_token": "tok-1",
            "engine_mode": "remote",
            "remote_server_url": "wss://example.lan:9090",
            "remote_auth_token": "tok-1",
        },
    ), patch(
        "app.engine.model_loader.RemoteRPCModel", return_value=fake_remote
    ) as remote_ctor, patch(
        "app.engine.model_loader.Llama"
    ) as llama_ctor:
        # First (and only) call builds the remote client.
        result = ModelLoader.get_instance()

    assert result is fake_remote
    assert ModelLoader._remote_instance is fake_remote
    # Remote mode must not also load a local model (that would defeat offloading and waste VRAM).
    # The local llama-cpp load path must never run in remote mode.
    llama_ctor.assert_not_called()
    remote_ctor.assert_called_once()
    # Positional args are (url, token); the callbacks are keyword-only and must be callable so the
    # client can notify the loader when it fails.
    _, ctor_args, ctor_kwargs = remote_ctor.mock_calls[0]
    assert ctor_args[0] == "wss://example.lan:9090"
    assert ctor_args[1] == "tok-1"
    assert callable(ctor_kwargs["on_fallback"])
    assert callable(ctor_kwargs["local_fallback_factory"])


# Rule 2: a cache hit. Reconnecting on every generation would add a TLS handshake per request.
def test_get_instance_reuses_cached_remote_instance_for_same_url_and_token(
    isolated_model_loader_remote_state,
):
    """Repeated get_instance() calls with unchanged url/token reuse one RemoteRPCModel instead of reconnecting every time."""
    from app.engine.model_loader import ModelLoader

    fake_remote = MagicMock(name="RemoteRPCModel-instance")
    fake_remote.server_url = "wss://example.lan:9090"
    fake_remote.auth_token = "tok-1"

    # Config is identical for both calls, so the second call must be served from cache.
    with patch(
        "app.engine.model_loader.config_store.get_engine_config",
        return_value={
            "remote_engine_enabled": True,
            "remote_engine_url": "wss://example.lan:9090",
            "remote_engine_token": "tok-1",
        },
    ), patch(
        "app.engine.model_loader.RemoteRPCModel", return_value=fake_remote
    ) as remote_ctor:
        # Two calls, one construction.
        first = ModelLoader.get_instance()
        second = ModelLoader.get_instance()

    assert first is fake_remote
    assert second is fake_remote
    remote_ctor.assert_called_once()


# Rule 3: cache invalidation. If the user edits the URL in System Config, a stale client pointing
# at the old host would send prompts to the wrong machine.
def test_get_instance_reconstructs_remote_instance_when_url_changes(
    isolated_model_loader_remote_state,
):
    """Changing the configured remote URL invalidates the cached client and a new RemoteRPCModel is built."""
    from app.engine.model_loader import ModelLoader

    first_remote = MagicMock(name="remote-1")
    first_remote.server_url = "wss://a.lan:9090"
    first_remote.auth_token = "tok"
    second_remote = MagicMock(name="remote-2")
    second_remote.server_url = "wss://b.lan:9090"
    second_remote.auth_token = "tok"

    # side_effect with a list returns the first config on call 1 and the second on call 2, simulating
    # the user changing settings between generations.
    with patch(
        "app.engine.model_loader.config_store.get_engine_config",
        side_effect=[
            {
                "remote_engine_enabled": True,
                "remote_engine_url": "wss://a.lan:9090",
                "remote_engine_token": "tok",
            },
            {
                "remote_engine_enabled": True,
                "remote_engine_url": "wss://b.lan:9090",
                "remote_engine_token": "tok",
            },
        ],
    ), patch(
        "app.engine.model_loader.RemoteRPCModel",
        side_effect=[first_remote, second_remote],
    ) as remote_ctor:
        first = ModelLoader.get_instance()
        second = ModelLoader.get_instance()

    assert first is first_remote
    assert second is second_remote
    # Two distinct constructions prove the cache key includes the URL.
    assert remote_ctor.call_count == 2


# Rule 4: the default. Most users never enable offloading, so the remote branch must be invisible
# for them; this test mirrors the circuit-breaker happy path to prove nothing else changed.
def test_get_instance_local_path_unchanged_when_remote_disabled(
    isolated_model_loader_remote_state,
):
    """Regression guard: with remote mode off (the default), get_instance()
    must still take the exact local llama-cpp load path — this reproduces the
    ModelLoader circuit-breaker tests' happy path to prove the new branch is
    a true no-op when disabled."""
    from app.engine.model_loader import ModelLoader

    # A fake local Llama object the patched constructor returns.
    loaded_model = MagicMock(name="local-llama-instance")

    # get_hardware_profile with no GPUs avoids the GPU-split branches.
    with _local_load_patches(), patch(
        "app.engine.model_loader.get_hardware_profile",
        return_value={"gpu_list": []},
    ), patch(
        "app.engine.model_loader.config_store.get_active_model",
        return_value={"filename": "mock.gguf", "adapter": None},
    ), patch(
        # remote_engine_enabled False is what config_store returns on a default install.
        "app.engine.model_loader.config_store.get_engine_config",
        return_value={"remote_engine_enabled": False, "engine_mode": "local"},
    # subprocess.run is patched because the local path may call nvidia-smi/system tools.
    ), patch(
        "subprocess.run", return_value=MagicMock(returncode=0),
    ), patch(
        "app.engine.model_loader.Llama", return_value=loaded_model,
    # The ctor patches double as spies: Llama called once, RemoteRPCModel never.
    ) as llama_ctor, patch(
        "app.engine.model_loader.RemoteRPCModel"
    ) as remote_ctor:
        result = ModelLoader.get_instance()

    assert result is loaded_model
    llama_ctor.assert_called_once()
    remote_ctor.assert_not_called()
    assert ModelLoader._remote_instance is None


# Rule 5: round trip of the failure path. After the remote dies mid-stream the client calls
# on_fallback (loader disables remote) and then local_fallback_factory (loader loads the local
# model through the normal path). This test checks the loader wires both callbacks that way.
def test_local_fallback_factory_disables_remote_and_reenters_local_path(
    isolated_model_loader_remote_state,
):
    """Simulates a mid-generation remote failure: RemoteRPCModel's own
    on_fallback/local_fallback_factory contract (exercised for real in
    test_remote_rpc_timeout_disables_remote_and_streams_local_fallback) is
    wired by get_instance() to disable remote mode and re-enter this same
    method, landing on the local load path."""
    from app.engine.model_loader import ModelLoader

    # Shared mutable config stands in for data/*.json on disk.
    engine_cfg = {
        "remote_engine_enabled": True,
        "remote_engine_url": "wss://example.lan:9090",
        "remote_engine_token": "tok-1",
        "remote_server_url": "wss://example.lan:9090",
        "remote_auth_token": "tok-1",
    }
    loaded_model = MagicMock(name="local-llama-instance")
    # Filled by fake_remote_ctor with the callbacks the loader passes in.
    captured = {}

    # Replaces the RemoteRPCModel constructor so the test can grab the callbacks without a real client.
    def fake_remote_ctor(url, token, on_fallback=None, local_fallback_factory=None, **_kw):
        """Fake RemoteRPCModel constructor that captures the on_fallback and local_fallback_factory callbacks and returns a mock remote."""
        captured["on_fallback"] = on_fallback
        captured["local_fallback_factory"] = local_fallback_factory
        remote = MagicMock(name="remote-instance")
        remote.server_url = url
        remote.auth_token = token
        return remote

    # Read-through so later reads see updates made by the set_remote_engine_config stub.
    def fake_get_engine_config():
        """Return a copy of the engine config, which set_remote_engine_config updates when remote mode is disabled."""
        # engine_cfg is shared with the set_remote_engine_config stub below, so disabling remote mode
        # is visible to later calls.
        # First call (remote-mode check) sees remote enabled; once
        # _remote_fallback() disables it, subsequent calls see it disabled —
        # mirroring what config_store.set_remote_engine_config(False, ...)
        # actually persists to disk.
        return dict(engine_cfg)

    with _local_load_patches(), patch(
        "app.engine.model_loader.get_hardware_profile",
        return_value={"gpu_list": []},
    ), patch(
        "app.engine.model_loader.config_store.get_active_model",
        return_value={"filename": "mock.gguf", "adapter": None},
    ), patch(
        "app.engine.model_loader.config_store.get_engine_config",
        side_effect=fake_get_engine_config,
    ), patch(
        "app.engine.model_loader.config_store.set_remote_engine_config",
        # Writing enabled=False into engine_cfg simulates persistence to disk.
        side_effect=lambda enabled, url=None, token=None: engine_cfg.update(
            {"remote_engine_enabled": enabled}
        ),
    ), patch(
        "subprocess.run", return_value=MagicMock(returncode=0),
    ), patch(
        "app.engine.model_loader.Llama", return_value=loaded_model,
    ) as llama_ctor, patch(
        "app.engine.model_loader.RemoteRPCModel", side_effect=fake_remote_ctor,
    ):
        # While enabled, the loader must return the remote and not touch llama-cpp.
        remote = ModelLoader.get_instance()
        assert remote.server_url == "wss://example.lan:9090"
        llama_ctor.assert_not_called()

        # Simulate the remote bridge failing mid-stream, exactly as
        # RemoteRPCModel._stream() does: call on_fallback(reason) first,
        # then local_fallback_factory().
        # Same order as RemoteRPCModel._stream: report the failure first, then ask for the local
        # model.
        captured["on_fallback"]("connection dropped")
        assert engine_cfg["remote_engine_enabled"] is False
        assert ModelLoader._remote_instance is None

        # After the fallback the factory must route through the local load path and give back the Llama mock.
        local_llm = captured["local_fallback_factory"]()

    assert local_llm is loaded_model
    llama_ctor.assert_called_once()
