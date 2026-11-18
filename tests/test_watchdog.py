# Program: karl-offline-lab - test_watchdog.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests for the inference watchdog in the language-model threads and for restoring and clearing the autosave checkpoint in the main window, using fakes for the model, stream and UI parts.
"""Tests for inference watchdog timeouts and autosave recovery in the main window."""
import os
import time

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget


def test_llm_watchdog_terminates_slow_stream(monkeypatch):
    # Reuse or create the Qt application object required by the threads.
    """Scenario: a model whose token stream is slower than the watchdog timeout.
    Expected: the run ends quickly, the stream is closed, the model is reset and unlocked, and a watchdog timeout error is emitted.
    """
    app = QApplication.instance() or QApplication([])
    from app.engine.llm_thread import LLMThread
    from app.engine.model_loader import ModelLoader
    import app.engine.llm_thread as llm_thread

    class SlowStream:
        """Fake token stream that sleeps before each token until it is closed."""
        def __init__(self):
            """Set up the fake's initial state."""
            self.closed = False

        def __iter__(self):
            """Return the stream itself as the iterator."""
            return self

        def __next__(self):
            """Wait 0.4 seconds, then end if closed or return one token chunk."""
            time.sleep(0.4)
            if self.closed:
                raise StopIteration
            return {"choices": [{"text": "late-token", "finish_reason": None}]}

        def close(self):
            """Mark the fake stream as closed."""
            self.closed = True

    class FakeLLM:
        """Fake model returning the slow stream and recording resets."""
        def __init__(self):
            """Set up the fake's initial state."""
            self.stream = SlowStream()
            self.reset_called = False

        def __call__(self, *args, **kwargs):
            """Return the slow stream for any prompt."""
            return self.stream

        def tokenize(self, data, add_bos=False):
            """Return a few placeholder tokens for the given bytes."""
            return list(data[: max(1, min(len(data), 4))])

        def reset(self):
            """Record that the model state was reset."""
            self.reset_called = True

    fake_llm = FakeLLM()
    errors = []
    unlocked = []

    # Replace the heavy model, prompt and cache pieces so no real model is needed.
    monkeypatch.setattr(
        llm_thread,
        "compile_and_reload",
        lambda module, path, notice_cb, logger: module,
    )
    monkeypatch.setattr(
        llm_thread.core.interaction_loop,
        "build_prompt",
        lambda system, history: "prompt",
    )
    monkeypatch.setattr(ModelLoader, "get_instance", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(ModelLoader, "lock_instance", lambda: None)
    monkeypatch.setattr(ModelLoader, "unlock_instance", lambda: unlocked.append(True))
    monkeypatch.setattr(ModelLoader, "context_limit", lambda: 4096)
    monkeypatch.setattr(ModelLoader, "model_name", lambda: "fake.gguf")
    monkeypatch.setattr(llm_thread, "kv_cache_stats", lambda llm, prompt: {})
    monkeypatch.setattr(llm_thread, "log_cache_stats", lambda *args, **kwargs: None)

    # A tiny watchdog timeout makes the slow fake stream trip it quickly.
    thread = LLMThread(
        system_prompt="system",
        chat_history=[],
        hyperparams={"max_tokens": 8, "watchdog_timeout_seconds": 0.1},
    )
    thread.error_occurred.connect(errors.append)

    started = time.perf_counter()
    # Call run directly on this thread instead of starting a QThread.
    thread.run()
    elapsed = time.perf_counter() - started
    app.processEvents()

    assert elapsed < 2.0
    assert fake_llm.stream.closed
    assert fake_llm.reset_called
    assert unlocked == [True]
    assert thread._watchdog_error_emitted
    assert any("Inference Watchdog Timeout" in msg for msg in errors)


def test_agentic_watchdog_cleanup_emits_exact_timeout():
    """Scenario: the agentic thread cleans up after a watchdog timeout.
    Expected: stop and timeout flags are set, the stream is closed, the model is reset, and the exact timeout error text is emitted once.
    """
    from app.engine.agentic_thread import AgenticThread

    class FakeLLM:
        """Fake model that only records whether it was reset."""
        def __init__(self):
            """Set up the fake's initial state."""
            self.reset_called = False

        def reset(self):
            """Record that the model state was reset."""
            self.reset_called = True

    class FakeStream:
        """Fake response generator that records being closed."""
        def __init__(self):
            """Set up the fake's initial state."""
            self.closed = False

        def close(self):
            """Mark the fake stream as closed."""
            self.closed = True

    errors = []
    fake_llm = FakeLLM()
    fake_stream = FakeStream()
    thread = AgenticThread(
        system_prompt="system",
        initial_history=[],
        hyperparams={"watchdog_timeout_seconds": 0.1},
    )
    thread._active_response_generator = fake_stream
    thread.error_occurred.connect(errors.append)

    thread._cleanup_after_watchdog_timeout(fake_llm)

    assert thread._stop_requested
    assert thread._watchdog_timed_out
    assert fake_stream.closed
    assert fake_llm.reset_called
    assert errors == [
        "Inference Watchdog Timeout: Token generation froze for more than 30s. "
        "Inference terminated safely."
    ]


def test_main_window_restores_and_clears_autosave_checkpoint(tmp_path, monkeypatch):
    # Use the offscreen Qt platform so no real window appears.
    """Scenario: an autosave checkpoint exists and the user agrees to recover it.
    Expected: the checkpoint file is removed, the saved workspace, chat history, settings, model and adapter are restored and the chat view shows the recovered message.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    from app.utils.memory_manager import MemoryManager
    from app.utils.session_tree import SessionTree
    import app.ui.main_window as main_window

    class FakeState(QObject):
        """Fake application state with the signals and fields the main window uses."""
        state_changed = pyqtSignal(str, object)
        change_workspace_requested = pyqtSignal(int)
        append_to_workbench_input = pyqtSignal(str)
        replace_workbench_input = pyqtSignal(str)
        attach_image_to_workbench = pyqtSignal(str)
        set_workbench_hyperparams = pyqtSignal(dict)
        set_workbench_system_prompt = pyqtSignal(str)

        def __init__(self):
            """Set up the fake's initial state."""
            super().__init__()
            self.memory = MemoryManager(sessions_dir=str(tmp_path))
            self.model_name = "none"
            self.adapter_name = None
            self.generating = False
            self.swarm_running = False

    class FakeSidebar(QWidget):
        """Fake sidebar that records the selected workspace and emits a change signal."""
        workspace_changed = pyqtSignal(int)

        def __init__(self):
            """Set up the fake's initial state."""
            super().__init__()
            self.selected = None

        def select(self, idx):
            """Remember the selected index and emit the change signal."""
            self.selected = idx
            self.workspace_changed.emit(idx)

    class FakeStatusBar(QWidget):
        """Fake status bar that accepts and mostly ignores updates."""
        def set_model(self, value):
            """Remember the model name shown."""
            self.model = value

        def set_adapter(self, value):
            """Remember the adapter name shown."""
            self.adapter = value

        def set_state(self, *args):
            """Ignore the update."""
            pass

        def set_context_stats(self, *args):
            """Ignore the update."""
            pass

        def set_load_stats(self, *args):
            """Ignore the update."""
            pass

        def set_bridge_status(self, *args):
            """Ignore the update."""
            pass

    class FakeChatView:
        """Fake chat view that keeps a simple list of messages."""
        def __init__(self):
            """Set up the fake's initial state."""
            self._messages = []

        def clear_display(self):
            """Clear the stored messages."""
            self._messages = []

        def _render_all(self):
            """Do nothing; rendering is not needed in the test."""
            pass

    class FakeWorkbench(QWidget):
        """Fake workbench holding a session tree, settings and fake chat view."""
        status_changed = pyqtSignal(str, bool)
        model_changed = pyqtSignal(str)
        adapter_changed = pyqtSignal(str)
        context_stats = pyqtSignal(int, int, int, int)
        appearance_requested = pyqtSignal()

        def __init__(self, state):
            """Set up the fake's initial state."""
            super().__init__()
            self.state = state
            self.chat_history = SessionTree()
            self._hyperparams = {}
            self._thread = None
            self._session_id = "existing"
            self._current_session_file = "existing.json"
            self._chat_view = FakeChatView()

        def _new_session(self):
            """Do nothing for a new session."""
            pass

        def _save_current_session(self):
            """Do nothing when saving the session."""
            pass

        def _toggle_all_huds(self):
            """Do nothing for the HUD toggle."""
            pass

        def _toggle_reasoning(self):
            """Do nothing for the reasoning toggle."""
            pass

        def _toggle_sessions(self):
            """Do nothing for the sessions toggle."""
            pass

        def _toggle_rag_hud(self):
            """Do nothing for the retrieval HUD toggle."""
            pass

        def _toggle_context_hud(self):
            """Do nothing for the context HUD toggle."""
            pass

        def update_theme(self):
            """Do nothing for a theme update."""
            pass

        def _populate_branches_tree(self):
            """Record that the branches tree was filled."""
            self.branches_populated = True

        def _refresh_sessions(self):
            """Record that the session list was refreshed."""
            self.sessions_refreshed = True

        def on_close(self):
            """Do nothing on close."""
            pass

    class FakeWorkspace(QWidget):
        """Generic fake for every other workspace the main window creates."""
        adapter_changed = pyqtSignal(str)
        appearance_changed = pyqtSignal()

        def __init__(self, *args, **kwargs):
            """Set up the fake's initial state."""
            super().__init__()

        def set_workbench(self, *args):
            """Ignore the workbench reference."""
            pass

        def _scan_adapters(self):
            """Do nothing for the adapter scan."""
            pass

    # Save a checkpoint with one message and model settings for the window to recover.
    tree = SessionTree()
    tree.add_message("user", "recover this")
    manager = MemoryManager(sessions_dir=str(tmp_path))
    manager.save_autosave_checkpoint(
        tree,
        active_workspace=3,
        model_settings={"model_name": "restored.gguf", "adapter_name": "adapter-a", "hyperparams": {"temperature": 0.2}},
    )

    # Swap real application parts for the fakes defined above.
    monkeypatch.setattr(main_window, "AppState", FakeState)
    monkeypatch.setattr(main_window, "Sidebar", FakeSidebar)
    monkeypatch.setattr(main_window, "StatusBar", FakeStatusBar)
    monkeypatch.setattr(main_window, "WorkbenchWorkspace", FakeWorkbench)
    # Every other workspace class is replaced by the generic fake.
    for name in (
        "PromptLabWorkspace",
        "KnowledgeBaseWorkspace",
        "VisionWorkbench",
        "TrainingStudioWorkspace",
        "EvalSuiteWorkspace",
        "SwarmStudioWorkspace",
        "SystemConfigWorkspace",
        "DocsWorkspace",
        "FlywheelStudioWorkspace",
    ):
        monkeypatch.setattr(main_window, name, FakeWorkspace)
    # Disable startup steps that need a real model, server, theme or shortcuts.
    monkeypatch.setattr(main_window.MainWindow, "_setup_shortcuts", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_load_theme_config", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_init_websocket_server", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_init_model", lambda self: None)
    import app.utils.keychain_manager as keychain_manager
    monkeypatch.setattr(keychain_manager, "revoke_tokens", lambda: None)
    monkeypatch.setattr(keychain_manager, "save_cached_token", lambda *args, **kwargs: None)
    # Skip delayed callbacks so the test controls when recovery runs.
    monkeypatch.setattr(main_window.QTimer, "singleShot", lambda *args, **kwargs: None)
    # Answer Yes automatically when asked whether to recover.
    monkeypatch.setattr(
        main_window.QMessageBox,
        "question",
        lambda *args, **kwargs: main_window.QMessageBox.StandardButton.Yes,
    )

    window = main_window.MainWindow()
    window._state.memory = manager
    # Run the recovery check directly and then verify what was restored.
    window._check_autosave_recovery()

    assert not os.path.exists(manager.autosave_path)
    assert window._sidebar.selected == 3
    assert window._workbench.chat_history[0]["content"] == "recover this"
    assert window._workbench._hyperparams["temperature"] == 0.2
    assert window._state.model_name == "restored.gguf"
    assert window._state.adapter_name == "adapter-a"
    assert window._workbench._chat_view._messages[0][1] == "recover this"

    window.close()
    app.processEvents()
