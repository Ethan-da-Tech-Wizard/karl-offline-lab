# ==============================================================================
# Program: test_watchdog.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising watchdog functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import os
import time

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget


def test_llm_watchdog_terminates_slow_stream(monkeypatch):
    """If the token stream stalls longer than watchdog_timeout_seconds (0.1s here, stream sleeps 0.4s per token), LLMThread.run() must return promptly (<2s), close the stream, reset the model, unlock the loader, set its watchdog-error flag and emit an 'Inference Watchdog Timeout' error. This is what stops a frozen GPU/llama call from hanging the UI forever."""
    # Step 1: Validate input parameters, operational preconditions, and state
    app = QApplication.instance() or QApplication([])
    from app.engine.llm_thread import LLMThread
    from app.engine.model_loader import ModelLoader
    import app.engine.llm_thread as llm_thread

    class SlowStream:
        """Fake token stream that blocks 0.4s per item until closed, simulating a stalled generation."""
        def __init__(self):
            """Start open."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.closed = False

        def __iter__(self):
            """Return self."""
            # Step 1: Validate input parameters, operational preconditions, and state
            return self

        def __next__(self):
            """Sleep 0.4s per token; stop once closed, otherwise yield a late token."""
            # Step 1: Validate input parameters, operational preconditions, and state
            time.sleep(0.4)
            if self.closed:
                raise StopIteration
            return {"choices": [{"text": "late-token", "finish_reason": None}]}

        def close(self):
            """Mark the stream closed so the next __next__ ends iteration."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.closed = True

    class FakeLLM:
        """Fake llama model returning the slow stream and recording whether reset() was called."""
        def __init__(self):
            """Create the slow stream and clear the reset flag."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.stream = SlowStream()
            self.reset_called = False

        def __call__(self, *args, **kwargs):
            """Return the slow stream."""
            # Step 1: Validate input parameters, operational preconditions, and state
            return self.stream

        def tokenize(self, data, add_bos=False):
            """Cheap tokenizer returning a few items."""
            # Step 1: Validate input parameters, operational preconditions, and state
            return list(data[: max(1, min(len(data), 4))])

        def reset(self):
            """Record that the model was reset."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.reset_called = True

    fake_llm = FakeLLM()
    errors = []
    unlocked = []

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

    thread = LLMThread(
        system_prompt="system",
        chat_history=[],
        hyperparams={"max_tokens": 8, "watchdog_timeout_seconds": 0.1},
    )
    thread.error_occurred.connect(errors.append)

    started = time.perf_counter()
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
    """AgenticThread._cleanup_after_watchdog_timeout must set the stop and timed-out flags, close the active stream, reset the model and emit the exact documented timeout message (the UI and logs match on that text)."""
    # Step 1: Validate input parameters, operational preconditions, and state
    from app.engine.agentic_thread import AgenticThread

    class FakeLLM:
        """Fake model that only records reset()."""
        def __init__(self):
            """Clear the reset flag."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.reset_called = False

        def reset(self):
            """Record that the model was reset."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.reset_called = True

    class FakeStream:
        """Fake response generator that only records close()."""
        def __init__(self):
            """Start open."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.closed = False

        def close(self):
            """Record that the stream was closed."""
            # Step 1: Validate input parameters, operational preconditions, and state
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
    """On startup MainWindow restores an autosave checkpoint (selected tab, chat history, hyperparameters, model and adapter, rendered messages) and then deletes the checkpoint file so a stale recovery is not applied twice. All heavy widgets are replaced by light fakes."""
    # Step 1: Validate input parameters, operational preconditions, and state
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    from app.utils.memory_manager import MemoryManager
    from app.utils.session_tree import SessionTree
    import app.ui.main_window as main_window

    class FakeState(QObject):
        """Minimal AppState replacement exposing the signals and fields MainWindow reads."""
        state_changed = pyqtSignal(str, object)
        change_workspace_requested = pyqtSignal(int)
        append_to_workbench_input = pyqtSignal(str)
        replace_workbench_input = pyqtSignal(str)
        attach_image_to_workbench = pyqtSignal(str)
        set_workbench_hyperparams = pyqtSignal(dict)
        set_workbench_system_prompt = pyqtSignal(str)

        def __init__(self):
            """Set up the default model/adapter names and idle flags (generating, swarm_running) that MainWindow reads."""
            # Step 1: Validate input parameters, operational preconditions, and state
            super().__init__()
            self.memory = MemoryManager(sessions_dir=str(tmp_path))
            self.model_name = "none"
            self.adapter_name = None
            self.generating = False
            self.swarm_running = False

    class FakeSidebar(QWidget):
        """Sidebar fake that records which page was selected."""
        workspace_changed = pyqtSignal(int)

        def __init__(self):
            """Start with no page selected."""
            # Step 1: Validate input parameters, operational preconditions, and state
            super().__init__()
            self.selected = None

        def select(self, idx):
            """Record the selected page index and emit workspace_changed like the real sidebar."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.selected = idx
            self.workspace_changed.emit(idx)

    class FakeStatusBar(QWidget):
        """Status bar fake whose setters ignore their input."""
        def set_model(self, value):
            """Remember the model name."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.model = value

        def set_adapter(self, value):
            """Remember the adapter name."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.adapter = value

        def set_state(self, *args):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def set_context_stats(self, *args):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def set_load_stats(self, *args):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def set_bridge_status(self, *args):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

    class FakeChatView:
        """Chat view fake that records the messages restored into it."""
        def __init__(self):
            """Start with an empty message list."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self._messages = []

        def clear_display(self):
            """Empty the message list."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self._messages = []

        def _render_all(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

    class FakeWorkbench(QWidget):
        """Workbench fake holding chat history and hyperparameters restored from the checkpoint."""
        status_changed = pyqtSignal(str, bool)
        model_changed = pyqtSignal(str)
        adapter_changed = pyqtSignal(str)
        context_stats = pyqtSignal(int, int, int, int)
        appearance_requested = pyqtSignal()

        def __init__(self, state):
            """Keep the state, create an empty SessionTree chat history, empty hyperparameters, placeholder session ids and the fake chat view."""
            # Step 1: Validate input parameters, operational preconditions, and state
            super().__init__()
            self.state = state
            self.chat_history = SessionTree()
            self._hyperparams = {}
            self._thread = None
            self._session_id = "existing"
            self._current_session_file = "existing.json"
            self._chat_view = FakeChatView()

        def _new_session(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def _save_current_session(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def _toggle_all_huds(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def _toggle_reasoning(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def _toggle_sessions(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def _toggle_rag_hud(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def _toggle_context_hud(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def update_theme(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def _populate_branches_tree(self):
            """Record that the branches tree was repopulated."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.branches_populated = True

        def _refresh_sessions(self):
            """Record that the session list was refreshed."""
            # Step 1: Validate input parameters, operational preconditions, and state
            self.sessions_refreshed = True

        def on_close(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

    class FakeWorkspace(QWidget):
        """Generic stand-in for the other workspaces so MainWindow can construct without real UI code."""
        adapter_changed = pyqtSignal(str)
        appearance_changed = pyqtSignal()

        def __init__(self, *args, **kwargs):
            """Accept any arguments."""
            # Step 1: Validate input parameters, operational preconditions, and state
            super().__init__()

        def set_workbench(self, *args):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

        def _scan_adapters(self):
            """Ignore."""
            # Step 1: Validate input parameters, operational preconditions, and state
            pass

    tree = SessionTree()
    tree.add_message("user", "recover this")
    manager = MemoryManager(sessions_dir=str(tmp_path))
    manager.save_autosave_checkpoint(
        tree,
        active_workspace=3,
        model_settings={"model_name": "restored.gguf", "adapter_name": "adapter-a", "hyperparams": {"temperature": 0.2}},
    )

    monkeypatch.setattr(main_window, "AppState", FakeState)
    monkeypatch.setattr(main_window, "Sidebar", FakeSidebar)
    monkeypatch.setattr(main_window, "StatusBar", FakeStatusBar)
    monkeypatch.setattr(main_window, "WorkbenchWorkspace", FakeWorkbench)
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
        # Iterate over items, evaluate loop invariant, and update progress
        monkeypatch.setattr(main_window, name, FakeWorkspace)
    # Neutralise window setup steps that need a real display, network sockets or a model; this test
    # only cares about checkpoint restore.
    monkeypatch.setattr(main_window.MainWindow, "_setup_shortcuts", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_load_theme_config", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_init_websocket_server", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_init_model", lambda self: None)
    import app.utils.keychain_manager as keychain_manager
    monkeypatch.setattr(keychain_manager, "revoke_tokens", lambda: None)
    monkeypatch.setattr(keychain_manager, "save_cached_token", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_window.QTimer, "singleShot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_window.QMessageBox,
        "question",
        lambda *args, **kwargs: main_window.QMessageBox.StandardButton.Yes,
    )

    window = main_window.MainWindow()
    window._state.memory = manager
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
