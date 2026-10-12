"""Tests for the System Config "Machine Speak" checkbox.

Covers the toggle added to DefaultsPanelMixin's GENERATION DEFAULTS tab: it
must reflect AppState.enable_machine_speak on construction, and toggling it
must persist through AppState.save_to_disk() (the full _PERSIST_FIELDS
round-trip), not just live in memory -- config_store.get_ui_config() is what
core.interaction_loop._machine_speak_enabled() reads at prompt-build time.
"""
import tests.qt_test_helper  # noqa: F401  — must come first to init QApplication

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.state import AppState
from app.engine import config_store
from app.ui.workspaces.system_config import SystemConfigWorkspace


_HEAVY_PATCHES = [
    patch("app.state.RAGPipeline", return_value=MagicMock()),
    patch("app.state.MemoryManager", return_value=MagicMock()),
    patch("app.state.TraceLogger", return_value=MagicMock()),
    patch("app.state.TrainingCurator", return_value=MagicMock()),
    patch("app.state.ImageStore", return_value=MagicMock()),
]


class TestMachineSpeakToggle(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._orig_cwd = os.getcwd()
        os.chdir(self._tmpdir)
        config_store._registry_cache = None
        config_store._registry_cache_mtime = None
        self._mocks = [p.start() for p in _HEAVY_PATCHES]

    def tearDown(self):
        for p in _HEAVY_PATCHES:
            p.stop()
        os.chdir(self._orig_cwd)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_checkbox_reflects_current_state_on_build(self):
        state = AppState()
        state.enable_machine_speak = False
        workspace = SystemConfigWorkspace(state)
        self.assertFalse(workspace._machine_speak_check.isChecked())

        state2 = AppState()
        state2.enable_machine_speak = True
        workspace2 = SystemConfigWorkspace(state2)
        self.assertTrue(workspace2._machine_speak_check.isChecked())

    def test_toggling_on_persists_to_disk(self):
        # Default is now False (a malformed/unparseable compact trace from a
        # small model can leak raw JSON into the chat response -- see
        # app/state.py and app/engine/config_store.py), so exercise the
        # persistence path by toggling it ON rather than off.
        state = AppState()
        self.assertFalse(state.enable_machine_speak)
        workspace = SystemConfigWorkspace(state)
        self.assertFalse(workspace._machine_speak_check.isChecked())

        workspace._machine_speak_check.setChecked(True)

        self.assertTrue(state.enable_machine_speak)
        with open(config_store.UI_CONFIG_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertTrue(payload["enable_machine_speak"])

        # And the exact path core.interaction_loop._machine_speak_enabled()
        # reads at prompt-build time must reflect it too.
        self.assertTrue(config_store.get_ui_config()["enable_machine_speak"])

    def test_toggling_survives_a_fresh_appstate_reload(self):
        state = AppState()
        workspace = SystemConfigWorkspace(state)
        workspace._machine_speak_check.setChecked(True)

        reloaded = AppState()
        self.assertTrue(reloaded.enable_machine_speak)

    def test_row_is_registered_for_settings_search(self):
        state = AppState()
        workspace = SystemConfigWorkspace(state)
        labels = [label for label, _ in workspace._settings_rows]
        self.assertIn("Machine Speak", labels)


if __name__ == "__main__":
    unittest.main()
