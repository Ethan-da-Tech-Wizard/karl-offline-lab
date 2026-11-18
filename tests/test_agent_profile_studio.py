# Program: karl-offline-lab - test_agent_profile_studio.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests for the Qt agent profile studio widget: loading, selecting, creating, saving and reverting profiles.
"""Tests for app/ui/workspaces/agent_profile_studio.py — the PyQt6 swarm
agent profile manager widget."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tests.qt_test_helper  # noqa: F401
from PyQt6.QtWidgets import QApplication

from app.ui.workspaces.agent_profile_studio import AgentProfileStudioWorkspace


class TestAgentProfileStudioWorkspace(unittest.TestCase):
    """Widget tests for AgentProfileStudioWorkspace using a temporary profile file."""
    def setUp(self):
        """Ensure a QApplication exists and redirect profile storage to a temporary directory."""
        self.app = QApplication.instance() or QApplication(sys.argv)
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch(
            "app.utils.swarm_agent_profiles.PROFILE_PATH",
            str(Path(self._tmp.name) / "agent_profiles.json"),
        )
        self._patcher.start()

    def tearDown(self):
        """Stop the patch and delete the temporary directory."""
        self._patcher.stop()
        self._tmp.cleanup()

    def test_loads_builtin_profiles_on_construction(self):
        """Scenario: construct the widget. Expected: three built-in profiles are listed and the architect is selected."""
        widget = AgentProfileStudioWorkspace(None)
        self.assertEqual(widget._list_widget.count(), 3)
        ids = {widget._list_widget.item(i).data(256) for i in range(3)}
        self.assertEqual(ids, {"architect", "coder", "tester"})
        self.assertEqual(widget._current_id, "architect")
        widget.deleteLater()

    def test_selecting_a_profile_populates_the_form(self):
        """Scenario: select the coder profile. Expected: the form shows its name and prompt and the id field is read-only."""
        widget = AgentProfileStudioWorkspace(None)
        for row in range(widget._list_widget.count()):
            # 256 is the Qt user data role that holds the profile id
            if widget._list_widget.item(row).data(256) == "coder":
                widget._list_widget.setCurrentRow(row)
                break
        self.assertEqual(widget._current_id, "coder")
        self.assertEqual(widget._name_input.text(), "Coder")
        self.assertTrue(widget._prompt_input.toPlainText())
        self.assertTrue(widget._id_input.isReadOnly())  # builtin profiles keep their id
        widget.deleteLater()

    def test_new_profile_clears_the_form(self):
        """Scenario: click New. Expected: the form is blank, the id is editable and no permissions are checked."""
        widget = AgentProfileStudioWorkspace(None)
        widget._on_new_profile()
        self.assertEqual(widget._current_id, "")
        self.assertEqual(widget._name_input.text(), "")
        self.assertFalse(widget._id_input.isReadOnly())
        for checkbox in widget._permission_checkboxes().values():
            self.assertFalse(checkbox.isChecked())
        widget.deleteLater()

    def test_save_profile_persists_and_emits_signal(self):
        """Scenario: fill in a new profile and save. Expected: profile_saved fires with its id and the profile is stored as non-built-in."""
        widget = AgentProfileStudioWorkspace(None)
        saved_names = []
        widget.profile_saved.connect(saved_names.append)

        widget._on_new_profile()
        widget._id_input.setText("doc_writer")
        widget._name_input.setText("Doc Writer")
        widget._icon_input.setText("D")
        widget._prompt_input.setPlainText("Write clear, accurate documentation.")
        widget._temp_spin.setValue(0.4)
        widget._tokens_spin.setValue(1024)
        widget._chk_read.setChecked(True)
        widget._on_save_profile()

        self.assertEqual(saved_names, ["doc_writer"])
        self.assertIn("doc_writer", widget._profiles)
        self.assertEqual(widget._profiles["doc_writer"]["name"], "Doc Writer")
        self.assertFalse(widget._profiles["doc_writer"]["builtin"])
        widget.deleteLater()

    def test_save_profile_rejects_invalid_id(self):
        """Scenario: save with an id containing spaces. Expected: a warning dialog is shown and nothing is stored."""
        widget = AgentProfileStudioWorkspace(None)
        widget._on_new_profile()
        widget._id_input.setText("bad id with spaces")
        # Patch the dialog so the test does not block on user input
        with patch("app.ui.workspaces.agent_profile_studio.QMessageBox.warning") as mock_warn:
            widget._on_save_profile()
        mock_warn.assert_called_once()
        self.assertNotIn("bad id with spaces", widget._profiles)
        widget.deleteLater()

    def test_revert_restores_original_values_after_edit(self):
        """Scenario: edit the prompt then revert. Expected: the original prompt text returns."""
        widget = AgentProfileStudioWorkspace(None)
        for row in range(widget._list_widget.count()):
            if widget._list_widget.item(row).data(256) == "tester":
                widget._list_widget.setCurrentRow(row)
                break
        original_prompt = widget._prompt_input.toPlainText()

        widget._prompt_input.setPlainText("some unsaved scratch edit")
        widget._on_revert_profile()

        self.assertEqual(widget._prompt_input.toPlainText(), original_prompt)
        widget.deleteLater()


if __name__ == "__main__":
    unittest.main()
