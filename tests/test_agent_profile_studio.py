# ==============================================================================
# Program: test_agent_profile_studio.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Tests for app/ui/workspaces/agent_profile_studio.py — the PyQt6 swarm agent profile manager widget.
#          Coordinates with adjacent subsystems within the Karl Offline Lab
#          architecture.
# ==============================================================================

# What is tested: the Agent Profile Studio widget, the editor in which users manage swarm agent
# personas (id, name, icon, system prompt, temperature, token limit, tool permissions). Profiles
# are persisted by app.utils.swarm_agent_profiles; three built-ins (architect/coder/tester) exist.
"""Tests for app/ui/workspaces/agent_profile_studio.py — the PyQt6 swarm
agent profile manager widget."""

# Annotations postponed so the typing syntax works across interpreter versions.
from __future__ import annotations

# sys.argv feeds QApplication; tempfile/Path create the redirected profile store.
import sys
import tempfile
import unittest
from pathlib import Path
# patch redirects the profile file path.
from unittest.mock import patch

# Side-effect import: Qt platform setup. A QApplication is required for any widget.
import tests.qt_test_helper  # noqa: F401
from PyQt6.QtWidgets import QApplication

from app.ui.workspaces.agent_profile_studio import AgentProfileStudioWorkspace


class TestAgentProfileStudioWorkspace(unittest.TestCase):
    """Widget tests for the Agent Profile Studio (list of swarm agent profiles with an editor form)."""
    # Redirecting PROFILE_PATH matters: without it, tests that save profiles would modify the user's real
    # data/agent_profiles.json.
    def setUp(self):
        """Redirect profile storage to a temp file and create the widget."""
        # One shared QApplication per process; creating a second one would crash Qt.
        self.app = QApplication.instance() or QApplication(sys.argv)
        # Temporary directory holds the isolated profile JSON file.
        self._tmp = tempfile.TemporaryDirectory()
        self._patcher = patch(
            "app.utils.swarm_agent_profiles.PROFILE_PATH",
            str(Path(self._tmp.name) / "agent_profiles.json"),
        )
        self._patcher.start()

    def tearDown(self):
        """Stop patches and clean up."""
        self._patcher.stop()
        self._tmp.cleanup()

    # Startup state: three built-ins listed, the first (architect) preselected so the form is never empty.
    def test_loads_builtin_profiles_on_construction(self):
        """The list shows the three built-in profiles and selects the architect first."""
        widget = AgentProfileStudioWorkspace(None)
        self.assertEqual(widget._list_widget.count(), 3)
        # Item data role 256 is Qt.UserRole, where the widget stores the profile id.
        ids = {widget._list_widget.item(i).data(256) for i in range(3)}
        self.assertEqual(ids, {"architect", "coder", "tester"})
        self.assertEqual(widget._current_id, "architect")
        widget.deleteLater()

    # The form must always mirror the selected list entry.
    def test_selecting_a_profile_populates_the_form(self):
        """Selecting a profile fills the name and prompt fields; built-in profiles have a read-only id."""
        widget = AgentProfileStudioWorkspace(None)
        # Select the coder row by its stored id (list order is not assumed).
        for row in range(widget._list_widget.count()):
            if widget._list_widget.item(row).data(256) == "coder":
                widget._list_widget.setCurrentRow(row)
                break
        self.assertEqual(widget._current_id, "coder")
        self.assertEqual(widget._name_input.text(), "Coder")
        self.assertTrue(widget._prompt_input.toPlainText())
        # Built-in ids are read-only because other code refers to them by name (e.g. swarm stage defaults).
        self.assertTrue(widget._id_input.isReadOnly())  # builtin profiles keep their id
        widget.deleteLater()

    # "New Profile" begins a blank draft: no current id (so Save creates rather than overwrites), an
    # editable id field, and every tool permission off (least privilege by default).
    def test_new_profile_clears_the_form(self):
        """New Profile empties the form, makes the id editable and unchecks all tools."""
        widget = AgentProfileStudioWorkspace(None)
        widget._on_new_profile()
        self.assertEqual(widget._current_id, "")
        self.assertEqual(widget._name_input.text(), "")
        self.assertFalse(widget._id_input.isReadOnly())
        for checkbox in widget._permission_checkboxes().values():
            self.assertFalse(checkbox.isChecked())
        widget.deleteLater()

    # Round trip: create a profile through the form, save, and verify (a) the registry has it, (b) the
    # signal fired so other workspaces (Swarm Studio dropdowns) refresh, (c) it is not marked builtin.
    def test_save_profile_persists_and_emits_signal(self):
        """Saving stores a non-builtin profile and emits the profile-saved signal with its id."""
        widget = AgentProfileStudioWorkspace(None)
        # profile_saved(str) is the notification other tabs listen to.
        saved_names = []
        widget.profile_saved.connect(saved_names.append)

        # Fill the form the way a user would; field values are chosen to differ from the defaults.
        widget._on_new_profile()
        widget._id_input.setText("doc_writer")
        widget._name_input.setText("Doc Writer")
        widget._icon_input.setText("D")
        widget._prompt_input.setPlainText("Write clear, accurate documentation.")
        widget._temp_spin.setValue(0.4)
        widget._tokens_spin.setValue(1024)
        # Granting read access only; the other tools stay off.
        widget._chk_read.setChecked(True)
        widget._on_save_profile()

        self.assertEqual(saved_names, ["doc_writer"])
        self.assertIn("doc_writer", widget._profiles)
        self.assertEqual(widget._profiles["doc_writer"]["name"], "Doc Writer")
        self.assertFalse(widget._profiles["doc_writer"]["builtin"])
        widget.deleteLater()

    # Validation: ids become dictionary keys and file-name-like identifiers, so spaces are rejected.
    def test_save_profile_rejects_invalid_id(self):
        """An invalid id shows a warning box and nothing is saved."""
        widget = AgentProfileStudioWorkspace(None)
        widget._on_new_profile()
        widget._id_input.setText("bad id with spaces")
        # QMessageBox.warning is patched so no modal dialog blocks the test.
        with patch("app.ui.workspaces.agent_profile_studio.QMessageBox.warning") as mock_warn:
            widget._on_save_profile()
        mock_warn.assert_called_once()
        # Nothing may be stored when validation fails.
        self.assertNotIn("bad id with spaces", widget._profiles)
        widget.deleteLater()

    # Undo support: Revert reloads the stored profile into the form, dropping unsaved edits.
    def test_revert_restores_original_values_after_edit(self):
        """Revert discards unsaved edits and restores the stored values."""
        widget = AgentProfileStudioWorkspace(None)
        for row in range(widget._list_widget.count()):
            if widget._list_widget.item(row).data(256) == "tester":
                widget._list_widget.setCurrentRow(row)
                break
        # Remember the stored prompt, then overwrite the field with scratch text.
        original_prompt = widget._prompt_input.toPlainText()

        # Revert must bring the stored text back.
        widget._prompt_input.setPlainText("some unsaved scratch edit")
        widget._on_revert_profile()

        self.assertEqual(widget._prompt_input.toPlainText(), original_prompt)
        widget.deleteLater()


if __name__ == "__main__":
    unittest.main()
