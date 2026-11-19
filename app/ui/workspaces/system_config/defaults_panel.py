# ==============================================================================
# Program: defaults_panel.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Defaults Panel.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Defaults and Identity tabs of the System workspace: generation defaults,
# logging/security toggles, and the system-prompt editor with named presets.
# Most handlers write straight into the shared state object and persist at once.

from __future__ import annotations

import logging
import re

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QInputDialog, QMessageBox,
)

from .common import _hline, _row, _section
from core.default_prompts import DEFAULT_SYSTEM_PROMPT, load_prompt_presets, save_user_preset, delete_user_preset

logger = logging.getLogger("karl.system_config")

# Two kinds of setting live here, with different persistence paths. Generation
# defaults (temperature, top-p, max tokens) are NOT persisted by their handlers:
# they are only published to the Workbench when "apply defaults" is pressed.
# Everything else (logging, security, thermal, theme mode) is written to AppState
# the moment it changes and saved through _save_appearance_config_silent()
# (appearance_runtime.py); Machine Speak is the one exception, saved by AppState.
class DefaultsPanelMixin:
    """Mixin that builds the Defaults and Identity tabs and handles their change events."""
    def _build_params_tab(self) -> QWidget:
        """Build the Defaults tab: generation limits, toggles, logging and security settings with a search filter."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Single "panel" card holding every row so the search filter below can hide rows
        # without touching the section headings (those are outside the filtered list).
        params_panel = QWidget()
        params_panel.setObjectName("panel")
        pp_layout = QVBoxLayout(params_panel)
        pp_layout.setContentsMargins(12, 12, 12, 12)
        pp_layout.setSpacing(10)

        pp_layout.addWidget(_section("GENERATION DEFAULTS"))
        
        desc = QLabel(
            "These apply as defaults to new Workbench sessions. Each session can override them."
        )
        desc.setObjectName("lbl-muted")
        desc.setWordWrap(True)
        pp_layout.addWidget(desc)

        # Every row is registered in self._settings_rows so the search box can hide non-matching ones.
        # Settings Search
        search_layout = QHBoxLayout()
        search_layout.setSpacing(8)
        # Search lives at the top of the Defaults tab only: this is the one tab long
        # enough to need it; Identity, Theme and Hardware are single-screen.
        search_lbl = QLabel("Search Settings:")
        search_lbl.setObjectName("lbl-muted")
        search_lbl.setStyleSheet("font-size: 8.5pt;")
        # Filters by case-insensitive substring of the row label (see
        # _on_settings_search_changed); an empty box shows everything.
        self._settings_search_input = QLineEdit()
        self._settings_search_input.setPlaceholderText("type to filter settings...")
        self._settings_search_input.textChanged.connect(self._on_settings_search_changed)
        search_layout.addWidget(search_lbl)
        search_layout.addWidget(self._settings_search_input)
        pp_layout.addLayout(search_layout)
        pp_layout.addWidget(_hline())

        # Ranges: temperature 0-2, top-p 0-1, max tokens 64-8192 (matching what the Workbench accepts).
        self._temp_spin = QDoubleSpinBox()
        # Initial values (0.7 / 0.95 / 2048) are the common chat defaults; they are not
        # read from state here, because the spin boxes only feed the next
        # "apply defaults" publish, and the Workbench owns the live values.
        self._temp_spin.setRange(0.0, 2.0)
        self._temp_spin.setSingleStep(0.05)
        self._temp_spin.setValue(0.7)
        # 90 px fixed width for all numeric inputs so the value column lines up under
        # the 130 px label column from _row.
        self._temp_spin.setFixedWidth(90)

        self._topp_spin = QDoubleSpinBox()
        # Top-p is a cumulative probability, so 0-1; 0.05 steps give 20 useful positions.
        self._topp_spin.setRange(0.0, 1.0)
        self._topp_spin.setSingleStep(0.05)
        self._topp_spin.setValue(0.95)
        self._topp_spin.setFixedWidth(90)

        self._maxtok_spin = QSpinBox()
        # Max tokens bounds the reply length per turn (not the context window): 64 is
        # the smallest useful answer, 8192 a generous cap, and the 128 step keeps values
        # on the round token budgets users tend to type. The ranges must stay inside
        # what the Workbench accepts, or applying a default would be silently clamped.
        self._maxtok_spin.setRange(64, 8192)
        self._maxtok_spin.setSingleStep(128)
        self._maxtok_spin.setValue(2048)
        self._maxtok_spin.setFixedWidth(90)

        # Mirrors the checkbox on the Theme tab; _on_reduced_motion_changed keeps both in step.
        self._reduced_motion_check = QCheckBox("Disable active glowing animations")
        self._reduced_motion_check.setChecked(getattr(self.state, "reduced_motion", False))
        # Connected after setChecked for the same reason: the initial value comes from
        # state and must not trigger a save.
        self._reduced_motion_check.stateChanged.connect(self._on_reduced_motion_changed)

        # (label, row widget) pairs. The label here is the search key and must match the
        # visible row caption, otherwise searching for what the user sees finds nothing.
        self._settings_rows = []
        
        # Each row is created, registered for search and added in the same three lines;
        # the tuple label is the search key, kept identical to the visible caption.
        row1 = _row("Temperature", self._temp_spin)
        self._settings_rows.append(("Temperature", row1))
        pp_layout.addWidget(row1)

        row2 = _row("Top-P", self._topp_spin)
        self._settings_rows.append(("Top-P", row2))
        pp_layout.addWidget(row2)

        row3 = _row("Max Tokens", self._maxtok_spin)
        self._settings_rows.append(("Max Tokens", row3))
        pp_layout.addWidget(row3)

        # Default off: the dense JSON history format is opt-in because small models
        # may answer in kind (see the tooltip).
        self._machine_speak_check = QCheckBox("Compress chat history for the model")
        self._machine_speak_check.setChecked(getattr(self.state, "enable_machine_speak", False))
        self._machine_speak_check.setToolTip(
            "Machine Speak: rewrites chat history into a dense JSON trace format before "
            "sending it to the model, saving tokens. Capable/larger models handle this "
            "format fine, but small models (e.g. the default 1.5B tier) can struggle to "
            "reply in kind, producing garbled or raw-JSON output. Disable this if small-model "
            "chat responses look malformed."
        )
        self._machine_speak_check.stateChanged.connect(self._on_machine_speak_changed)

        # The row3b name is just a local; the order on screen
        # (generation, compression, accessibility, theme, logs, security) groups related
        # settings and the names carry no meaning.
        row3b = _row("Machine Speak", self._machine_speak_check)
        self._settings_rows.append(("Machine Speak", row3b))
        pp_layout.addWidget(row3b)

        row4 = _row("Accessibility", self._reduced_motion_check)
        self._settings_rows.append(("Accessibility", row4))
        pp_layout.addWidget(row4)

        # Theme mode (midnight/slate/ember) is separate from the preset palette chosen on the Theme tab.
        # Theme Mode settings row
        # Modes are the three background ramps defined in themes.get_theme_colors:
        # midnight (identity), slate (cool grey-blue) and ember (warm, orange accent).
        self._theme_mode_combo = QComboBox()
        self._theme_mode_combo.addItems(["midnight", "slate", "ember"])
        self._theme_mode_combo.setCurrentText(getattr(self.state, "theme_mode", "midnight"))
        # Connected after setCurrentText so building the tab does not recompile the app
        # stylesheet and write config as a side effect.
        self._theme_mode_combo.currentTextChanged.connect(self._on_theme_mode_changed)
        self._theme_mode_combo.setFixedWidth(120)

        row5 = _row("Theme", self._theme_mode_combo)
        self._settings_rows.append(("Theme", row5))
        pp_layout.addWidget(row5)

        self._log_rotation_spin = QSpinBox()
        # Megabytes per log file before rotation; the 1-500 range is a UI bound, the
        # logger applies whatever value is stored.
        self._log_rotation_spin.setRange(1, 500)
        self._log_rotation_spin.setSingleStep(5)
        self._log_rotation_spin.setValue(getattr(self.state, "log_rotation_size_mb", 10))
        self._log_rotation_spin.setFixedWidth(90)
        self._log_rotation_spin.valueChanged.connect(self._on_log_rotation_changed)

        row6 = _row("Log Size Limit (MB)", self._log_rotation_spin)
        self._settings_rows.append(("Log Size Limit (MB)", row6))
        pp_layout.addWidget(row6)

        self._log_retention_spin = QSpinBox()
        # Same 1-365 day bounds as the Observability tab's retention control; both write
        # state.log_retention_days, so whichever saves last wins.
        self._log_retention_spin.setRange(1, 365)
        self._log_retention_spin.setSingleStep(5)
        self._log_retention_spin.setValue(getattr(self.state, "log_retention_days", 30))
        self._log_retention_spin.setFixedWidth(90)
        self._log_retention_spin.valueChanged.connect(self._on_log_retention_changed)

        row7 = _row("Log Retention (Days)", self._log_retention_spin)
        self._settings_rows.append(("Log Retention (Days)", row7))
        pp_layout.addWidget(row7)

        pp_layout.addWidget(_hline())
        pp_layout.addWidget(_section("SECURITY"))

        # Security: with this on, the cached bridge token is wiped from the OS keychain on
        # exit (see tooltip), trading convenience for not leaving a credential at rest.
        self._single_session_auth_check = QCheckBox("Enforce Single-Session Authorization")
        self._single_session_auth_check.setChecked(getattr(self.state, "single_session_auth", False))
        self._single_session_auth_check.setToolTip(
            "Wipe cached bridge token from OS keychain immediately upon exiting the application."
        )
        self._single_session_auth_check.stateChanged.connect(self._on_single_session_auth_changed)

        row8 = _row("Session Security", self._single_session_auth_check)
        self._settings_rows.append(("Session Security", row8))
        pp_layout.addWidget(row8)

        # Applies the three generation defaults to the Workbench via _apply_defaults in workspace.py.
        # Sends the three values through state.set_workbench_hyperparams (a signal the
        # Workbench listens to), see _apply_defaults in workspace.py.
        apply_btn = QPushButton("apply defaults")
        apply_btn.setObjectName("btn-primary")
        apply_btn.setToolTip("Save and apply default generation limits")
        apply_btn.clicked.connect(self._apply_defaults)
        pp_layout.addWidget(apply_btn)

        layout.addWidget(params_panel)
        layout.addStretch()
        return w


    def _on_reduced_motion_changed(self, state):
        """Store the reduced-motion flag, mirror it to the Theme tab checkbox, then re-apply and save."""
        # Qt may deliver the state as int 2 or as an enum value depending on binding version;
        # the trailing `or state` makes any truthy value count as checked.
        # Every checkbox slot below normalises stateChanged's int. PyQt6 delivers
        # Qt.CheckState as an int (2 = Checked) from stateChanged, hence the numeric
        # comparison; here the trailing `or state` also counts any other non-zero value
        # as checked, while the sibling slots use `state is True` instead.
        is_checked = (state == 2 or state == Qt.CheckState.Checked.value or state)
        # Order: state, mirror checkbox (signals blocked so it cannot bounce back here),
        # then apply and save. The same flag is edited from the Theme tab, which does
        # the symmetric sync in _on_control_changed.
        self.state.reduced_motion = is_checked
        if hasattr(self, "_reduced_motion_check_app"):
            self._reduced_motion_check_app.blockSignals(True)
            self._reduced_motion_check_app.setChecked(is_checked)
            self._reduced_motion_check_app.blockSignals(False)
        self._apply_active_theme()
        self._save_appearance_config_silent()


    def _on_theme_mode_changed(self, text: str):
        """Switch the theme mode live by recompiling the application stylesheet, then persist it."""
        self.state.theme_mode = text
        from app.ui import themes
        from PyQt6.QtWidgets import QApplication
        # Applied straight to the QApplication instead of going through
        # appearance_changed because only the mode changed; a full _apply_active_theme
        # would also rewrite preset/accent state from the Theme tab controls.
        QApplication.instance().setStyleSheet(themes.get_theme_stylesheet(self.state))
        self._save_appearance_config_silent()


    def _on_log_rotation_changed(self, val):
        """Store the log rotation size limit (MB) and persist."""
        self.state.log_rotation_size_mb = val
        self._save_appearance_config_silent()


    def _on_log_retention_changed(self, val):
        """Store the log retention period (days) and persist."""
        self.state.log_retention_days = val
        self._save_appearance_config_silent()


    def _on_machine_speak_changed(self, state):
        """Store the Machine Speak flag and save through AppState itself."""
        is_checked = (state == 2 or state == Qt.CheckState.Checked.value or state is True)
        self.state.enable_machine_speak = is_checked
        # Machine Speak isn't in appearance_runtime's narrower save dict, so go
        # through AppState's own full _PERSIST_FIELDS round-trip instead of
        # _save_appearance_config_silent() (which would leave this unpersisted).
        # Round-trips through AppState's own persisted-field list, which includes
        # enable_machine_speak; the appearance save dict does not.
        self.state.save_to_disk()


    def _on_single_session_auth_changed(self, state):
        """Store the single-session-auth flag and persist."""
        is_checked = (state == 2 or state == Qt.CheckState.Checked.value or state is True)
        self.state.single_session_auth = is_checked
        self._save_appearance_config_silent()


    def _on_thermal_protection_enabled_changed(self, state):
        """Store the thermal-protection enabled flag and persist (wired from the hardware tab)."""
        is_checked = (state == 2 or state == Qt.CheckState.Checked.value or state is True)
        self.state.thermal_protection_enabled = is_checked
        self._save_appearance_config_silent()


    def _on_thermal_threshold_changed(self, val: int):
        """Store the thermal-protection temperature threshold and persist."""
        self.state.thermal_protection_threshold = val
        self._save_appearance_config_silent()

    # ── identity tab ──────────────────────────────────────────────────────────

    # Identity = who the model is told it is. The text box edits a draft; only
    # "apply system prompt" publishes it, and presets only load text into the draft.
    def _build_identity_tab(self) -> QWidget:
        """Build the Identity tab: preset picker and the editable system prompt."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        identity_panel = QWidget()
        identity_panel.setObjectName("panel")
        ip_layout = QVBoxLayout(identity_panel)
        ip_layout.setContentsMargins(12, 12, 12, 12)
        ip_layout.setSpacing(8)

        ip_layout.addWidget(_section("SYSTEM PROMPT"))

        desc = QLabel(
            "This prompt defines Karl's persona and is sent before every conversation."
        )
        desc.setObjectName("lbl-muted")
        desc.setWordWrap(True)
        ip_layout.addWidget(desc)

        # Presets row: pick, load into the editor, save the editor text as a new preset, or delete.
        # ── Presets row ────────────────────────────────────────────────────────
        # Presets: built-in keys start with "_" and are read-only; user presets are saved
        # through core.default_prompts and listed after a separator with a star.
        presets_row = QHBoxLayout()
        presets_row.setSpacing(6)

        presets_lbl = QLabel("Preset:")
        presets_lbl.setObjectName("lbl-muted")
        presets_lbl.setStyleSheet("font-size: 8.5pt;")
        # Narrow fixed widths (44/52/72/56 px) let the combo take all remaining width
        # in the row.
        presets_lbl.setFixedWidth(44)
        presets_row.addWidget(presets_lbl)

        self._preset_combo = QComboBox()
        self._preset_combo.setToolTip("Load a named system prompt into the editor below")
        # AdjustToContents so long preset labels are readable, with 160 px as the floor
        # so the row does not collapse when presets have short names.
        self._preset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._preset_combo.setMinimumWidth(160)
        self._reload_preset_combo()
        presets_row.addWidget(self._preset_combo, 1)

        # Loading a preset replaces the editor text without applying it, so an
        # accidental click cannot change the persona of a running chat.
        load_preset_btn = QPushButton("load")
        load_preset_btn.setObjectName("btn-ghost")
        load_preset_btn.setFixedWidth(52)
        load_preset_btn.setToolTip("Load selected preset into the editor (does not apply immediately)")
        load_preset_btn.clicked.connect(self._on_load_preset)
        presets_row.addWidget(load_preset_btn)

        save_preset_btn = QPushButton("save as…")
        save_preset_btn.setObjectName("btn-ghost")
        save_preset_btn.setFixedWidth(72)
        save_preset_btn.setToolTip("Save the current editor text as a new named preset")
        save_preset_btn.clicked.connect(self._on_save_preset)
        presets_row.addWidget(save_preset_btn)

        self._delete_preset_btn = QPushButton("delete")
        self._delete_preset_btn.setObjectName("btn-danger")
        self._delete_preset_btn.setFixedWidth(56)
        self._delete_preset_btn.setToolTip("Delete the selected user-defined preset (built-ins are protected)")
        self._delete_preset_btn.clicked.connect(self._on_delete_preset)
        presets_row.addWidget(self._delete_preset_btn)

        ip_layout.addLayout(presets_row)

        # Shows the selected preset's description (italic, muted).
        # Preset description label
        self._inference_preset_desc_lbl = QLabel("")
        self._inference_preset_desc_lbl.setObjectName("lbl-muted")
        self._inference_preset_desc_lbl.setWordWrap(True)
        self._inference_preset_desc_lbl.setStyleSheet("font-size: 8pt; font-style: italic;")
        ip_layout.addWidget(self._inference_preset_desc_lbl)
        # Connected after the first fill: _reload_preset_combo blocks signals, and the
        # explicit call below sets up the initial description.
        self._preset_combo.currentIndexChanged.connect(self._on_preset_combo_changed)
        # Populate the description for the initial selection.
        # Trigger once to populate description for the initial selection
        self._on_preset_combo_changed(self._preset_combo.currentIndex())

        ip_layout.addWidget(_hline())

        from PyQt6.QtWidgets import QTextEdit
        # Imported at point of use above: QTextEdit is needed only by this tab. The editor
        # is stretched (factor 1) so long prompts get all spare height.
        self._system_edit = QTextEdit()
        self._system_edit.setToolTip(
            "System prompt active during generation. Determines Karl's personality and guidelines."
        )
        # Show the persisted prompt if present, otherwise the built-in default.
        # Load currently-active prompt (persisted or default)
        # Imported lazily to avoid an import cycle at module load.
        from app.engine import config_store as _cs
        # A whitespace-only stored prompt counts as unset, so the built-in default
        # reappears instead of an empty persona.
        stored = _cs.get_ui_config().get("workbench_system_prompt", "")
        self._system_edit.setPlainText(stored if stored.strip() else DEFAULT_SYSTEM_PROMPT)
        ip_layout.addWidget(self._system_edit, 1)

        # Editing the text does nothing until this is pressed, so the user can load a
        # preset, tweak it and compare before the Workbench sees it.
        apply_btn = QPushButton("apply system prompt")
        apply_btn.setObjectName("btn-primary")
        apply_btn.setToolTip("Apply this system prompt to all future generations in this session")
        apply_btn.clicked.connect(self._apply_identity)
        ip_layout.addWidget(apply_btn)

        layout.addWidget(identity_panel, 1)
        return w

    # ── vision tab ────────────────────────────────────────────────────────────

    def _reload_preset_combo(self) -> None:
        """Rebuild the Presets combo box from the current merged preset dict."""
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        # Re-read on every rebuild so a preset just saved or deleted is reflected
        # immediately; _presets_data is also what the description/delete handlers use.
        self._presets_data: dict = load_prompt_presets()
        # Convention: a leading underscore marks a shipped preset. The same test guards
        # deletion in _update_delete_btn_state and _on_delete_preset.
        built_in_keys = [k for k in self._presets_data if k.startswith("_")]
        user_keys = [k for k in self._presets_data if not k.startswith("_")]

        for key in built_in_keys:
            label = self._presets_data[key].get("label", key)
            self._preset_combo.addItem(label, userData=key)

        if user_keys:
            # Separator index equals the count of built-ins because insertSeparator takes a
            # row index, placing the divider right after them.
            self._preset_combo.insertSeparator(len(built_in_keys))
            for key in user_keys:
                label = self._presets_data[key].get("label", key)
                self._preset_combo.addItem(f"★ {label}", userData=key)

        self._preset_combo.blockSignals(False)
        self._update_delete_btn_state()

    def _on_preset_combo_changed(self, index: int) -> None:
        """Update the description label when the combo selection changes."""
        if not hasattr(self, "_presets_data"):
            return
        key = self._preset_combo.itemData(index)
        # Separator rows have no item data (None), so they fall through to the else
        # and clear the description.
        if key and key in self._presets_data:
            desc = self._presets_data[key].get("description", "")
            self._inference_preset_desc_lbl.setText(desc)
        else:
            self._inference_preset_desc_lbl.setText("")
        self._update_delete_btn_state()

    def _update_delete_btn_state(self) -> None:
        """Enable Delete only for user-defined presets (non-underscore keys)."""
        if not hasattr(self, "_delete_preset_btn"):
            return
        idx = self._preset_combo.currentIndex()
        key = self._preset_combo.itemData(idx) if idx >= 0 else None
        # isinstance guard: separators and an empty combo return None.
        is_user = isinstance(key, str) and not key.startswith("_")
        self._delete_preset_btn.setEnabled(is_user)

    def _on_load_preset(self) -> None:
        """Load the selected preset text into the editor."""
        idx = self._preset_combo.currentIndex()
        key = self._preset_combo.itemData(idx)
        if not key or not hasattr(self, "_presets_data"):
            return
        entry = self._presets_data.get(key, {})
        prompt = entry.get("prompt", "")
        if prompt and hasattr(self, "_system_edit"):
            self._system_edit.setPlainText(prompt)

    def _on_save_preset(self) -> None:
        """Save the current editor text as a new user preset."""
        if not hasattr(self, "_system_edit"):
            return
        text = self._system_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty Prompt", "Cannot save an empty system prompt as a preset.")
            return
        label, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:",
        )
        if not ok or not label.strip():
            return
        label = label.strip()
        # Generate a safe key from the label
        # Key is a slug of the label (lowercase a-z, 0-9, _), so two labels differing only
        # in punctuation or case map to the same key and the later save overwrites the
        # earlier one.
        key = re.sub(r"[^a-z0-9_]", "_", label.lower()).strip("_") or "custom_preset"
        # Avoid collision with built-in keys
        if key.startswith("_"):
            key = key.lstrip("_")
        desc, ok2 = QInputDialog.getText(
            self, "Save Preset", "Short description (optional):",
        )
        if not ok2:
            desc = ""
        # save_user_preset returns False on I/O failure rather than raising, so the
        # else-branch shows the dialog; success reloads the combo so the new entry shows.
        if save_user_preset(key, label, desc.strip(), text):
            self._reload_preset_combo()
            # Select the newly saved preset
            for i in range(self._preset_combo.count()):
                if self._preset_combo.itemData(i) == key:
                    self._preset_combo.setCurrentIndex(i)
                    break
        else:
            QMessageBox.critical(self, "Save Failed", f"Could not save preset '{label}'. Check logs.")

    def _on_delete_preset(self) -> None:
        """Delete the selected user preset after confirmation."""
        idx = self._preset_combo.currentIndex()
        key = self._preset_combo.itemData(idx)
        if not key or key.startswith("_"):
            return
        entry = self._presets_data.get(key, {})
        label = entry.get("label", key)
        reply = QMessageBox.question(
            self,
            "Delete Preset",
            f"Delete preset '{label}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Deletion removes only user presets (built-ins are filtered out above); the
        # combo is rebuilt so the removed entry disappears and selection resets.
        if delete_user_preset(key):
            self._reload_preset_combo()
        else:
            QMessageBox.critical(self, "Delete Failed", f"Could not delete preset '{label}'. Check logs.")

    def _apply_identity(self):
        # Signal rather than a direct call: the Workbench owns the active prompt and
        # this tab must not reach into it. Stripped so stray whitespace is not sent as a
        # leading/trailing part of every conversation.
        self.state.set_workbench_system_prompt.emit(
            self._system_edit.toPlainText().strip()
        )


    def _on_settings_search_changed(self, text: str):
        # Rows are hidden, not removed, so their widgets keep their current values and
        # clearing the search restores the full list instantly.
        query = text.strip().lower()
        for label, row_widget in self._settings_rows:
            if not query or query in label.lower():
                row_widget.setVisible(True)
            else:
                row_widget.setVisible(False)


