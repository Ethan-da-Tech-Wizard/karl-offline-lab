# ==============================================================================
# Program: appearance_runtime.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Appearance Runtime.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Runtime half of the Theme tab: reacts to control changes, pushes the chosen
# appearance into shared app state, tells the shell to re-apply its stylesheet
# (appearance_changed signal), and persists the result to data/ui_config.json.

from __future__ import annotations

import logging
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox,
)

from app.engine import config_store
from app.ui.themes import THEMES, get_theme_colors

logger = logging.getLogger("karl.system_config")

# Runtime half of the Theme tab (widgets live in appearance_panel.py). Flow on any
# control change: _on_control_changed -> (120 ms debounce) -> _apply_active_theme
# -> state fields written -> appearance_changed signal -> main_window's
# _apply_theme_from_state recompiles and installs the QSS. Nothing is written to
# disk until Save (_save_appearance_config), so a live preview can be abandoned.
class AppearanceRuntimeMixin:
    """Mixin with the apply/preview/save logic for appearance settings."""
    def _on_control_changed(self):
        """Update slider read-outs, mirror reduced-motion, and schedule a debounced theme apply."""
        # Slider ints are tenths, so divide by 10 for the real multiplier.
        # Update sliders numeric indicators
        gs = self._glow_strength_slider.value() / 10.0
        ai = self._animation_intensity_slider.value() / 10.0
        self._glow_strength_val_lbl.setText(f"{gs:.1f}x")
        self._animation_intensity_val_lbl.setText(f"{ai:.1f}x")

        # Reduced motion exists as a checkbox on both the Defaults and Theme tabs; keep them equal.
        # Also sync reduced motion check in defaults tab
        self._reduced_motion_check.blockSignals(True)
        self._reduced_motion_check.setChecked(self._reduced_motion_check_app.isChecked())
        self._reduced_motion_check.blockSignals(False)

        # Debounce: slider drags fire valueChanged per tick; recompiling and
        # re-applying the app stylesheet on every tick stutters the drag.
        # Labels above update live, the theme applies once the drag settles.
        if not hasattr(self, "_theme_apply_timer"):
            from PyQt6.QtCore import QTimer
            self._theme_apply_timer = QTimer(self)
            self._theme_apply_timer.setSingleShot(True)
            self._theme_apply_timer.setInterval(120)
            self._theme_apply_timer.timeout.connect(self._apply_active_theme)
        self._theme_apply_timer.start()



    def _apply_active_theme(self):
        """Copy control values into app state and notify the shell to rebuild and apply the stylesheet."""
        # Controls are the source of truth here; state is written from them, not the reverse.
        # Read from controls
        # All reads happen before any write so a half-updated state is never visible to
        # listeners that react to the signal below.
        theme_preset = self._theme_preset_combo.currentText()
        layout_preset = self._layout_preset_combo.currentText()
        glow_enabled = self._glow_enabled_check.isChecked()
        reduced_motion = self._reduced_motion_check_app.isChecked()
        # Sliders hold integer tenths (0-20); dividing by 10.0 gives the 0.0-2.0
        # multiplier stored in state (glow_strength is read by GlowPanel, animation_intensity
        # by TracingPanel).
        glow_strength = self._glow_strength_slider.value() / 10.0
        animation_intensity = self._animation_intensity_slider.value() / 10.0
        custom_accent = self._active_custom_accent

        # Write to state
        self.state.theme_preset = theme_preset
        self.state.layout_preset = layout_preset
        self.state.glow_enabled = glow_enabled
        self.state.reduced_motion = reduced_motion
        self.state.glow_strength = glow_strength
        self.state.animation_intensity = animation_intensity
        self.state.custom_accent = custom_accent

        # The shell listens to this signal and recompiles the QApplication stylesheet.
        # Trigger QApplication stylesheet compilation & reload
        # Payload-free signal on purpose: receivers re-read state, so the signal cannot
        # carry stale values if two changes land close together.
        self.appearance_changed.emit()

        # Preview widgets paint from state, so they must be told to repaint.
        # Update preview card elements
        # The preview card and icons are not styled by QSS alone (custom-painted), so
        # they need an explicit repaint; then the swatches are rebuilt from the same state.
        self._preview_card.update_style()
        self._check_icon_preview.update()
        self._cross_icon_preview.update()
        self._update_swatches()



    def _update_swatches(self):
        """Rebuild the five colour swatches (accent, alt, surface, deep, text) from the current theme."""
        # deleteLater (not immediate delete) is the safe way to discard widgets still in the layout.
        # Clear existing swatches
        # takeAt(0) detaches the layout item; deleteLater schedules the widget for
        # deletion once control returns to the event loop, which is safe even if this
        # runs inside a signal emitted by one of those widgets.
        while self._swatches_layout.count():
            item = self._swatches_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # get_theme_colors resolves preset + custom accent + mode into one palette dict.
        # Generate colors for current state
        colors = get_theme_colors(self.state)
        # (palette key, caption) pairs: five roles are enough to judge a theme at a glance
        # (highlight, its darker partner, two background depths, primary text).
        swatch_keys = [
            ("accent", "ACCENT"),
            ("accent_dark", "ALT"),
            ("bg_surface", "SURFACE"),
            ("bg_deep", "DEEP"),
            ("text_hi", "TEXT")
        ]
        
        for key, label in swatch_keys:
            hex_val = colors.get(key, "#000000")
            
            swatch_unit = QWidget()
            su_layout = QVBoxLayout(swatch_unit)
            su_layout.setContentsMargins(0, 0, 0, 0)
            su_layout.setSpacing(2)
            
            color_box = QLabel()
            # 36x24 px chips with a 1 px border in the theme border colour, so dark
            # background swatches stay visible against a dark panel.
            color_box.setFixedSize(36, 24)
            color_box.setStyleSheet(
                f"background-color: {hex_val}; border: 1px solid {colors.get('border', '#35356E')}; border-radius: 3px;"
            )
            color_box.setToolTip(f"{label}: {hex_val}")
            
            lbl_text = QLabel(label)
            lbl_text.setStyleSheet("font-size: 7.5pt; font-weight: bold; text-align: center;")
            lbl_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_text.setObjectName("lbl-muted")
            
            su_layout.addWidget(color_box, alignment=Qt.AlignmentFlag.AlignCenter)
            su_layout.addWidget(lbl_text, alignment=Qt.AlignmentFlag.AlignCenter)
            
            self._swatches_layout.addWidget(swatch_unit)



    def _update_theme_gallery(self):
        """Rebuild the theme gallery cards, skipping the work if the selection has not changed."""
        if not hasattr(self, "_theme_gallery_layout"):
            return
        # The 20 cards only differ by which one carries the ACTIVE marker;
        # skip the full rebuild when the selection hasn't changed.
        # Cache key is just the selected name: the card contents depend only on THEMES
        # (static) and on which card is marked ACTIVE. Rebuilding on every slider tick
        # would destroy and recreate every card while the user drags.
        selected_now = self._theme_preset_combo.currentText()
        if (getattr(self, "_gallery_built_for", None) == selected_now
                and self._theme_gallery_layout.count() > 0):
            return
        self._gallery_built_for = selected_now
        while self._theme_gallery_layout.count():
            item = self._theme_gallery_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        selected = self._theme_preset_combo.currentText()
        # Cards read colours straight from the raw THEMES entry (not get_theme_colors) so
        # each card previews its own theme regardless of the current mode or accent
        # override. .get() defaults mirror the Karl Obsidian Core palette for entries
        # that omit a key.
        for name, data in THEMES.items():
            border = data.get("border_hi", "#35356E")
            accent = data.get("accent", "#00E5FF")
            accent_alt = data.get("accent_alt", "#0099AA")
            bg_deep = data.get("bg_deep", "#020205")
            bg_surface = data.get("bg_surface", "#0D0D1B")
            bg_raised = data.get("bg_raised", "#14142D")
            text_hi = data.get("text_hi", "#F0F5FF")
            text_mid = data.get("text_mid", "#A0AEC0")

            card = QWidget()
            card.setObjectName("panel")
            card.setStyleSheet(
                f"QWidget#panel {{ background-color: {bg_surface}; border: 1px solid {border}; border-radius: 6px; }}"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(10, 8, 10, 8)
            cl.setSpacing(6)

            header = QWidget()
            hl = QHBoxLayout(header)
            hl.setContentsMargins(0, 0, 0, 0)
            title = QLabel(name + ("  ACTIVE" if name == selected else ""))
            title.setStyleSheet(f"color: {accent}; font-weight: bold; font-size: 9.5pt;")
            hl.addWidget(title, 1)
            apply_btn = QPushButton("Apply")
            apply_btn.setObjectName("btn-primary" if name == selected else "btn-secondary")
            # Default-arg capture (n=name) binds each loop value at definition time; a plain
            # closure would make every button apply the last theme. _checked swallows the
            # bool Qt passes for clicked.
            apply_btn.clicked.connect(lambda _checked=False, n=name: self._apply_theme_preset_from_gallery(n))
            hl.addWidget(apply_btn)
            cl.addWidget(header)

            desc = QLabel(data.get("description", ""))
            desc.setWordWrap(True)
            desc.setStyleSheet(f"color: {text_mid}; font-size: 8.2pt;")
            cl.addWidget(desc)

            swatch_row = QWidget()
            sl = QHBoxLayout(swatch_row)
            sl.setContentsMargins(0, 0, 0, 0)
            sl.setSpacing(5)
            # Six chips per card: wider palette than the live swatches because here the
            # user is choosing between themes and needs the raised/surface/deep elevations.
            for value, label in (
                (accent, "accent"),
                (accent_alt, "alt"),
                (bg_deep, "deep"),
                (bg_surface, "surface"),
                (bg_raised, "raised"),
                (text_hi, "text"),
            ):
                box = QLabel()
                box.setFixedSize(34, 16)
                box.setToolTip(f"{label}: {value}")
                box.setStyleSheet(f"background-color: {value}; border: 1px solid {border}; border-radius: 3px;")
                sl.addWidget(box)
            sl.addStretch()
            cl.addWidget(swatch_row)
            self._theme_gallery_layout.addWidget(card)

        # Trailing stretch keeps the cards top-aligned when fewer cards than the
        # viewport height are shown.
        self._theme_gallery_layout.addStretch(1)



    def _apply_theme_preset_from_gallery(self, name: str):
        """Select the given preset in the combo; the combo's change signal does the actual apply."""
        # Going through the combo (not applying directly) keeps one code path: the
        # combo's currentTextChanged runs _on_preset_changed, which updates the
        # description, accent preview, theme and gallery.
        idx = self._theme_preset_combo.findText(name)
        if idx >= 0:
            self._theme_preset_combo.setCurrentIndex(idx)



    def _save_appearance_config_silent(self):
        """Persist all appearance and related runtime settings to ui_config without any dialog."""
        # The dict also carries unrelated keys (log rotation, thermal, remote engine) because
        # save_ui_config replaces the stored config as a whole; omitting them would reset them.
        # Keys here must cover everything save_ui_config validates, because it replaces
        # the stored config whole. Note that the remote token is persisted in this file
        # alongside the URL (the Hardware tab also calls set_remote_engine_config), so
        # ui_config.json should be treated as sensitive on shared machines.
        config_store.save_ui_config({
            "theme_preset": self.state.theme_preset,
            "custom_accent": self.state.custom_accent,
            "layout_preset": self.state.layout_preset,
            "reduced_motion": self.state.reduced_motion,
            "glow_enabled": self.state.glow_enabled,
            "animation_intensity": self.state.animation_intensity,
            "glow_strength": self.state.glow_strength,
            "theme_mode": getattr(self.state, "theme_mode", "midnight"),
            "log_rotation_size_mb": getattr(self.state, "log_rotation_size_mb", 10),
            "log_retention_days": getattr(self.state, "log_retention_days", 30),
            "single_session_auth": getattr(self.state, "single_session_auth", False),
            "thermal_protection_enabled": getattr(self.state, "thermal_protection_enabled", True),
            "thermal_protection_threshold": getattr(self.state, "thermal_protection_threshold", 95),
            "engine_mode": getattr(self.state, "engine_mode", "local"),
            "remote_server_url": getattr(self.state, "remote_server_url", ""),
            "remote_auth_token": getattr(self.state, "remote_auth_token", ""),
            "remote_engine_enabled": getattr(self.state, "remote_engine_enabled", False),
            "remote_engine_url": getattr(self.state, "remote_engine_url", "wss://localhost:8080"),
            "remote_engine_token": getattr(self.state, "remote_engine_token", ""),
        })



    def _save_appearance_config(self):
        """Save the appearance config, remove the legacy theme_config.json and confirm to the user."""
        self._save_appearance_config_silent()
        
        # data/theme_config.json is the old storage location; ui_config.json supersedes it.
        # Delete old config if exists to clean up
        # get_ui_config falls back to this legacy file only when ui_config.json is
        # missing; after a successful save the new file exists, so the old one can go.
        old_path = "data/theme_config.json"
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            # Failing to remove the legacy file is harmless, so the error is ignored.
            except Exception:
                pass
                
        QMessageBox.information(
            self, "Settings Saved",
            "Appearance configuration saved successfully to data/ui_config.json."
        )
