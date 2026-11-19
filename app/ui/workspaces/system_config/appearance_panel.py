# ==============================================================================
# Program: appearance_panel.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Desktop workspace implementation for Appearance Panel.
#          Provides responsive layout controls, parameter adjustment surfaces,
#          and visual feedback components within Karl's PyQt6 shell.
# ==============================================================================
# Theme tab for the System workspace. This mixin builds the controls (preset,
# accent override, layout density, glow/motion) plus a live preview card; the
# runtime side that actually applies and persists settings lives in
# appearance_runtime.py (AppearanceRuntimeMixin), which shares `self`.

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QScrollArea, QComboBox, QColorDialog, QCheckBox, QSlider,
)

from app.engine import config_store
from app.ui.themes import THEMES, get_theme_colors
from app.ui.widgets.symbolic_icon import CheckIcon, CrossIcon
from app.ui.widgets.tracing_panel import TracingPanel
from .common import _hline, _row, _section

logger = logging.getLogger("karl.system_config")

# Division of labour: this mixin owns widget construction and the two-way sync
# between widgets and state; AppearanceRuntimeMixin (appearance_runtime.py) owns
# what happens on change (_on_control_changed, _apply_active_theme, saving,
# swatches, gallery). Both share `self`, so attribute names here (for example
# _glow_strength_slider) are a contract with that file: renaming one breaks the
# other without any import error.
class AppearancePanelMixin:
    """Mixin that builds the Theme tab and keeps its controls in sync with app state."""
    def _build_theme_tab(self) -> QWidget:
        """Build the Theme tab: scrollable controls on the left, palette swatches and a live preview card on the right."""
        w = QWidget()
        # Two columns: controls (stretch 3) and preview (stretch 2). The preview is kept
        # visible beside the controls, not on a separate tab, because the whole point of
        # this page is judging a change by looking at it.
        layout = QHBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        # Controls are wrapped in a scroll area so the tab stays usable on small screens.
        # Left Column: Controls (scrollable in case of small screen)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        # Zero margins here because ctrl_panel supplies its own 12 px padding; doubling
        # them would waste width in an already narrow column.
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        ctrl_panel = QWidget()
        ctrl_panel.setObjectName("panel")
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(12, 12, 12, 12)
        ctrl_layout.setSpacing(10)

        ctrl_layout.addWidget(_section("VISUAL THEME PRESETS"))

        # One entry per theme in THEMES; changing it re-applies the theme immediately.
        # Presets Combobox
        self._theme_preset_combo = QComboBox()
        # Populated from the registry so a theme added to THEMES appears here with no
        # UI change. Order is the dict insertion order, which puts the default first.
        for name in THEMES.keys():
            self._theme_preset_combo.addItem(name)
        # Connected after population: addItem on an empty combo would otherwise fire
        # currentTextChanged and apply a theme before the saved config is loaded.
        self._theme_preset_combo.currentTextChanged.connect(self._on_preset_changed)
        ctrl_layout.addWidget(_row("Preset Palette", self._theme_preset_combo))

        # Shows the selected preset's description text beneath the combo.
        # Preset Description
        self._preset_desc_lbl = QLabel("")
        self._preset_desc_lbl.setObjectName("lbl-muted")
        self._preset_desc_lbl.setWordWrap(True)
        # margin-left 130 px matches the fixed label column of _row (see common.py), so
        # the description lines up under the combo rather than under its caption.
        self._preset_desc_lbl.setStyleSheet("font-size: 8.5pt; margin-left: 130px; margin-bottom: 6px;")
        ctrl_layout.addWidget(self._preset_desc_lbl)

        ctrl_layout.addWidget(_hline())
        ctrl_layout.addWidget(_section("CUSTOM ACCENT OVERRIDE"))

        # A custom accent overrides the preset accent; the swatch previews the active colour.
        # Custom Accent Picker Row
        # Button text doubles as state display: it reads "Accent: #RRGGBB" while an
        # override is active (see _update_accent_button_text), so no extra label is needed.
        self._custom_accent_btn = QPushButton("Pick Custom Accent...")
        self._custom_accent_btn.clicked.connect(self._pick_custom_accent)
        
        # Reset clears the override to None, which means "use the preset accent"; it is
        # not the same as picking the preset's hex, because the preset accent can still
        # change with the theme mode (ember turns it orange).
        self._clear_accent_btn = QPushButton("Reset Accent")
        self._clear_accent_btn.clicked.connect(self._reset_custom_accent)
        
        self._accent_color_preview = QLabel()
        # 24 px square: the same height as a default button row, so it reads as part
        # of the control strip. Colour is set in _update_accent_button_text.
        self._accent_color_preview.setFixedSize(24, 24)
        self._accent_color_preview.setStyleSheet("border: 1px solid #35356E; border-radius: 4px;")

        accent_layout = QHBoxLayout()
        accent_layout.setSpacing(8)
        accent_layout.addWidget(self._custom_accent_btn, 1)
        accent_layout.addWidget(self._clear_accent_btn)
        accent_layout.addWidget(self._accent_color_preview)
        
        # The accent row is composed from a nested layout wrapped in a widget only
        # because _row() accepts a single widget, not a layout.
        accent_widget = QWidget()
        accent_widget_layout = QHBoxLayout(accent_widget)
        accent_widget_layout.setContentsMargins(0, 0, 0, 0)
        accent_widget_layout.addLayout(accent_layout)
        ctrl_layout.addWidget(_row("Custom Accent", accent_widget))

        ctrl_layout.addWidget(_hline())
        ctrl_layout.addWidget(_section("THEME GALLERY"))

        # Gallery cards are filled in later by _update_theme_gallery().
        self._theme_gallery = QScrollArea()
        self._theme_gallery.setWidgetResizable(True)
        self._theme_gallery.setFrameShape(QFrame.Shape.NoFrame)
        # 320 px shows roughly three gallery cards without scrolling, enough to compare
        # themes at a glance while leaving the layout/effects controls reachable.
        self._theme_gallery.setMinimumHeight(320)
        # Gallery contents are replaced wholesale by _update_theme_gallery, so the
        # inner widget/layout pair is kept as attributes for it to clear and refill.
        self._theme_gallery_content = QWidget()
        self._theme_gallery_layout = QVBoxLayout(self._theme_gallery_content)
        self._theme_gallery_layout.setContentsMargins(0, 0, 0, 0)
        self._theme_gallery_layout.setSpacing(8)
        self._theme_gallery.setWidget(self._theme_gallery_content)
        ctrl_layout.addWidget(self._theme_gallery)

        ctrl_layout.addWidget(_hline())
        ctrl_layout.addWidget(_section("LAYOUT DENSITY PRESETS"))

        # Layout density names are persisted as-is and interpreted by the runtime mixin.
        # Layout density selector
        self._layout_preset_combo = QComboBox()
        # Names are persisted verbatim in the UI config. get_theme_stylesheet maps
        # some of them to margin/spacing/font size (Compact Laptop, Wide Monitor
        # Command, Minimal Distraction); a name it does not know keeps the default
        # density, so adding a name here needs a consumer to give it any effect.
        self._layout_preset_combo.addItems([
            "Focused Workbench",
            "Research Lab",
            "Training Console",
            "Evaluation Wall",
            "Compact Laptop",
            "Wide Monitor Command",
            "Minimal Distraction",
            "Max Introspection",
            "Knowledge Heavy",
            "Prompt Engineering Lab"
        ])
        self._layout_preset_combo.currentTextChanged.connect(self._on_control_changed)
        ctrl_layout.addWidget(_row("Layout Density", self._layout_preset_combo))

        ctrl_layout.addWidget(_hline())
        ctrl_layout.addWidget(_section("EFFECTS & MOTION SYSTEM"))

        # Glow enabled checkbox
        # Every control below funnels through the same _on_control_changed slot, which
        # debounces (120 ms) before re-applying, so dragging a slider never recompiles
        # the stylesheet per tick.
        # Enabled/strength are separate controls: disabling keeps the chosen strength
        # so re-enabling restores it.
        self._glow_enabled_check = QCheckBox("Enable Glow Effects")
        self._glow_enabled_check.stateChanged.connect(self._on_control_changed)
        ctrl_layout.addWidget(_row("Edge Glow", self._glow_enabled_check))

        # Sliders use integer 0..20 steps; the value is divided by 10 to give a 0.0x-2.0x multiplier.
        # Glow Strength Slider
        self._glow_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._glow_strength_slider.setRange(0, 20)
        self._glow_strength_slider.setSingleStep(1)
        self._glow_strength_slider.setPageStep(2)
        # 10 = 1.0x, the neutral multiplier: glow looks exactly as the theme author
        # intended (themes carry their own glow_strength). 20 (2.0x) is the slider
        # ceiling, set by setRange above.
        self._glow_strength_slider.setValue(10)
        self._glow_strength_slider.valueChanged.connect(self._on_control_changed)
        
        self._glow_strength_val_lbl = QLabel("1.0x")
        # Fixed width stops the slider from resizing as the text changes between
        # "0.5x" and "1.5x" while dragging.
        self._glow_strength_val_lbl.setFixedWidth(40)
        self._glow_strength_val_lbl.setObjectName("lbl-muted")
        
        glow_row_widget = QWidget()
        gr_layout = QHBoxLayout(glow_row_widget)
        gr_layout.setContentsMargins(0, 0, 0, 0)
        gr_layout.setSpacing(8)
        gr_layout.addWidget(self._glow_strength_slider, 1)
        gr_layout.addWidget(self._glow_strength_val_lbl)
        ctrl_layout.addWidget(_row("Glow Strength", glow_row_widget))

        # Animation Intensity (Speed) Slider
        self._animation_intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self._animation_intensity_slider.setRange(0, 20)
        self._animation_intensity_slider.setSingleStep(1)
        self._animation_intensity_slider.setPageStep(2)
        # Same 0-20 tenths scale as the glow slider; 10 = normal speed. TracingPanel
        # reads this as animation_intensity from state (default 1.0).
        self._animation_intensity_slider.setValue(10)
        self._animation_intensity_slider.valueChanged.connect(self._on_control_changed)
        
        self._animation_intensity_val_lbl = QLabel("1.0x")
        self._animation_intensity_val_lbl.setFixedWidth(40)
        self._animation_intensity_val_lbl.setObjectName("lbl-muted")
        
        anim_row_widget = QWidget()
        ar_layout = QHBoxLayout(anim_row_widget)
        ar_layout.setContentsMargins(0, 0, 0, 0)
        ar_layout.setSpacing(8)
        ar_layout.addWidget(self._animation_intensity_slider, 1)
        ar_layout.addWidget(self._animation_intensity_val_lbl)
        ctrl_layout.addWidget(_row("Motion Speed", anim_row_widget))

        # Reduced motion checkbox (Disable active glowing animations)
        # The "_app" suffix distinguishes this from the checkbox of the same purpose on
        # the Defaults tab; _on_control_changed keeps the two checked-states equal so
        # either one can be used.
        self._reduced_motion_check_app = QCheckBox("Disable active glowing animations (Reduced Motion)")
        self._reduced_motion_check_app.stateChanged.connect(self._on_control_changed)
        ctrl_layout.addWidget(_row("Accessibility", self._reduced_motion_check_app))

        # Save persists the current selection through config_store (see appearance_runtime).
        # Save & Apply Button
        # Changes already apply live (debounced); Save only persists them, so cancelling
        # out of the tab without saving leaves the previous config on disk for next launch.
        apply_btn = QPushButton("Save Appearance Config")
        apply_btn.setObjectName("btn-primary")
        apply_btn.clicked.connect(self._save_appearance_config)
        ctrl_layout.addWidget(apply_btn)

        scroll_layout.addWidget(ctrl_panel)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        # Stretch factors 3:2 give controls ~60% of the width; the preview needs less
        # because it is a fixed-size mockup.
        layout.addWidget(scroll, 3)

        # Right column: shows the effect of the chosen settings without leaving the tab.
        # Right Column: Live Preview Card + Swatches
        preview_col = QWidget()
        pr_layout = QVBoxLayout(preview_col)
        pr_layout.setContentsMargins(0, 0, 0, 0)
        pr_layout.setSpacing(12)

        # Swatches Panel
        swatches_panel = QWidget()
        swatches_panel.setObjectName("panel")
        sp_layout = QVBoxLayout(swatches_panel)
        sp_layout.setContentsMargins(12, 12, 12, 12)
        sp_layout.setSpacing(8)
        sp_layout.addWidget(_section("ACTIVE PALETTE OVERVIEW"))
        
        # Filled by _update_swatches (runtime mixin): one small colour chip per palette
        # role, cleared and rebuilt whenever the theme changes.
        self._swatches_container = QWidget()
        self._swatches_layout = QHBoxLayout(self._swatches_container)
        self._swatches_layout.setContentsMargins(0, 4, 0, 4)
        self._swatches_layout.setSpacing(6)
        sp_layout.addWidget(self._swatches_container)
        pr_layout.addWidget(swatches_panel)

        # TracingPanel is the same glowing panel used elsewhere, so glow/motion changes are visible here.
        # Live Preview Panel using TracingPanel
        # Reuses the real TracingPanel so the glow/motion preview is the genuine
        # renderer, not an imitation; passing self.state lets it react to the same
        # glow_enabled / glow_strength / reduced_motion fields the sliders change.
        self._preview_card = TracingPanel(self.state)
        # Active = animated border, so motion and glow settings are visible immediately.
        self._preview_card.set_active(True)
        # 280 px leaves room for the title, button row, badge row and two chat bubbles
        # without clipping at the smallest density preset.
        self._preview_card.setMinimumHeight(280)
        
        card_layout = QVBoxLayout(self._preview_card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(10)
        
        self._add_preview_elements(card_layout)
        # Stretch 1 lets the preview card take the remaining column height under the
        # swatches.
        pr_layout.addWidget(self._preview_card, 1)

        layout.addWidget(preview_col, 2)

        # Populate controls from the saved UI config last, after every widget exists.
        # Load active configuration if file exists
        # Last on purpose: it sets values and then calls _on_preset_changed, which
        # reads widgets created above (gallery, swatches, preview).
        self._load_active_appearance_config()
        
        return w


    def _add_preview_elements(self, card_layout: QVBoxLayout):
        """Fill the preview card with sample buttons, status icons and chat bubbles."""
        # Built with the same object names the real UI uses (section-header, lbl-muted,
        # btn-primary), so the QSS applies to the mockup with no special casing.
        title = QLabel("LIVE PREVIEW")
        title.setObjectName("section-header")
        card_layout.addWidget(title)
        
        desc = QLabel("Realtime mockup card showing active styles.")
        desc.setObjectName("lbl-muted")
        desc.setStyleSheet("font-size: 8pt; margin-bottom: 2px;")
        card_layout.addWidget(desc)
        
        # Object names btn-primary / btn-secondary pick up the active QSS theme.
        # Primary & Secondary Buttons mockup
        btn_row = QWidget()
        br_layout = QHBoxLayout(btn_row)
        br_layout.setContentsMargins(0, 0, 0, 0)
        br_layout.setSpacing(8)
        
        btn_prim = QPushButton("Primary Button")
        btn_prim.setObjectName("btn-primary")
        # 26 px with reduced padding keeps the mockup compact enough for the 280 px card.
        btn_prim.setFixedHeight(26)
        btn_prim.setStyleSheet("font-size: 8.5pt; padding: 2px 10px;")
        
        btn_sec = QPushButton("Secondary Button")
        btn_sec.setObjectName("btn-secondary")
        btn_sec.setFixedHeight(26)
        btn_sec.setStyleSheet("font-size: 8.5pt; padding: 2px 10px;")
        
        br_layout.addWidget(btn_prim)
        br_layout.addWidget(btn_sec)
        card_layout.addWidget(btn_row)
        
        # Icons are custom painted and take the shared state so they follow palette changes.
        # Success & Error badges using custom painted icons
        badge_row = QWidget()
        bg_layout = QHBoxLayout(badge_row)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setSpacing(16)
        
        # Success badge
        succ_widget = QWidget()
        sw_layout = QHBoxLayout(succ_widget)
        sw_layout.setContentsMargins(0, 0, 0, 0)
        sw_layout.setSpacing(4)
        # Icons take the shared state and a colour role ("green"/"red") rather than a
        # hex value, so they repaint from the live palette when the theme changes.
        self._check_icon_preview = CheckIcon(self.state, color_role="green", size=14)
        lbl_succ = QLabel("Success Badge")
        # Hard-coded colours here (and on the error badge) are a visual mockup of the
        # status colours; the icons beside them follow the palette.
        lbl_succ.setStyleSheet("color: #00FFAA; font-size: 8.5pt;")
        sw_layout.addWidget(self._check_icon_preview)
        sw_layout.addWidget(lbl_succ)
        
        # Error badge
        err_widget = QWidget()
        ew_layout = QHBoxLayout(err_widget)
        ew_layout.setContentsMargins(0, 0, 0, 0)
        ew_layout.setSpacing(4)
        self._cross_icon_preview = CrossIcon(self.state, color_role="red", size=14)
        lbl_err = QLabel("Error Badge")
        lbl_err.setStyleSheet("color: #FF3366; font-size: 8.5pt;")
        ew_layout.addWidget(self._cross_icon_preview)
        ew_layout.addWidget(lbl_err)
        
        bg_layout.addWidget(succ_widget)
        bg_layout.addWidget(err_widget)
        card_layout.addWidget(badge_row)
        
        # Fake conversation, including a collapsed think block, to preview chat styling.
        # Chat bubbles mockup
        chat_mockup = QWidget()
        cm_layout = QVBoxLayout(chat_mockup)
        cm_layout.setContentsMargins(0, 0, 0, 0)
        cm_layout.setSpacing(6)
        
        # User bubble
        # Bubble backgrounds use low-alpha rgba so the active theme's card colour shows
        # through, which is how the real chat bubbles look.
        user_bubble = QLabel("How does local introspection work?")
        user_bubble.setStyleSheet(
            "background: rgba(31, 31, 61, 0.3); border: 1px solid rgba(53, 53, 110, 0.4); "
            "border-radius: 4px; padding: 6px; font-size: 8.5pt; color: #F0F5FF;"
        )
        cm_layout.addWidget(user_bubble)
        
        # Assistant bubble
        assistant_bubble = QLabel(
            "Analyzing trace stream...<br/>"
            "<span style='color: #A0AEC0; font-family: monospace; font-size: 8pt;'>"
            "&lt;think&gt;<br/>Reasoning tier 1: verifying model budget context... ok.<br/>&lt;/think&gt;</span><br/>"
            "Introspection processes generation traces in real-time."
        )
        assistant_bubble.setTextFormat(Qt.TextFormat.RichText)
        assistant_bubble.setStyleSheet(
            "background: rgba(13, 13, 27, 0.5); border: 1px solid rgba(31, 31, 61, 0.5); "
            "border-radius: 4px; padding: 6px; font-size: 8.5pt;"
        )
        cm_layout.addWidget(assistant_bubble)
        card_layout.addWidget(chat_mockup)
        
        card_layout.addStretch()


    def _load_active_appearance_config(self):
        """Load the persisted UI config into the controls without triggering change handlers."""
        # get_ui_config merges the file over UI_CONFIG_DEFAULTS and range-validates
        # known keys, so direct indexing cannot raise KeyError; theme_mode still uses
        # .get() as a belt-and-braces default for configs predating that key.
        config = config_store.get_ui_config()
        theme_preset = config["theme_preset"]
        custom_accent = config["custom_accent"]
        layout_preset = config["layout_preset"]
        reduced_motion = config["reduced_motion"]
        glow_enabled = config["glow_enabled"]
        animation_intensity = config["animation_intensity"]
        glow_strength = config["glow_strength"]
        theme_mode = config.get("theme_mode", "midnight")

        # Set the mode on state before touching widgets so dependent handlers see it.
        # State first: the preset handler at the bottom of this function re-applies
        # the theme and reads state.theme_mode.
        self.state.theme_mode = theme_mode
        if hasattr(self, "_theme_mode_combo"):
            self._theme_mode_combo.blockSignals(True)
            self._theme_mode_combo.setCurrentText(theme_mode)
            self._theme_mode_combo.blockSignals(False)

        # blockSignals around every setter prevents each programmatic change from re-applying the theme.
        self._theme_preset_combo.blockSignals(True)
        idx = self._theme_preset_combo.findText(theme_preset)
        if idx >= 0:
            self._theme_preset_combo.setCurrentIndex(idx)
        self._theme_preset_combo.blockSignals(False)

        self._layout_preset_combo.blockSignals(True)
        idx = self._layout_preset_combo.findText(layout_preset)
        if idx >= 0:
            self._layout_preset_combo.setCurrentIndex(idx)
        self._layout_preset_combo.blockSignals(False)

        self._glow_enabled_check.blockSignals(True)
        self._glow_enabled_check.setChecked(glow_enabled)
        self._glow_enabled_check.blockSignals(False)

        self._reduced_motion_check_app.blockSignals(True)
        self._reduced_motion_check_app.setChecked(reduced_motion)
        self._reduced_motion_check_app.blockSignals(False)

        self._glow_strength_slider.blockSignals(True)
        self._glow_strength_slider.setValue(int(glow_strength * 10))
        self._glow_strength_slider.blockSignals(False)
        self._glow_strength_val_lbl.setText(f"{glow_strength:.1f}x")

        self._animation_intensity_slider.blockSignals(True)
        self._animation_intensity_slider.setValue(int(animation_intensity * 10))
        self._animation_intensity_slider.blockSignals(False)
        self._animation_intensity_val_lbl.setText(f"{animation_intensity:.1f}x")

        # Remember the accent separately; None means use the preset accent.
        self._active_custom_accent = custom_accent
        
        # Now apply the whole config once with signals active again.
        # Initial updates
        self._update_accent_button_text()
        # Runs with signals live again so the theme, description, gallery and accent
        # preview are all rebuilt once from the freshly loaded values.
        self._on_preset_changed()


    # Counterpart to _load_active_appearance_config: that one reads the persisted
    # file at build time, this one reads the live AppState and is used when state
    # was changed elsewhere (for example a theme picked from the gallery or set by
    # the shell) so the controls never disagree with what is on screen.
    def _sync_appearance_controls_from_state(self):
        """Refresh the controls from the live app state (used when the tab is shown)."""
        # getattr with defaults because state may not yet have every appearance attribute set.
        theme_preset = getattr(self.state, "theme_preset", "Karl Obsidian Core")
        layout_preset = getattr(self.state, "layout_preset", "Focused Workbench")
        custom_accent = getattr(self.state, "custom_accent", None)
        reduced_motion = getattr(self.state, "reduced_motion", False)
        glow_enabled = getattr(self.state, "glow_enabled", True)
        # float() because older saved states may hold ints or strings.
        animation_intensity = float(getattr(self.state, "animation_intensity", 1.0))
        glow_strength = float(getattr(self.state, "glow_strength", 1.0))
        theme_mode = getattr(self.state, "theme_mode", "midnight")

        if hasattr(self, "_theme_mode_combo"):
            self._theme_mode_combo.blockSignals(True)
            self._theme_mode_combo.setCurrentText(theme_mode)
            self._theme_mode_combo.blockSignals(False)

        self._theme_preset_combo.blockSignals(True)
        idx = self._theme_preset_combo.findText(theme_preset)
        if idx >= 0:
            self._theme_preset_combo.setCurrentIndex(idx)
        self._theme_preset_combo.blockSignals(False)

        self._layout_preset_combo.blockSignals(True)
        idx = self._layout_preset_combo.findText(layout_preset)
        if idx >= 0:
            self._layout_preset_combo.setCurrentIndex(idx)
        self._layout_preset_combo.blockSignals(False)

        self._glow_enabled_check.blockSignals(True)
        self._glow_enabled_check.setChecked(glow_enabled)
        self._glow_enabled_check.blockSignals(False)

        self._reduced_motion_check_app.blockSignals(True)
        self._reduced_motion_check_app.setChecked(reduced_motion)
        self._reduced_motion_check_app.blockSignals(False)

        self._glow_strength_slider.blockSignals(True)
        self._glow_strength_slider.setValue(int(glow_strength * 10))
        self._glow_strength_slider.blockSignals(False)
        self._glow_strength_val_lbl.setText(f"{glow_strength:.1f}x")

        self._animation_intensity_slider.blockSignals(True)
        self._animation_intensity_slider.setValue(int(animation_intensity * 10))
        self._animation_intensity_slider.blockSignals(False)
        self._animation_intensity_val_lbl.setText(f"{animation_intensity:.1f}x")

        self._active_custom_accent = custom_accent
        self._update_accent_button_text()
        self._preset_desc_lbl.setText(THEMES.get(theme_preset, {}).get("description", ""))
        # Swatches and gallery are rebuilt last, after the combo/slider values they
        # depend on have been set.
        self._update_swatches()
        self._update_theme_gallery()


    def _update_accent_button_text(self):
        """Update the accent button label and preview swatch to show the override or the preset default."""
        if self._active_custom_accent:
            self._custom_accent_btn.setText(f"Accent: {self._active_custom_accent}")
            self._accent_color_preview.setStyleSheet(
                f"background-color: {self._active_custom_accent}; border: 1px solid #35356E; border-radius: 4px;"
            )
        else:
            self._custom_accent_btn.setText("Pick Custom Accent...")
            # No override: show what the preset (with current mode) would use, resolved
            # through get_theme_colors so mode tweaks such as ember's orange are reflected.
            colors = get_theme_colors(self.state)
            accent_default = colors.get("accent", "#00C2FF")
            self._accent_color_preview.setStyleSheet(
                f"background-color: {accent_default}; border: 1px solid #35356E; border-radius: 4px;"
            )


    def _on_preset_changed(self):
        """Handle a preset change: refresh description, accent preview, theme and gallery."""
        preset_name = self._theme_preset_combo.currentText()
        # THEMES.get with {} default: a stale preset name from an old config degrades to
        # an empty description instead of a KeyError.
        preset_data = THEMES.get(preset_name, {})
        self._preset_desc_lbl.setText(preset_data.get("description", ""))
        self._update_accent_button_text()
        self._apply_active_theme()
        self._update_theme_gallery()


    def _pick_custom_accent(self):
        """Open a colour dialog seeded with the current accent and apply the chosen colour."""
        # Seed with the active override if any, otherwise the preset's own accent.
        preset_name = self._theme_preset_combo.currentText()
        default_color = THEMES.get(preset_name, {}).get("accent", "#00C2FF")
        if self._active_custom_accent:
            default_color = self._active_custom_accent
            
        color = QColorDialog.getColor(QColor(default_color), self, "Select Custom Accent Color")
        # isValid is False when the user cancels the dialog.
        if color.isValid():
            # Stored upper-case "#RRGGBB" (QColor.name() default format), the exact shape
            # darken_hex_color / hex_to_rgba in themes.py parse.
            self._active_custom_accent = color.name().upper()
            self._update_accent_button_text()
            self._apply_active_theme()


    def _reset_custom_accent(self):
        """Clear the custom accent so the preset's own accent is used again."""
        self._active_custom_accent = None
        self._update_accent_button_text()
        self._apply_active_theme()


