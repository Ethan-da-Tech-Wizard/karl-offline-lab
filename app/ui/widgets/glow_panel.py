# Program: karl-offline-lab - glow_panel.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Provides GlowPanel, a QFrame that draws a configurable neon drop-shadow glow honoring theme and reduced-motion settings.

"""Glow-effect panel widget."""
from PyQt6.QtWidgets import QFrame, QGraphicsDropShadowEffect
from PyQt6.QtGui import QColor
from app.ui.themes import get_theme_colors

# ---------------------------------------------------------------------------
# Glow panel widget
# ---------------------------------------------------------------------------
class GlowPanel(QFrame):
    """
    A QFrame that displays a soft neon glow around its edges using
    QGraphicsDropShadowEffect. The glow is bypassed if reduced_motion is active.
    """
    def __init__(self, state, parent=None, glow_color=None, blur_radius=12):
        """Create the panel; glow_color overrides the theme accent and blur_radius sets the base glow size."""
        # state supplies glow/motion preferences and may be None
        super().__init__(parent)
        self.state = state
        self._glow_color = glow_color
        self._blur_radius = blur_radius
        self._effect = None
        
        # "panel" object name lets the theme stylesheet target this frame
        self.setObjectName("panel")
        self.setStyleSheet("border-radius: 4px;")
        self._apply_glow()

    def set_glow_color(self, hex_color: str):
        """Change the glow colour (a hex string) and redraw the effect."""
        self._glow_color = hex_color
        self._apply_glow()

    def _apply_glow(self):
        """Apply or remove the drop-shadow effect based on the current state settings."""
        # Read preferences defensively: missing attributes fall back to sensible defaults
        is_reduced = getattr(self.state, "reduced_motion", False) if self.state else False
        glow_enabled = getattr(self.state, "glow_enabled", True) if self.state else True
        glow_strength = getattr(self.state, "glow_strength", 1.0) if self.state else 1.0
        
        # Reduced motion, disabled glow or zero strength all mean "no effect"
        if is_reduced or not glow_enabled or glow_strength <= 0:
            self.setGraphicsEffect(None)
            self._effect = None
        else:
            # Pull theme colours so the default glow matches the accent colour
            colors = get_theme_colors(self.state) if self.state else {}
            color_to_use = self._glow_color or colors.get("accent", "#00C2FF")
            
            self._effect = QGraphicsDropShadowEffect(self)
            self._effect.setColor(QColor(color_to_use))
            
            # Combine the user's glow strength with the theme preset's own multiplier
            preset_glow = colors.get("glow_strength", 1.0) if (self.state and hasattr(self.state, "theme_preset")) else 1.0
            actual_blur = self._blur_radius * glow_strength * preset_glow
            
            # Keep the blur at least 1 so Qt always has a valid radius; zero offsets centre the glow
            self._effect.setBlurRadius(max(1.0, actual_blur))
            self._effect.setXOffset(0)
            self._effect.setYOffset(0)
            self.setGraphicsEffect(self._effect)
            
    def update_style(self):
        """Re-evaluate the glow after a theme or setting change."""
        self._apply_glow()
