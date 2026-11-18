# Program: karl-offline-lab - tracing_panel.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Provides TracingPanel, a QFrame that paints an animated neon outline whose motion follows the theme and motion settings.

"""Animated border-tracing panel widget."""
from PyQt6.QtWidgets import QFrame
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPainter, QPainterPath, QPen, QColor
from app.ui.themes import get_theme_colors

# ---------------------------------------------------------------------------
# Tracing panel
# ---------------------------------------------------------------------------
class TracingPanel(QFrame):
    """
    A customizable QFrame that draws an animated neon segment tracing along
    its border when set_active(True) is called. Adapts to theme-specific
    motion styles (normal, slow, pulse, static) and user speed configurations.
    """
    def __init__(self, state, parent=None):
        """Set up animation state (dash offset, pulse value) and the tick timer."""
        # _offset drives the travelling dash; _pulse_val/_pulse_direction drive the pulse style
        super().__init__(parent)
        self.state = state
        self._active = False
        self._offset = 0
        self._pulse_val = 0.6
        self._pulse_direction = 1
        self._accent_color = None
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        
        self.setObjectName("panel")
        self.setStyleSheet("border-radius: 4px;")

    def set_active(self, active: bool):
        """Turn the animated outline on or off and start or stop the timer to match."""
        self._active = active
        self._update_timer_state()
        self.update()

    def set_accent_color(self, hex_color: str):
        """Override the theme accent with a specific hex colour for the outline."""
        self._accent_color = hex_color
        self.update()

    def _update_timer_state(self):
        """Run the timer only when active, motion is allowed and the theme is not static."""
        is_reduced = getattr(self.state, "reduced_motion", False) if self.state else False
        colors = get_theme_colors(self.state) if self.state else {}
        motion_style = colors.get("motion_style", "normal")
        
        # Reduced motion and static themes get a still border instead of animation
        if self._active and not is_reduced and motion_style != "static":
            if not self._timer.isActive():
                self._timer.start(25)
        else:
            self._timer.stop()

    def _on_tick(self):
        """Advance one animation frame: pulse the opacity or move the dash along the border."""
        # animation_intensity scales how fast the effect moves
        intensity = getattr(self.state, "animation_intensity", 1.0) if self.state else 1.0
        colors = get_theme_colors(self.state) if self.state else {}
        motion_style = colors.get("motion_style", "normal")
        
        if motion_style == "pulse":
            # Pulse style: opacity bounces between 0.2 and 1.0
            step = 0.03 * intensity
            self._pulse_val += step * self._pulse_direction
            if self._pulse_val >= 1.0:
                self._pulse_val = 1.0
                self._pulse_direction = -1
            elif self._pulse_val <= 0.2:
                self._pulse_val = 0.2
                self._pulse_direction = 1
        else:
            # Default style: dashes travel along the border; "slow" uses a smaller step
            step = 2.0 * intensity
            if motion_style == "slow":
                step = 0.8 * intensity
            self._offset -= int(step)
            # Wrap the offset so it never grows without bound
            if self._offset <= -1000:
                self._offset = 0
        self.update()

    def update_style(self):
        """Re-evaluate the timer and repaint after a theme or setting change."""
        self._update_timer_state()
        self.update()

    def paintEvent(self, event):
        """Paint the outline in the style that matches the motion settings (still, pulse or travelling dash)."""
        super().paintEvent(event)
        # Draw the normal frame first, then the overlay only when active
        if not self._active:
            return
            
        is_reduced = getattr(self.state, "reduced_motion", False) if self.state else False
        colors = get_theme_colors(self.state) if self.state else {}
        motion_style = colors.get("motion_style", "normal")
        accent = self._accent_color or colors.get("accent", "#00C2FF")
        color = QColor(accent)
        
        # Antialiasing keeps the rounded outline smooth
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Rounded rectangle inset by 1px so the pen stays inside the widget
        path = QPainterPath()
        path.addRoundedRect(1.0, 1.0, self.width() - 2.0, self.height() - 2.0, 4.0, 4.0)
        
        # Still outline for reduced motion or static themes
        if is_reduced or motion_style == "static":
            pen = QPen(color, 1.5, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawPath(path)
        # Pulse: same outline with a changing alpha
        elif motion_style == "pulse":
            pulse_color = QColor(color)
            pulse_color.setAlphaF(self._pulse_val)
            pen = QPen(pulse_color, 1.5, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawPath(path)
        # Travelling dash: a 60px segment followed by a 180px gap, shifted by the offset
        else:
            pen = QPen(color, 1.5)
            pen.setDashPattern([60, 180])
            pen.setDashOffset(self._offset)
            painter.setPen(pen)
            painter.drawPath(path)
